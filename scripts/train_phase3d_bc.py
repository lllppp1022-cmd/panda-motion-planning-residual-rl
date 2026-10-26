"""Supervised BC only; trajectory-level split and train-only normalization."""
from _bootstrap import ROOT,destination,main_guard
from pathlib import Path
import time,math
import numpy as np
from src.bc_residual.core import BCActor,design
import torch
from scripts.phase3c_common import read,rows
from src.common import write_json,write_csv,sha


def dataset(records,split):
    selected=[r for r in records if r['split']==split and r['quality_accepted']=='True']
    arrays=[np.load(r['samples_file']) for r in selected]
    x=np.concatenate([a['observations85'] for a in arrays]).astype(np.float32);y=np.concatenate([a['labels'] for a in arrays]).astype(np.float32)
    for a in arrays:a.close()
    if not np.isfinite(x).all() or not np.isfinite(y).all() or x.shape[1]!=85 or y.shape[1]!=7:raise ValueError('Invalid supervised arrays')
    assert np.array_equal(x[:,-7:],y),'85D integral feature must be independently causal target; inspect mismatch'
    return x,y,selected


def main():
    cfg=design();manifest=rows(ROOT/'results/phase3d_bc_dataset_manifest.csv');x,y,train=dataset(manifest,'train');v,z,val=dataset(manifest,'validation')
    assert len(train)>=16 and len(val)==4 and len(x)>=2000
    assert set(r['source'] for r in train).isdisjoint(r['source'] for r in val)
    dest=destination('phase3d_bc_training');torch.set_num_threads(1);logs=[];results=[]
    write_json(dest/'frozen_training_design.json',dict(config=cfg,config_sha256=sha(ROOT/'configs/phase3d_bc.json'),training_files=train,validation_files=val,test_data_opened=False,
        training_samples=len(x),validation_samples=len(v),normalization_from='accepted training trajectories ONLY',
        demonstration_physics_total=sum(int(r['actual_physics_steps']) for r in manifest)))
    for dimension in cfg['bc']['dimensions']:
        began=time.perf_counter();torch.manual_seed(cfg['seed']);rng=np.random.default_rng(cfg['seed']);model=BCActor(dimension)
        with torch.no_grad():model.net[-2].weight.zero_();model.net[-2].bias.zero_()
        mean=torch.as_tensor(x[:,:dimension].mean(axis=0));std=torch.as_tensor(np.maximum(x[:,:dimension].std(axis=0),cfg['bc']['normalization_std_floor']))
        xt=(torch.as_tensor(x[:,:dimension])-mean)/std;yt=torch.as_tensor(y);vt=(torch.as_tensor(v[:,:dimension])-mean)/std;zt=torch.as_tensor(z)
        optimizer=torch.optim.Adam(model.parameters(),lr=cfg['bc']['learning_rate']);best=float('inf');best_epoch=0;updates=0
        def save(name,epoch):
            path=dest/f'BC{dimension}_{name}.pt';torch.save(dict(dimension=dimension,model=model.state_dict(),mean=mean,std=std,epoch=epoch,
                seed=cfg['seed'],training_samples=len(x),validation_samples=len(v),normalization_files=[r['samples_file'] for r in train],config_sha256=sha(ROOT/'configs/phase3d_bc.json')),path);return path
        save('initial',0)
        for epoch in range(1,cfg['bc']['epochs']+1):
            model.train()
            for start in range(0,len(x),cfg['bc']['batch_size']):
                if start==0:order=rng.permutation(len(x))
                idx=order[start:start+cfg['bc']['batch_size']];loss=torch.mean((model(xt[idx])-yt[idx])**2)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite supervised loss')
                optimizer.zero_grad();loss.backward();optimizer.step();updates+=1
            model.eval()
            with torch.no_grad():train_loss=float(torch.mean((model(xt)-yt)**2));val_loss=float(torch.mean((model(vt)-zt)**2))
            if val_loss<best:best=val_loss;best_epoch=epoch;save('best',epoch)
            logs.append(dict(dimension=dimension,seed=cfg['seed'],epoch=epoch,training_mse=train_loss,validation_mse=val_loss,best_validation_mse=best,best_epoch=best_epoch,optimizer_steps=updates))
            if epoch%10==0:print('BC',dimension,epoch,train_loss,val_loss,flush=True);write_csv(dest/'training.csv',logs)
            if epoch-best_epoch>=cfg['bc']['patience']:break
        final=save('final',epoch);best_file=dest/f'BC{dimension}_best.pt'
        results.append(dict(dimension=dimension,seed=cfg['seed'],epochs=epoch,optimizer_steps=updates,best_epoch=best_epoch,best_validation_mse=best,
            final_training_mse=train_loss,final_validation_mse=val_loss,wall_s=time.perf_counter()-began,best_checkpoint=str(best_file),best_sha256=sha(best_file),final_checkpoint=str(final),final_sha256=sha(final)))
    write_csv(dest/'training.csv',logs);write_csv(ROOT/'results/phase3d_bc_training.csv',logs)
    write_json(dest/'complete.json',dict(results=results,training_samples=len(x),validation_samples=len(v),test_not_opened=True));write_json(ROOT/'results/phase3d_bc_training_latest.json',dict(run=dest));print('TRAINED',dest,flush=True)


if __name__=='__main__':main_guard(main)
