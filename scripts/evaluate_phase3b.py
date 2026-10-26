"""Frozen-model paired diagnostics; final layout only after validation is archived."""
if __package__:
    from scripts._bootstrap import ROOT,destination,main_guard
else:
    from _bootstrap import ROOT,destination,main_guard
import argparse,concurrent.futures,json
from pathlib import Path
import numpy as np
from src.common import write_json,write_csv,sha
from src.rl.dependencies import activate
activate()
import torch
from stable_baselines3 import PPO
from src.rl.panda_residual_env import PandaResidualEnv,config
from scripts.validate_phase3b import sources,read,run_episode,trace
from scripts.run_phase3a import UncertainRun,load_new

def job(args):
    source,condition,method,model_file,folder=args;folder=Path(folder);torch.set_num_threads(1)
    if method in ('M1','M3'):
        r=UncertainRun(load_new(source),method,condition,8001,folder).run();r['ppo_decisions']=0
    else:
        policy=PPO.load(model_file,device='cpu');env=PandaResidualEnv([source],folder,conditions=[condition],mode='evaluation')
        try:r=run_episode(env,lambda obs,i:policy.predict(obs,deterministic=True)[0],seed=8001,options=dict(perturbation=condition,physical_seed=8001))
        finally:env.close()
    r.update(comparison_method=method,condition=condition,source_json=source,model_sha256=sha(model_file) if model_file else None,
        sample_mode='deterministic_evaluation')
    rows=trace(Path(r['run_directory'])/'tracking.csv');ctrl=np.array([[float(x[f'ctrl_{j}']) for j in range(1,8)] for x in rows])
    r['command_max_rate_rad_s']=float(np.max(abs(np.diff(ctrl,axis=0)/.002)))
    r['command_rate_squared_mean']=float(np.mean((np.diff(ctrl,axis=0)/.002)**2))
    r['completion_progress']=max(float(x['progress']) for x in rows if x['phase']!='protective_hold')
    r['task_valid']=not bool(r['errors'])
    write_json(folder/'comparison_summary.json',r);return r

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['validation','final'],required=True);parser.add_argument('--workers',type=int,choices=[1,2,3],default=3)
    a=parser.parse_args();dest=destination('phase3b_eval_'+a.stage);cfg=config();training=Path(read(ROOT/'results/phase3b_training_latest.json')['run'])
    completed=read(training/'complete.json');final=completed['final_model'];initial=str(training/'checkpoints/ppo_initial_zero_mean.zip')
    if sha(final)!=completed['final_model_sha256']:raise ValueError('Frozen final checkpoint changed')
    if a.stage=='final':
        prior=Path(read(ROOT/'results/phase3b_eval_validation_latest.json')['run'])
        if not (prior/'complete.json').exists():raise ValueError('Validate before final opening')
        write_json(dest/'heldout_opening.json',dict(final_model_sha256=sha(final),validation_run=str(prior),
            no_further_training_or_tuning=True,test_source=sources('test')))
    jobs=[]
    splits=('train','validation') if a.stage=='validation' else ('test',)
    for split in splits:
        for p in cfg['training_conditions']:
            for method in (('M1','M3','PPO_initial','M3_PPO') if a.stage=='validation' else ('M1','M3','M3_PPO')):
                model_file=initial if method=='PPO_initial' else final if method=='M3_PPO' else None
                jobs.append((sources(split)[0],p,method,model_file,str(dest/f"{split}_{p['name']}_{method}")))
    if a.stage=='validation':
        for p in cfg['diagnostic_conditions'][2:]:
            for method in ('M1','M3','M3_PPO'):
                jobs.append((sources('validation')[0],p,method,final if method=='M3_PPO' else None,str(dest/f"validation_{p['name']}_{method}")))
    write_json(dest/'frozen_jobs.json',dict(jobs=jobs,checkpoint_sha256=sha(final),evaluation_seed=8001))
    results=[]
    def accept(r):
        results.append(r);write_json(dest/'partial.json',results)
        print('EVAL',len(results),len(jobs),r['scene'],r['condition']['name'],r['comparison_method'],r['success'],r['tcp_rmse_m'],flush=True)
    if a.workers==1:
        for args in jobs:accept(job(args))
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
            for future in concurrent.futures.as_completed([pool.submit(job,args) for args in jobs]):accept(future.result())
    write_json(dest/'complete.json',dict(results=results,stage=a.stage,final_model_sha256=sha(final)))
    write_json(ROOT/f'results/phase3b_eval_{a.stage}_latest.json',dict(run=dest));print('COMPLETE',dest,flush=True)

if __name__=='__main__':main_guard(main)
