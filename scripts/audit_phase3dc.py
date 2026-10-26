from _bootstrap import ROOT,destination,main_guard
from src.rl.dependencies import activate
activate()
import numpy as np
import torch
from src.phase3dc.common import *
from src.phase3dc.environment import DCEnv,design
from src.phase3dc.policy import make_model
from src.bc_residual.core import Policy
from src.bc_residual.runner import run_case

def matrix(path):
    rr=csvrows(Path(path)/'tracking.csv');return np.array([[float(r[f'{key}_{j}']) for key in ('q_actual','ctrl') for j in range(1,8)] for r in rr])

def main():
    d=destination('phase3dc_interface');torch.set_num_threads(1);results=[];bc=Policy(bc_checkpoint())
    dummy=DCEnv(sources('train'),d/'dummy',method='A',mode='interface_check');m=make_model(dummy,42,bc_checkpoint(),'A')
    probe=np.concatenate([np.load(r['samples_file'])['observations85'] for r in csvrows(ROOT/'results/phase3d_bc_dataset_manifest.csv')])
    with torch.no_grad():expected=np.array([bc.predict(x) for x in probe]);actual=m.predict(probe,deterministic=True)[0]
    err=float(np.max(abs(expected-actual)));assert err<design()['interface_tolerance']['action_max_abs']
    m.save(str(d/'initialized_A'));from stable_baselines3 import PPO
    reloaded=PPO.load(d/'initialized_A.zip');assert np.array_equal(m.predict(probe,deterministic=True)[0],reloaded.predict(probe,deterministic=True)[0]);dummy.close()
    for i,condition in enumerate((design()['train_conditions'][1],design()['train_conditions'][2])):
        source=sources('train')[i];seed=4201+i
        for method,reference in [('A','BC85'),('B','Integral20')]:
            old=run_case((str(source),condition,seed,reference,bc_checkpoint() if method=='A' else None,str(d/f'old_{i}_{reference}')))
            new=execute(source,condition,seed,method,d/f'new_{i}_{reference}',policy=m if method=='A' else None)
            x=matrix(old['run_directory']);y=matrix(new['run_directory']);assert x.shape==y.shape
            diff=float(np.max(abs(x-y)));tol=design()['interface_tolerance']['joint_max_abs_rad'] if method=='A' else 0.
            assert diff<=tol,(method,diff)
            assert old['success'] and new['success'] and new['safety_violation_steps']==new['prohibited_contact_observations']==0
            results.append(dict(method=method,condition=condition,old=old,new=new,maximum_state_ctrl_difference=diff,bitwise_equivalent=bool(np.array_equal(x,y))))
    probes=[]
    for k,sign in enumerate((1.,-1.)):
        r=execute(sources('train')[0],design()['train_conditions'][1],5201,'B',d/f'probe_{k}',action=np.ones(7)*sign)
        control=csvrows(Path(r['run_directory'])/'dc_control.csv');active=control[750:]
        integral=np.array([json.loads(x['integral20_request']) for x in active]);target=np.array([json.loads(x['combined_target']) for x in active]);correction=np.array([json.loads(x['rl_correction_target']) for x in active])
        effective=target-integral;mask=abs(correction)>1e-12;fraction=float(np.mean(abs(effective[mask])>1e-8));assert fraction>.1
        assert r['success'] and r['safety_violation_steps']==r['prohibited_contact_observations']==0
        probes.append(dict(result=r,effective_fraction=fraction,clipped_fraction=float(np.mean(abs(effective[mask]-correction[mask])>1e-8))))
    write_json(d/'complete.json',dict(passed=True,action_max_abs_error=err,exact_reload=True,equivalence=results,nonzero_probes=probes,code_hashes=code_hashes()))
    write_json(ROOT/'results/phase3dc_interface_latest.json',dict(run=d));print('INTERFACE PASSED',err,flush=True)

if __name__=='__main__':main_guard(main)
