from pathlib import Path
import json
import numpy as np
from src.common import ROOT,write_json,write_csv,sha

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def prepared():return Path(read(ROOT/'results/phase3dc_prepare_latest.json')['run'])
def sources(split):return [Path(r['source']) for r in read(prepared()/'plans.json') if r['split']==split and r['success'] and r['final_validation_passed']]
def bc_checkpoint():return next(r['best_checkpoint'] for r in read(Path(read(ROOT/'results/phase3d_bc_training_latest.json')['run'])/'complete.json')['results'] if r['dimension']==85)
def code_hashes():return {str(p.relative_to(ROOT)):sha(p) for p in list((ROOT/'src/phase3dc').glob('*.py'))+[ROOT/'configs/phase3dc.json']}
def csvrows(path):
    import csv
    with Path(path).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def execute(source,condition,seed,method,dest,policy=None,action=None,record=False):
    from src.phase3dc.environment import DCEnv
    env=DCEnv([source],Path(dest),method=method,conditions=[condition],mode='evaluation',record=record,max_recorded=1)
    samples=[]
    try:
        obs,_=env.reset(seed=seed,options=dict(source=str(source),perturbation=condition,physical_seed=seed))
        while True:
            a=policy.predict(obs,deterministic=True)[0] if policy else np.asarray(action if action is not None else np.zeros(7),dtype=np.float32)
            env.intent=dict(raw=a.tolist(),clipped=a.tolist(),mean=a.tolist(),std=[0.]*7)
            obs,reward,t,tr,info=env.step(a)
            samples.append(dict(decision=len(samples)+1,reward=reward,**info))
            if t or tr:break
    finally:env.close()
    p=env.folder;r=dict(env.result);r.update(method=method,condition=condition,physical_seed=seed,run_directory=str(p),source=str(source),actual_physics_steps=env.total_physics_steps,
        recording_errors=env.record_errors,decision_count=len(samples))
    write_json(p/'dc_summary.json',r);write_csv(p/'dc_decisions.csv',samples)
    return r
