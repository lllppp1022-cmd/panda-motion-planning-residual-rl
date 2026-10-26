"""Shared bounded evaluation on frozen saved paths, no hidden planner calls."""
from pathlib import Path
import csv,json
import numpy as np
from src.common import ROOT,write_json
from src.rl.dependencies import activate
activate()
from src.experiments.phase3c import design,DiagnosticEnv
from scripts.run_phase3a import load_new,UncertainRun
from scripts.validate_phase3b import run_episode

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sources(split):
    p=Path(read(ROOT/'results/phase3c_prepare_latest.json')['run'])
    return [x['source_json'] for x in read(p/'plans.json') if x['success'] and x['split']==split]
def rows(p):
    with Path(p).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def evaluate_job(args):
    source,condition,method,checkpoint,dest=args;dest=Path(dest)
    import torch
    torch.set_num_threads(1)
    if method in ('M1','M3'):
        result=UncertainRun(load_new(source),method,condition,9001,dest).run()
    elif method=='Integral':
        from src.experiments.integral_control import IntegralM3Controller
        from src.control.trajectory_executor import CSVStream
        run=UncertainRun(load_new(source),'M3',condition,9001,dest);e=run.e
        e.controller=IntegralM3Controller(e.robot,run.nominal,condition,9001,lambda:bool(e.failure_reason),ki=design()['integral']['ki_per_s'])
        run.signals.close();stream=CSVStream(dest/'integral_control_signals.csv')
        def observe(ex,row):
            last=e.controller.last;signal=dict(time=row['time'],physics_step=row['physics_step'])
            for key in ('observed_q','residual_rad','residual_target_rad','delayed_target','base_target'):
                for j,value in enumerate(last[key]):signal[f'{key}_{j+1}']=float(value)
            for j,value in enumerate(e.data.actuator_force[e.robot.actuator_ids]):signal[f'actuator_force_{j+1}']=float(value)
            stream.append(signal)
        try:result=e.run(on_step=observe)
        finally:stream.close()
        result.update(antiwindup_steps=e.controller.antiwindup_steps,integral_clamp_steps=e.controller.integral_clamp_steps)
    else:
        from stable_baselines3 import PPO
        env=DiagnosticEnv([source],dest,mode='evaluation',version='R0')
        if checkpoint:policy=PPO.load(checkpoint,device='cpu');action=lambda o,i:policy.predict(o,deterministic=True)[0]
        elif method=='FeedbackProbe':action=lambda o,i:-np.clip(o[28:35]*.05/.002,-1,1)
        elif method=='CounterProbe':action=lambda o,i:np.clip(o[28:35]*.05/.002,-1,1)
        else:action=lambda o,i:np.zeros(7)
        try:result=run_episode(env,action,seed=9001,options=dict(perturbation=condition,physical_seed=9001))
        finally:env.close()
    t=rows(Path(result['run_directory'])/'tracking.csv');u=np.array([[float(x[f'ctrl_{j}']) for j in range(1,8)] for x in t]);contact=np.array([int(x['prohibited_contact_count'])>0 for x in t])
    result.update(method=method,condition=condition,source=source,checkpoint=checkpoint,
        command_smoothness_mean_squared_rate=float(np.mean((np.diff(u,axis=0)/.002)**2)),
        command_max_rate_rad_s=float(np.max(abs(np.diff(u,axis=0)/.002))),
        contact_intervals=int(np.sum(contact&~np.r_[False,contact[:-1]])),
        completion_progress=max(float(x['progress']) for x in t if x['phase']!='protective_hold'))
    signals=Path(result['run_directory'])/('integral_control_signals.csv' if method=='Integral' else 'control_signals.csv')
    s=rows(signals)
    result['maximum_actual_residual_rad']=max((abs(float(x[f'residual_rad_{j}'])) for x in s for j in range(1,8)),default=0) if s and 'residual_rad_1' in s[0] else 0.
    write_json(dest/'comparison_summary.json',result);return result
