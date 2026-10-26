"""Pre-training gates: real full-episode equivalence and bounded nonzero actions."""
if __package__:
    from scripts._bootstrap import ROOT,destination,main_guard
else:
    from _bootstrap import ROOT,destination,main_guard
import argparse,csv,json
from pathlib import Path
import numpy as np
from src.common import write_json,sha
from src.rl.dependencies import activate
activate()
from src.rl.panda_residual_env import PandaResidualEnv,config
from scripts.run_phase3a import load_new,UncertainRun

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sources(split=None):
    prep=Path(read(ROOT/'results/phase3b_prepare_latest.json')['run'])
    return [x['source_json'] for x in read(prep/'plans.json') if x['success'] and (split is None or x['split']==split)]
def trace(path):
    with Path(path).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def run_episode(env,action_fn,seed=7001,options=None):
    obs,info=env.reset(seed=seed,options=options);steps=0;ret=0.;rows=[]
    while True:
        action=action_fn(obs,steps);obs,reward,terminated,truncated,info=env.step(action);steps+=1;ret+=reward
        rows.append(info)
        if terminated or truncated:break
    result=dict(**env.result,ppo_decisions=steps,episode_return=ret,decision_physics_steps=sum(x['physical_steps_this_decision'] for x in rows),
        reset_physics_steps=env.initial_physics_steps,mode=env.mode,residual_cap_rad=env.cap)
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['interface','equivalence','safety'],required=True);a=parser.parse_args()
    dest=destination('phase3b_'+a.stage);cfg=config();train=sources('train')[0]
    write_json(dest/'frozen_validation_config.json',dict(config=cfg,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src/rl').glob('*.py')}))
    if a.stage=='interface':
        from gymnasium.utils.env_checker import check_env
        from stable_baselines3.common.env_checker import check_env as sb3_check
        env=PandaResidualEnv([train],dest/'gym',conditions=[{'name':'nominal'}],mode='interface_check')
        try:check_env(env,skip_render_check=True);sb3_check(env,warn=True)
        finally:env.close()
        write_json(dest/'complete.json',dict(passed=True,gymnasium_check=True,sb3_check=True,observation_dimension=78,action_dimension=7,
            physical_steps=env.total_physics_steps,notes='checker episodes may be externally truncated; all archived'))
        write_json(ROOT/'results/phase3b_interface_latest.json',dict(run=dest));print('INTERFACE PASS',dest,flush=True)
    elif a.stage=='equivalence':
        results=[]
        for p in cfg['training_conditions']:
            base=UncertainRun(load_new(train),'M3',p,7001,dest/(p['name']+'_original_M3')).run()
            env=PandaResidualEnv([train],dest/(p['name']+'_zero_residual'),conditions=[p],mode='evaluation')
            try:r=run_episode(env,lambda o,i:np.zeros(7),options=dict(perturbation=p,physical_seed=7001))
            finally:env.close()
            old=trace(Path(base['run_directory'])/'tracking.csv');new=trace(Path(r['run_directory'])/'tracking.csv')
            fields=[f'{key}_{j}' for key in ('q_ref','dq_ref','q_actual','dq_actual','ctrl') for j in range(1,8)]+['time','minimum_clearance_m','tcp_error_m']
            equal=len(old)==len(new) and all(x[k]==y[k] for x,y in zip(old,new) for k in fields)
            results.append(dict(condition=p,physics_steps_old=len(old),physics_steps_zero=len(new),full_trace_bitwise_equal=equal,
                original_run=base['run_directory'],zero_run=r['run_directory'],old_success=base['success'],zero_success=r['success']))
            print('EQUIVALENCE',p['name'],equal,len(new),flush=True)
        result=dict(passed=all(r['full_trace_bitwise_equal'] and r['old_success']==r['zero_success'] for r in results),results=results,
            controller='Frozen Phase3A M3 versus full Gym bridge with exactly zero residual, every physics step including reset hold')
        write_json(dest/'complete.json',result);write_json(ROOT/'results/phase3b_zero_residual_equivalence.json',result)
        write_json(ROOT/'results/phase3b_equivalence_latest.json',dict(run=dest))
        if not result['passed']:raise RuntimeError('Zero residual equivalence gate failed')
    else:
        if not read(ROOT/'results/phase3b_zero_residual_equivalence.json')['passed']:raise ValueError('Equivalence first')
        results=[]
        # Fixed full trajectories, different speeds/states along each segment and original stopping response.
        for split in ('train','validation'):
            source=sources(split)[0]
            for p in cfg['training_conditions']:
                for pattern in cfg['safety_patterns']:
                    folder=dest/f"{split}_{p['name']}_{pattern}";env=PandaResidualEnv([source],folder,conditions=[p],mode='safety_probe')
                    def action(o,i):
                        if pattern=='positive':return np.ones(7)
                        if pattern=='negative':return -np.ones(7)
                        return np.ones(7)*(1 if i%2 else -1)
                    try:r=run_episode(env,action,options=dict(perturbation=p,physical_seed=7001))
                    finally:env.close()
                    r.update(pattern=pattern,split=split);results.append(r)
                    print('SAFETY',split,p['name'],pattern,r['success'],r['minimum_actual_clearance_m'],flush=True)
        # Explicit early close at nonzero speed verifies real limited hold rather than teleportation.
        env=PandaResidualEnv([train],dest/'midmotion_stop',conditions=[{'name':'nominal'}],mode='safety_probe')
        env.reset(seed=7001)
        for _ in range(50):env.step(np.ones(7))
        before=env.snapshot['row'];env.close();stop=env.result
        with (Path(stop['run_directory'])/'tracking.csv').open(encoding='utf-8-sig') as f:allrows=list(csv.DictReader(f))
        after=[x for x in allrows if float(x['time'])>before['time']]
        motion=max(abs(float(x[f'q_actual_{j}'])-before[f'q_actual_{j}']) for x in after for j in range(1,8))
        passed=all(r['success'] and not r['safety_violation_steps'] and not r['prohibited_contact_observations'] for r in results)
        result=dict(passed=passed,candidate_cap_rad=cfg['residual_cap_rad'],residual_slew_rad_s=cfg['residual_slew_rad_s'],results=results,stop_response=dict(result=stop,additional_physics_steps=len(after),
            maximum_additional_joint_motion_rad=motion),scope='ONLY tested full train/validation paths, nominal and mild gain, three bounded patterns; not universal safety',
            excluded='No final-test layout used; no narrow historical D claimed safe')
        write_json(dest/'complete.json',result);write_json(ROOT/'results/phase3b_action_safety_tests.json',result)
        write_json(ROOT/'results/phase3b_safety_latest.json',dict(run=dest))
        if not passed:raise RuntimeError('Nonzero action gate failed; do not train at this cap')
    print('COMPLETE',dest,flush=True)

if __name__=='__main__':main_guard(main)
