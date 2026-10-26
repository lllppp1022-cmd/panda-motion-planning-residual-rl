"""Freeze R1 ranking and independently validate limited integral/probe controllers."""
if __package__:
    from scripts._bootstrap import ROOT,destination,main_guard
else:
    from _bootstrap import ROOT,destination,main_guard
import concurrent.futures,json
import numpy as np
from src.common import write_json,write_csv,sha
from src.experiments.phase3c import design,reward_r1
from scripts.phase3c_common import sources,evaluate_job,read

def rankings():
    outcomes=[];gamma=.995
    for n in (160,327,600,750):
        for err in (0.,.0005,.002):
            def score(stop_at=None,success=True):
                rewards=[]
                for i in range(n if stop_at is None else stop_at):
                    end=i==(n-1 if stop_at is None else stop_at-1)
                    row=dict(progress=(i+1)/n,tcp_error_m=err,**{f'q_actual_{j}':err for j in range(1,8)},
                        **{f'q_ref_{j}':0. for j in range(1,8)},**{f'dq_actual_{j}':0. for j in range(1,8)},**{f'dq_ref_{j}':0. for j in range(1,8)})
                    r,_=reward_r1([row]*25,np.zeros(7),np.zeros(7),i/n,n*.05,end and success,end,False);rewards.append(r)
                return sum(rewards),sum(r*gamma**i for i,r in enumerate(rewards))
            completed=score();failed=[score(k,False) for k in (1,n//2,n)]
            assert all(completed[0]>f[0] and completed[1]>f[1] for f in failed)
            outcomes.append(dict(length_decisions=n,error=err,success_undiscounted=completed[0],success_discounted=completed[1],
                failure_discounted=[x[1] for x in failed],same_prefix_completion_beats_stopping=True))
    c=design()['R1'];worst_cost=c['joint_weight']+c['tcp_weight']+c['velocity_weight']+c['action_weight']+c['smooth_weight']+c['time_weight']
    assert c['failure_penalty']>worst_cost*(1+2.5/c['minimum_trajectory_duration_s'])
    return outcomes

def main():
    dest=destination('phase3c_validation');write_json(dest/'reward_rankings.json',rankings())
    jobs=[]
    for split in ('train','validation'):
        for p in design()['evaluation_conditions']:
            jobs.append((sources(split)[0],p,'Integral',None,str(dest/f"{split}_{p['name']}_Integral")))
        for method in ('M3','FeedbackProbe','CounterProbe'):
            jobs.append((sources(split)[0],design()['training_conditions'][1],method,None,str(dest/f'{split}_gain90_{method}')))
    write_json(dest/'frozen_jobs.json',dict(jobs=jobs,config=design()))
    results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
        for f in concurrent.futures.as_completed([pool.submit(evaluate_job,j) for j in jobs]):
            r=f.result();results.append(r);print('PROBE',r['scene'],r['condition']['name'],r['method'],r['success'],r['tcp_rmse_m'],flush=True)
    passed=all(r['success'] and not r['safety_violation_steps'] and not r['prohibited_contact_observations'] and not r['errors'] for r in results)
    write_json(dest/'complete.json',dict(passed=passed,results=results,integral_frozen=design()['integral'],R1=design()['R1'],
        scope='bounded nominal/gain/mass pilots, no final-test dynamics, no changes after opening heldout'))
    write_json(ROOT/'results/phase3c_validation_latest.json',dict(run=dest))
    if not passed:raise RuntimeError('Controller/probe gate failed, do not expand training')
    print('VALIDATED',dest,flush=True)

if __name__=='__main__':main_guard(main)
