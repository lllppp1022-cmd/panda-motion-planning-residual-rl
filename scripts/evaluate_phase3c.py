"""Common-budget midgate, frozen validation, then heldout final diagnostics."""
from _bootstrap import ROOT,destination,main_guard
import argparse,concurrent.futures,json
from pathlib import Path
import numpy as np
from src.common import write_json,write_csv,sha
from src.experiments.phase3c import design
from scripts.phase3c_common import read,sources,evaluate_job

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['gate','validation','final'],required=True);a=parser.parse_args()
    dest=destination('phase3c_eval_'+a.stage);cfg=design();jobs=[]
    if a.stage=='gate':
        training=Path(read(ROOT/'results/phase3c_training_initial_latest.json')['run']);group=read(training/'complete.json')['results']
        for r in group:
            for p in cfg['evaluation_conditions'][1:]:jobs.append((sources('validation')[0],p,f"{r['version']}_seed{r['seed']}",r['checkpoint'],str(dest/f"{r['version']}_{r['seed']}_{p['name']}")))
    else:
        gate=read(ROOT/'results/phase3c_2048_gate.json')
        which='continue' if gate['all_four_passed'] else 'initial'
        training=Path(read(ROOT/f'results/phase3c_training_{which}_latest.json')['run']);group=read(training/'complete.json')['results']
        if a.stage=='final':
            validation=Path(read(ROOT/'results/phase3c_eval_validation_latest.json')['run'])
            if not (validation/'complete.json').exists():raise ValueError('Validate before opening heldout')
            write_json(dest/'final_test_opening.json',dict(checkpoints={f"{r['version']}_{r['seed']}":dict(path=r['checkpoint'],sha256=sha(r['checkpoint'])) for r in group},
                protocol_sha256=sha(ROOT/'configs/phase3c_pilot.json'),no_tuning_after_opening=True,validation_run=str(validation)))
        split='test' if a.stage=='final' else 'validation'
        for p in cfg['evaluation_conditions']:
            for method in ('M1','M3','Integral'):jobs.append((sources(split)[0],p,method,None,str(dest/f"{method}_{p['name']}")))
            for r in group:jobs.append((sources(split)[0],p,f"{r['version']}_seed{r['seed']}",r['checkpoint'],str(dest/f"{r['version']}_{r['seed']}_{p['name']}")))
    write_json(dest/'frozen_jobs.json',dict(jobs=jobs,config=cfg));results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
        for f in concurrent.futures.as_completed([pool.submit(evaluate_job,j) for j in jobs]):
            r=f.result();results.append(r);write_json(dest/'partial.json',results)
            print('EVAL',a.stage,len(results),len(jobs),r['method'],r['condition']['name'],r['success'],r['tcp_rmse_m'],flush=True)
    write_json(dest/'complete.json',dict(results=results,stage=a.stage,training_run=str(training)))
    write_json(ROOT/f'results/phase3c_eval_{a.stage}_latest.json',dict(run=dest))
    if a.stage=='gate':
        gates=[]
        for g in group:
            ev=[r for r in results if r['method']==f"{g['version']}_seed{g['seed']}"]
            mismatch=[r for r in g['completed_episodes'] if r['condition']!='nominal' and r['success']]
            safe=all(not r['safety_violation_steps'] and not r['prohibited_contact_observations'] and not r['errors'] for r in g['episode_results'])
            updated=all(u['actor_parameter_delta_l2']>0 and u['critic_parameter_delta_l2']>0 and np.isfinite(u['value_loss']) and np.isfinite(u['approx_kl']) for u in g['updates'])
            ev_ok=all(r['success'] and not r['safety_violation_steps'] and not r['prohibited_contact_observations'] and not r['errors'] for r in ev)
            gates.append(dict(version=g['version'],seed=g['seed'],training_safe=safe,networks_updated=updated,
                completed_mismatch_episodes=len(mismatch),validation_safe_and_complete=ev_ok,
                passed=bool(safe and updated and len(mismatch)>=2 and ev_ok)))
        decision=dict(all_four_passed=len(gates)==4 and all(g['passed'] for g in gates),groups=gates,training_run=str(training),evaluation_run=str(dest),
            common_budget='continue all to4096 only if all pass; otherwise all stop2048')
        write_json(ROOT/'results/phase3c_2048_gate.json',decision);print('GATE',decision,flush=True)
    print('COMPLETE',dest,flush=True)

if __name__=='__main__':main_guard(main)
