"""New deterministic diagnostics of three historical intermediate checkpoints."""
from _bootstrap import ROOT,destination,main_guard
import concurrent.futures
from pathlib import Path
from src.common import write_json,write_csv
from scripts.phase3c_common import read
from scripts.evaluate_phase3b import job

def main():
    dest=destination('phase3c_3b_updates');accepted=read(ROOT/'results/phase3b_acceptance.json');training=Path(accepted['training_run'])
    original=read(Path(accepted['evaluation_validation_run'])/'complete.json')['results'];results=[];jobs=[]
    for condition in ({'name':'nominal'},{'name':'mild_gain','kp_scale':.95}):
        source=next(r['source_json'] for r in original if r['scene']=='b_validation_cylinder')
        for method,update in (('PPO_initial',0),('M3_PPO',4)):
            r=next(r for r in original if r['scene']=='b_validation_cylinder' and r['condition']==condition and r['comparison_method']==method)
            results.append(dict(update=update,condition=condition,reused_frozen_actual_evaluation=True,result=r))
        for update in (1,2,3):jobs.append((source,condition,f'update{update}',str(training/f'checkpoints/ppo_after_update_{update:02d}.zip'),str(dest/f"{condition['name']}_update{update}")))
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
        future_jobs={pool.submit(job,j):j for j in jobs}
        for f in concurrent.futures.as_completed(future_jobs):
            j=future_jobs[f];r=f.result();results.append(dict(update=int(j[2][-1]),condition=j[1],reused_frozen_actual_evaluation=False,result=r));print('UPDATE EVAL',j[2],j[1]['name'],r['tcp_rmse_m'],flush=True)
    write_json(dest/'complete.json',dict(results=results,new_dynamic_episodes=6,cached_dynamic_episodes=4))
    write_json(ROOT/'results/phase3c_3b_updates_latest.json',dict(run=dest));print('COMPLETE',dest,flush=True)

if __name__=='__main__':main_guard(main)
