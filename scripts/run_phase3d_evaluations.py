"""Five independent deterministic policy-version evaluations plus strong baselines."""
from _bootstrap import ROOT,destination,main_guard
from pathlib import Path
import concurrent.futures
from scripts.phase3c_common import read,sources,evaluate_job
from src.common import write_json,sha


def main():
    pair=Path(read(ROOT/'results/phase3d_training_latest.json')['run']);on=Path(read(pair/'equivalence.json')['on']['run']);dest=destination('phase3d_evaluation')
    condition=dict(name='gain90',kp_scale=.9);jobs=[]
    for i in range(5):
        checkpoint=on/'checkpoints'/('initial.zip' if i==0 else f'ppo_after_update_{i:02d}.zip')
        jobs.append((sources('train')[0],condition,f'update{i:02d}',str(checkpoint),str(dest/f'update{i:02d}')))
    for method in ('M3','Integral'):jobs.append((sources('train')[0],condition,method,None,str(dest/method)))
    write_json(dest/'frozen_evaluation_jobs.json',dict(jobs=jobs,checkpoints={j[2]:sha(j[3]) for j in jobs if j[3]},
        source='INDEPENDENT DETERMINISTIC CHECKPOINT EVALUATION / NOT TRAINING',no_policy_updates=True))
    results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        for f in concurrent.futures.as_completed([pool.submit(evaluate_job,j) for j in jobs]):
            r=f.result();results.append(r);write_json(dest/'partial.json',results);print('EVALUATED',r['method'],r['success'],r['tcp_rmse_m'],flush=True)
    write_json(dest/'complete.json',dict(results=results));write_json(ROOT/'results/phase3d_evaluation_latest.json',dict(run=dest));print('COMPLETE',dest,flush=True)


if __name__=='__main__':main_guard(main)
