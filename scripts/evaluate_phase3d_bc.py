"""Matched true closed-loop validation, then frozen independent final scenes."""
from _bootstrap import ROOT,destination,main_guard
from pathlib import Path
import argparse,concurrent.futures
from src.bc_residual.core import design
from src.bc_residual.runner import run_case
from scripts.phase3c_common import read,rows
from src.common import write_json,write_csv,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['validation','final'],required=True);a=p.parse_args();cfg=design();dest=destination('phase3d_bc_eval_'+a.stage)
    prep=Path(read(ROOT/'results/phase3d_bc_prepare_latest.json')['run']);training=Path(read(ROOT/'results/phase3d_bc_training_latest.json')['run']);models=read(training/'complete.json')['results']
    split='validation' if a.stage=='validation' else 'test';plans=[r for r in read(prep/'plans.json') if r['success'] and r['final_validation_passed'] and r['split']==split]
    checkpoints={f"BC{m['dimension']}":m['best_checkpoint'] for m in models};previous=[]
    if a.stage=='final':
        validation=read(Path(read(ROOT/'results/phase3d_bc_eval_validation_latest.json')['run'])/'complete.json')
        if any(not r['valid_model_and_execution'] or r['safety_violation_steps'] or r['prohibited_contact_observations'] for r in validation['results']):raise RuntimeError('Validation numerical/safety gate failed; final student execution not allowed')
        assert sha(ROOT/'configs/phase3d_bc.json')==read(training/'frozen_training_design.json')['config_sha256']
        write_json(dest/'final_test_opening.json',dict(config_sha256=sha(ROOT/'configs/phase3d_bc.json'),checkpoints={k:dict(path=p,sha256=sha(p)) for k,p in checkpoints.items()},
            validation_run=read(ROOT/'results/phase3d_bc_eval_validation_latest.json')['run'],no_further_training_or_selection=True,methods=cfg['methods'],
            controller_code_sha256={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src/bc_residual').glob('*.py')}))
    else:
        data=read(Path(read(ROOT/'results/phase3d_bc_data_complete_latest.json')['run'])/'complete.json')['results']
        previous=[r for r in data if any(p['source']==r['source'] for p in plans)]
    conditions=cfg['validation_conditions'] if split=='validation' else cfg['test_conditions'];seed=cfg['validation_physical_seed'] if split=='validation' else cfg['test_physical_seed'];jobs=[]
    for plan in plans:
        for c in conditions:
            for method in cfg['methods']:
                if any(r['source']==plan['source'] and r['condition']==c and r['method']==method and r['physical_seed']==seed for r in previous):continue
                jobs.append((plan['source'],c,seed,method,checkpoints.get(method),str(dest/f"{plan['scene']}_{c['name']}_{method}")))
    write_json(dest/'frozen_jobs.json',dict(jobs=jobs,config=cfg,reused_expert_runs=[r['run_directory'] for r in previous]));results=list(previous)
    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
        for f in concurrent.futures.as_completed([pool.submit(run_case,j) for j in jobs]):
            r=f.result();results.append(r);write_json(dest/'partial.json',results);print('CLOSED LOOP',a.stage,len(results),r['method'],r['scene'],r['condition']['name'],r['success'],r['tcp_rmse_m'],flush=True)
    write_json(dest/'complete.json',dict(results=results,stage=a.stage));write_json(ROOT/f'results/phase3d_bc_eval_{a.stage}_latest.json',dict(run=dest));print('COMPLETE',dest,flush=True)


if __name__=='__main__':main_guard(main)
