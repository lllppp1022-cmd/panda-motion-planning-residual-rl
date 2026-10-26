"""Bounded pilot4 then remaining24 expert episodes, no final test dynamics."""
from _bootstrap import ROOT,destination,main_guard
from pathlib import Path
import argparse,concurrent.futures
from src.bc_residual.core import design
from src.bc_residual.runner import run_case
from scripts.phase3c_common import read
from src.common import write_json,write_csv,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['pilot','complete'],required=True);a=p.parse_args();dest=destination('phase3d_bc_data_'+a.stage)
    prep=Path(read(ROOT/'results/phase3d_bc_prepare_latest.json')['run']);plans=[x for x in read(prep/'plans.json') if x['success'] and x['final_validation_passed']];cfg=design();jobs=[];previous=[]
    for plan in plans:
        split=plan['split']
        if split=='test':continue
        conditions=cfg['train_conditions'] if split=='train' else cfg['validation_conditions'];seeds=cfg['train_physical_seeds'] if split=='train' else [cfg['validation_physical_seed']]
        for c in conditions:
            for seed in seeds:
                name=f"{plan['scene']}_{c['name']}_{seed}"
                jobs.append((plan['source'],c,seed,'Integral500',None,str(dest/name)))
    if a.stage=='pilot':jobs=[j for j in jobs if 'bc_train_' in j[0] and j[1]['name'] in ('nominal','gain90') and j[2]==4101]
    else:
        last=Path(read(ROOT/'results/phase3d_bc_data_pilot_latest.json')['run']);previous=read(last/'complete.json')['results']
        if len(previous)!=4 or not all(r['dataset_quality_accepted'] for r in previous):raise RuntimeError('Expert pilot failed; full collection blocked')
        existing={(r['source'],r['condition']['name'],r['physical_seed']) for r in previous};jobs=[j for j in jobs if (j[0],j[1]['name'],j[2]) not in existing]
    write_json(dest/'frozen_jobs.json',dict(jobs=jobs,config=cfg,config_sha256=sha(ROOT/'configs/phase3d_bc.json')));results=list(previous)
    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
        for f in concurrent.futures.as_completed([pool.submit(run_case,j) for j in jobs]):
            r=f.result();results.append(r);write_json(dest/'partial.json',results);print('EXPERT',len(results),r['scene'],r['condition']['name'],r['success'],r['samples'],flush=True)
    write_json(dest/'complete.json',dict(results=results,stage=a.stage));write_json(ROOT/f'results/phase3d_bc_data_{a.stage}_latest.json',dict(run=dest))
    if a.stage=='complete':
        manifest=[]
        for r in results:
            plan=next(p for p in plans if p['source']==r['source'])
            manifest.append(dict(split=plan['split'],scene=r['scene'],condition=r['condition']['name'],physical_seed=r['physical_seed'],success=r['success'],quality_accepted=r['dataset_quality_accepted'],
                samples=r['samples'],actual_physics_steps=r['physics_steps'],samples_file=r['samples_file'],samples_sha256=sha(r['samples_file']),source=r['source'],source_sha256=sha(r['source']),
                minimum_clearance_m=r['minimum_actual_clearance_m'],failure_reason=r['failure_reason'],run_directory=r['run_directory']))
        write_csv(ROOT/'results/phase3d_bc_dataset_manifest.csv',manifest)
    print('COMPLETE',dest,flush=True)


if __name__=='__main__':main_guard(main)
