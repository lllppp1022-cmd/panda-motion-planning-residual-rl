from _bootstrap import destination,main_guard
import argparse,json,shutil
from pathlib import Path
from src.common import ROOT,sha,write_json,write_csv
from src.control.validated_path import load_verified_path
from src.control.trajectory_executor import TrajectoryExecutor,options


def main():
    p=argparse.ArgumentParser(description='Bounded actual servo experiments using saved Phase2B paths')
    p.add_argument('--stage',choices=['pilot','batch'],required=True)
    p.add_argument('--scenes',nargs='+',choices=['A_clear','B_sphere','C_cylinder','D_two_obstacles'],default=['A_clear','B_sphere','C_cylinder','D_two_obstacles'])
    a=p.parse_args();frozen=Path(json.loads((ROOT/'results/phase2c_freeze_latest.json').read_text())['run'])
    hashes=json.loads((frozen/'frozen_inputs.json').read_text())['hashes']
    if not all(sha(ROOT/f)==h for f,h in hashes.items()):raise RuntimeError('Frozen input changed')
    dest=destination('phase2c_'+a.stage);results=[];o=options()
    speeds=o['reference_speeds_rad_s'] if a.stage=='batch' else o['reference_speeds_rad_s'][:1]
    repeats=o['repeats'] if a.stage=='batch' else 1
    for scene in a.scenes:
        for speed in speeds:
            for repeat in range(repeats):
                run=dest/f'{scene}_v{speed:.2f}_repeat{repeat+1}'
                executor=TrajectoryExecutor(load_verified_path(scene,42),speed,run)
                r=executor.run();r['repeat']=repeat+1;results.append(r)
                write_csv(dest/'execution_summary.csv',results)
                print(a.stage,scene,speed,repeat+1,r['success'],r['failure_reason'],'sim',round(r['simulation_time_s'],3),
                      'gap_mm',None if r['minimum_actual_clearance_m'] is None else round(r['minimum_actual_clearance_m']*1000,4),
                      'goal_mm',round(r['final_tcp_error_m']*1000,4),flush=True)
    write_json(dest/'complete.json',dict(stage=a.stage,results=results))
    write_json(ROOT/f'results/phase2c_{a.stage}_latest.json',dict(run=dest))
    if a.stage=='batch':
        shutil.copy2(dest/'execution_summary.csv',ROOT/'results/phase2c_execution_summary.csv')
        for name in ['tracking','clearance']:
            out=ROOT/f'results/phase2c_{name}.csv';header=False
            with out.open('w',encoding='utf-8-sig',newline='') as w:
                for r in results:
                    file=Path(r['run_directory'])/(name+'.csv')
                    with file.open(encoding='utf-8-sig') as f:
                        first=f.readline()
                        if not header:w.write('scene,reference_speed_rad_s,repeat,'+first);header=True
                        for line in f:w.write(f"{r['scene']},{r['reference_speed_rad_s']},{r['repeat']},"+line)
    print('COMPLETE',dest,flush=True)


if __name__=='__main__':main_guard(main)
