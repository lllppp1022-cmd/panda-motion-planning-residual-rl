"""Replay saved, fine-validated joint paths using independent kinematic MjData."""
from _bootstrap import destination,main_guard
import argparse
import json
from pathlib import Path
import numpy as np
from src.common import ROOT,write_json,sha
from scripts.phase2b_common import scene_checker
from scripts.phase2b_visualization import render_demo


def main():
    p=argparse.ArgumentParser(description='KINEMATIC PLANNING VISUALIZATION / NOT DYNAMIC SERVO EXECUTION')
    p.add_argument('--viewer',action='store_true');p.add_argument('--record',action='store_true')
    p.add_argument('--scene',default='B_sphere');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--run',type=Path);p.add_argument('--fps',type=int,default=15)
    a=p.parse_args()
    if not 3<=a.fps<=30:raise ValueError('fps must be 3..30')
    if not (a.viewer or a.record):p.error('Choose --viewer and/or --record')
    run=a.run or Path(json.loads((ROOT/'results/phase2b_batch_latest.json').read_text())['run'])
    source=run/f'{a.scene}_seed{a.seed}.json'
    record=json.loads(source.read_text(encoding='utf-8'));r=record['result'];scene=record['scene_configuration']
    if not r['success'] or not r['final_validation_passed']:raise ValueError('Refuse to show a rejected path as successful')
    c=scene_checker(scene);start,goal=scene['q_start'],scene['q_goal']
    assert c.state_valid(start).valid and c.state_valid(goal).valid
    direct=c.edge_valid(start,goal,step_rad=.005)
    audit={}
    for name,path in [('raw',r['raw_path']),('shortcut',r['path'])]:
        audit[name]=[c.edge_valid(x,y,step_rad=.005).to_dict() for x,y in zip(path,path[1:])]
        if not all(e['valid'] for e in audit[name]):raise AssertionError('Saved path failed independent replay audit')
    frames=[]
    def hold(q,label,count=15):frames.extend(dict(q=q,scene=scene['scene'],label=label) for _ in range(count))
    hold(start,'START / valid');hold(goal,'GOAL / valid')
    if not direct.valid:
        hold(direct.first_invalid_q,'DIRECT LINE: FIRST INVALID 5mm CLEARANCE',20)
        contact=next((s for s in direct.samples if any(x['distance_m']<=0 for x in s['offending_geom_pairs'])),None)
        if contact:hold(contact['q'],'DIRECT LINE: ACTUAL INTERSECTION',20)
    for name,path in [('RAW RRT-CONNECT',r['raw_path']),('CHECKED SHORTCUT',r['path'])]:
        hold(path[0],name+' / start',10)
        for i,(x,y) in enumerate(zip(path,path[1:])):
            n=max(1,int(np.ceil(np.max(np.abs(np.array(y)-x))/.02)))
            for alpha in np.linspace(0,1,n+1)[1:]:
                q=(np.array(x)+(np.array(y)-x)*alpha).tolist()
                frames.append(dict(q=q,scene=scene['scene'],label=f'{name} / edge {i+1}/{len(path)-1}'))
        hold(path[-1],name+' / goal reached (kinematic)',15)
    if len(frames)>1200:raise ValueError('Bounded demonstration exceeded 1200 frames')
    dest=destination('phase2b_rrt_demo')
    write_json(dest/'source_and_validation.json',dict(source=source,source_sha256=sha(source),scene=scene,
                seed=a.seed,direct=direct.to_dict(),independent_replay_audit=audit,
                raw_path=r['raw_path'],shortcut_path=r['path'],dynamic_execution=False))
    result=render_demo(c,frames,dest,'phase2b_rrt_planning_demo',viewer=a.viewer,record=a.record,fps=a.fps)
    write_json(ROOT/'results/phase2b_demo_latest.json',dict(run=dest))
    print('OUTPUT',dest,result,flush=True)


if __name__=='__main__':main_guard(main)
