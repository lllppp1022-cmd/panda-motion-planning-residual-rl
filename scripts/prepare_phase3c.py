"""Freeze 3B and prior artifacts; create new physically distinct checked scenes."""
from _bootstrap import ROOT,destination,main_guard
import json
from pathlib import Path
from src.common import sha,write_json,write_csv
from src.rl.dependencies import activate
activate()
from src.experiments.phase3c import design
from src.planning.collision_checker import CollisionChecker
from src.planning.rrt_connect import RRTConnectPlanner,PlannerConfig
from src.control.uncertain_models import signature
from scripts.phase2a_cases import park,snapshot_scene

def main():
    dest=destination('phase3c_prepare');paths=set()
    for directory in ('src','scripts','configs','docs','tests','models'):
        for p in (ROOT/directory).rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts or 'phase3c' in p.name.lower() or p.name.startswith(('22_','23_','24_')):continue
            if directory=='models' and p.suffix!='.xml':continue
            paths.add(p)
    for p in (ROOT/'results').glob('*'):
        if p.is_file() and 'phase3c' not in p.name.lower():paths.add(p)
    acc=json.loads((ROOT/'results/phase3b_acceptance.json').read_text(encoding='utf-8'))
    for key in ('training_run','evaluation_validation_run','evaluation_final_run'):
        paths.update(p for p in Path(acc[key]).rglob('*') if p.is_file())
    oldprep=Path(json.loads((ROOT/'results/phase3b_prepare_latest.json').read_text())['run'])
    inherited=json.loads((oldprep/'history_sha256.json').read_text(encoding='utf-8'))
    paths.update(ROOT/p for p in inherited)
    manifest={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)};write_json(dest/'history_sha256.json',manifest)
    write_json(dest/'design_before_planning.json',design());write_json(ROOT/'results/phase3c_prepare_latest.json',dict(run=dest))
    plans=[]
    for layout in design()['layout_configs']:
        c=CollisionChecker(options={'minimum_clearance_m':.02});park(c);a,b=c.home.copy(),c.home.copy()
        a[0]=layout['start_joint1_rad'];b[0]=layout['goal_joint1_rad'];a[1]=b[1]=layout['joint2_rad'];q=a.copy();q[0]=layout['crossing_angle_rad'];c.state_valid(q)
        finger=next(g for g in c.robot_geoms if 'left_finger_fingertip_pad_collision_1' in c.name(g));c.set_obstacle_pose(layout['obstacle'],c.data.geom_xpos[finger].copy())
        scene=snapshot_scene(c)
        for seed in design()['planning_seeds']:
            r=RRTConnectPlanner(c,PlannerConfig(),seed).plan(a,b);source=dest/f"{layout['name']}_{seed}.json"
            write_json(source,dict(layout=layout,scene=scene,q_start=a,q_goal=b,planning_seed=seed,model_signature=signature(c.model),
                planning_clearance_m=.02,result=r.to_dict()))
            plans.append(dict(source_json=str(source),split=layout['split'],success=r.success,failure_reason=r.failure_reason))
            print('PLAN',layout['name'],seed,r.success,r.failure_reason,flush=True)
            if r.success and r.final_validation_passed:
                write_csv(source.with_suffix('.csv'),[dict(node=i,**{f'q{j+1}':v for j,v in enumerate(q)}) for i,q in enumerate(r.path)]);break
    write_json(dest/'plans.json',plans);print('PREPARED',dest,len(manifest),'HISTORICAL HASHES',flush=True)

if __name__=='__main__':main_guard(main)
