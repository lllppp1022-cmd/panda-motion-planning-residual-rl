"""Freeze prior work and generate new split-specific, fine-checked RRT paths."""
from _bootstrap import ROOT,destination,main_guard
import json,time
from pathlib import Path
from src.common import sha,write_json,write_csv
from src.rl.dependencies import provenance
from src.rl.panda_residual_env import config
from src.planning.collision_checker import CollisionChecker
from src.planning.rrt_connect import RRTConnectPlanner,PlannerConfig
from src.control.uncertain_models import signature
from scripts.phase2a_cases import park,snapshot_scene

def freeze(dest):
    paths=set()
    for directory in ('src','scripts','configs','docs','tests','models'):
        for p in (ROOT/directory).rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts:continue
            rel=str(p.relative_to(ROOT)).replace('\\','/')
            if '/rl/' in rel or 'phase3b' in p.name.lower() or p.name.startswith(('20_','21_')):continue
            if directory=='models' and p.suffix!='.xml':continue
            paths.add(p)
    for p in (ROOT/'results').glob('*'):
        if p.is_file() and ('phase3a' in p.name or 'phase2' in p.name):paths.add(p)
    accepted=json.loads((ROOT/'results/phase3a_acceptance.json').read_text(encoding='utf-8'))
    for key in ('prepare_run','pilot_run','test_run','video_run'):
        paths.update(p for p in Path(accepted[key]).rglob('*') if p.is_file())
    # Historical B/C/D paths are included through their prior provenance manifest,
    # plus all saved RRT path/source records explicitly.
    for pattern in ('phase2b_*','phase2d_planning_*'):
        for folder in (ROOT/'results/runs').glob(pattern):
            if folder.is_dir():paths.update(p for p in folder.rglob('*') if p.is_file() and p.suffix in ('.csv','.json','.yaml'))
    paths.update((ROOT/'assets/videos').glob('phase[23]*.mp4'))
    manifest={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}
    write_json(dest/'history_sha256.json',manifest);return len(manifest)

def main():
    dest=destination('phase3b_prepare');count=freeze(dest);cfg=config()
    write_json(dest/'preplanning_freeze.json',dict(config=cfg,prior_files=count,dependencies=provenance(),time_ns=time.time_ns()))
    plans=[]
    for layout in cfg['layouts']:
        c=CollisionChecker(options={'minimum_clearance_m':cfg['planning_clearance_m']});park(c)
        a,b=c.home.copy(),c.home.copy();a[0]=layout['start_joint1_rad'];b[0]=layout['goal_joint1_rad'];a[1]=b[1]=layout['joint2_rad']
        cross=a.copy();cross[0]=layout['crossing_angle_rad'];c.state_valid(cross)
        finger=next(g for g in c.robot_geoms if 'left_finger_fingertip_pad_collision_1' in c.name(g))
        c.set_obstacle_pose(layout['obstacle'],c.data.geom_xpos[finger].copy());scene=snapshot_scene(c)
        direct=c.edge_valid(a,b,step_rad=.005).to_dict()
        for seed in cfg['planning_seeds'][:cfg['maximum_planning_attempts_per_layout']]:
            r=RRTConnectPlanner(c,PlannerConfig(),seed).plan(a,b);source=dest/f"{layout['name']}_seed{seed}.json"
            write_json(source,dict(layout=layout,scene=scene,q_start=a,q_goal=b,planning_seed=seed,model_signature=signature(c.model),
                planning_clearance_m=cfg['planning_clearance_m'],checker_options=dict(c.options),direct=direct,result=r.to_dict()))
            plans.append(dict(layout=layout['name'],split=layout['split'],source_json=str(source),success=r.success,reason=r.failure_reason))
            print('PLAN',layout['name'],seed,r.success,r.failure_reason,flush=True)
            if r.success and r.final_validation_passed:
                write_csv(source.with_suffix('.csv'),[dict(node=i,**{f'q{j+1}':v for j,v in enumerate(q)}) for i,q in enumerate(r.path)]);break
    write_json(dest/'plans.json',plans);write_json(ROOT/'results/phase3b_prepare_latest.json',dict(run=dest));print('COMPLETE',dest,flush=True)

if __name__=='__main__':main_guard(main)
