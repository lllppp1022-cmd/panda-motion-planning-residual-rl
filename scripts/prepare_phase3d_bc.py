"""Freeze historical releases and create disjoint, independently fine-checked BC paths."""
from _bootstrap import ROOT,destination,main_guard
from pathlib import Path
from src.common import write_json,write_csv,sha
from src.bc_residual.core import design
from src.planning.collision_checker import CollisionChecker
from src.planning.rrt_connect import RRTConnectPlanner,PlannerConfig
from src.control.uncertain_models import signature
from scripts.phase2a_cases import park,snapshot_scene


def main():
    dest=destination('phase3d_bc_prepare');history={}
    for directory in ('src','scripts','configs','docs','tests','models','assets','results'):
        for p in (ROOT/directory).rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts or 'bc_residual' in p.parts or 'phase3d_bc' in str(p).lower() or p.name.startswith(('27_','28_')):continue
            if any(part.startswith('phase3d_viewer') for part in p.parts):continue
            if p.name in ('live_status.json','live_frame.png','abort_training.request') or 'live_frames' in p.parts:continue
            history[str(p.relative_to(ROOT))]=sha(p)
    write_json(dest/'history_sha256.json',history);write_json(dest/'frozen_design.json',design());write_json(ROOT/'results/phase3d_bc_prepare_latest.json',dict(run=dest))
    plans=[]
    for layout in design()['layouts']:
        c=CollisionChecker(options={'minimum_clearance_m':design()['planning_clearance_m']});park(c)
        a,b=c.home.copy(),c.home.copy();a[0]=layout['start'];b[0]=layout['goal'];a[1]=b[1]=layout['q2'];q=a.copy();q[0]=layout['cross'];c.state_valid(q)
        fingertip=next(g for g in c.robot_geoms if 'left_finger_fingertip_pad_collision_1' in c.name(g));c.set_obstacle_pose(layout['obstacle'],c.data.geom_xpos[fingertip].copy())
        for seed in design()['planning_seeds']:
            result=RRTConnectPlanner(c,PlannerConfig(),seed).plan(a,b);source=dest/f"{layout['name']}_{seed}.json"
            write_json(source,dict(layout=layout,scene=snapshot_scene(c),q_start=a,q_goal=b,planning_seed=seed,model_signature=signature(c.model),
                planning_clearance_m=design()['planning_clearance_m'],result=result.to_dict()))
            if result.success:write_csv(source.with_suffix('.csv'),[dict(node=i,**{f'q{j+1}':v for j,v in enumerate(q)}) for i,q in enumerate(result.path)])
            record=dict(source=str(source),scene=layout['name'],split=layout['split'],seed=seed,success=result.success,final_validation_passed=result.final_validation_passed,
                failure_reason=result.failure_reason,minimum_clearance_m=result.minimum_sampled_clearance);plans.append(record);print('PLAN',record,flush=True)
            if result.success and result.final_validation_passed:break
    write_json(dest/'plans.json',plans)
    if len([p for p in plans if p['success'] and p['final_validation_passed']])!=5:raise RuntimeError('Some new layouts not verified; no data collection permitted')
    write_json(dest/'complete.json',dict(plans=plans,historical_files=len(history),final_dynamics_not_opened=True));print('PREPARED',dest,flush=True)


if __name__=='__main__':main_guard(main)
