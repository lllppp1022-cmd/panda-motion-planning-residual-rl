from _bootstrap import ROOT,destination,main_guard
from src.rl.dependencies import activate
activate()
from src.phase3dc.environment import design
from src.phase3dc.common import bc_checkpoint
from src.common import write_json,write_csv,sha
from src.planning.collision_checker import CollisionChecker
from src.planning.rrt_connect import RRTConnectPlanner,PlannerConfig
from src.control.uncertain_models import signature
from scripts.phase2a_cases import park,snapshot_scene

def main():
    d=destination('phase3dc_prepare');history={}
    for directory in ('src','scripts','configs','docs','tests','models','assets','results'):
        for p in (ROOT/directory).rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts or 'phase3dc' in str(p).lower() or p.name.startswith(('29_','30_')):continue
            if any(part.startswith('phase3d_viewer') for part in p.parts) or 'live_frames' in p.parts or p.name in ('live_status.json','live_frame.png','abort_training.request'):continue
            history[str(p.relative_to(ROOT))]=sha(p)
    write_json(d/'history_sha256.json',history);write_json(d/'frozen_protocol.json',dict(config=design(),bc_checkpoint=bc_checkpoint(),bc_sha256=sha(bc_checkpoint()),historical_files=len(history)))
    write_json(ROOT/'results/phase3dc_prepare_latest.json',dict(run=d));plans=[]
    for l in design()['layouts']:
        c=CollisionChecker(options={'minimum_clearance_m':.02});park(c);a=c.home.copy();b=a.copy();a[0]=l['start'];b[0]=l['goal'];a[1]=b[1]=l['q2'];cross=a.copy();cross[0]=l['cross'];c.state_valid(cross)
        finger=next(g for g in c.robot_geoms if 'left_finger_fingertip_pad_collision_1' in c.name(g));c.set_obstacle_pose(l['obstacle'],c.data.geom_xpos[finger].copy())
        for seed in design()['planning_seeds']:
            r=RRTConnectPlanner(c,PlannerConfig(),seed).plan(a,b);source=d/f"{l['name']}_{seed}.json"
            write_json(source,dict(layout=l,scene=snapshot_scene(c),q_start=a,q_goal=b,planning_seed=seed,model_signature=signature(c.model),planning_clearance_m=.02,result=r.to_dict()))
            if r.success:write_csv(source.with_suffix('.csv'),[dict(node=i,**{f'q{j+1}':v for j,v in enumerate(q)}) for i,q in enumerate(r.path)])
            plans.append(dict(source=str(source),split=l['split'],scene=l['name'],seed=seed,success=r.success,final_validation_passed=r.final_validation_passed,failure_reason=r.failure_reason))
            print('PLAN',plans[-1],flush=True)
            if r.success and r.final_validation_passed:break
    write_json(d/'plans.json',plans)
    assert sum(r['success'] and r['final_validation_passed'] for r in plans)==5
    write_json(d/'complete.json',dict(plans=plans,test_dynamics_not_opened=True))

if __name__=='__main__':main_guard(main)
