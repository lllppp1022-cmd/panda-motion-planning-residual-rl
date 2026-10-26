"""Persist real narrow-clearance rejection and bounded-failure evidence."""
from _bootstrap import destination,main_guard
import numpy as np
from src.common import ROOT,write_json
from src.planning.rrt_connect import RRTConnectPlanner,PlannerConfig
from scripts.phase2a_cases import configured_checker,park,external_distance,snapshot_scene
from scripts.phase2b_common import load_scenes,scene_checker


def main():
    dest=destination('phase2b_regression_evidence')
    c=configured_checker(shapes={'obstacle_sphere':[.003]});park(c)
    mid=c.home.copy();mid[0]=.005;c.state_valid(mid)
    g=next(g for g in c.robot_geoms if c.name(g)=='panda_left_finger_collision_2')
    centre=c.data.geom_xpos[g].copy();direction=np.array([0.,0.,1.]);lo,hi=0.,.3
    for _ in range(45):
        off=(lo+hi)/2;c.set_obstacle_pose('obstacle_sphere',centre+direction*off)
        if external_distance(c,c.state_valid(mid))<.00499:lo=off
        else:hi=off
    c.set_obstacle_pose('obstacle_sphere',centre+direction*((lo+hi)/2))
    start,goal=c.home.copy(),c.home.copy();start[0]=-.1;goal[0]=.1
    resolutions={str(step):c.edge_valid(start,goal,step_rad=step).to_dict() for step in [.02,.01,.005,.0025]}
    r=RRTConnectPlanner(c,PlannerConfig(max_extension_rad=.5,goal_bias=1.,shortcut_attempts=0),42).plan(start,goal)
    assert resolutions['0.02']['valid'] and not resolutions['0.005']['valid']
    assert not r.success and r.failure_reason=='raw_fine_validation_failed'
    write_json(dest/'narrow_clearance.json',dict(scene=snapshot_scene(c),q_start=start,q_goal=goal,
               clearance_m=.005,resolutions=resolutions,result=r.to_dict(),original_phase2a_counterexample_unchanged=True))
    s=next(s for s in load_scenes()['scenes'] if s['name']=='A_clear')
    for name,config in [('iteration_budget',PlannerConfig(max_iterations=0)),('time_budget',PlannerConfig(max_planning_time_s=1e-9))]:
        r=RRTConnectPlanner(scene_checker(s),config,42).plan(s['q_start'],s['q_goal'])
        assert not r.success and r.failure_reason==name
        write_json(dest/(name+'.json'),r.to_dict())
    write_json(ROOT/'results/phase2b_regression_latest.json',dict(run=dest))
    print(dest,flush=True)


if __name__=='__main__':main_guard(main)
