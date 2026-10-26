"""Build explicit static scenes from actual Panda FK; no inverse kinematics/planning."""
from _bootstrap import ROOT,main_guard,destination
import yaml
import numpy as np
from src.common import write_json
from src.planning.collision_checker import CollisionChecker
from scripts.phase2a_cases import park,snapshot_scene,restore_scene


def main():
    dest=destination('phase2b_scene_setup');c=CollisionChecker();park(c)
    home=c.home.copy();start=home.copy();goal=home.copy();start[0]=-.65;goal[0]=.65
    finger=next(g for g in c.robot_geoms if 'left_finger_fingertip_pad_collision_1' in c.name(g))
    cases=[]
    def pose(angle):
        q=home.copy();q[0]=angle;c.state_valid(q);return c.data.geom_xpos[finger].copy()
    def add(name,note,a=start,b=goal):
        scene=snapshot_scene(c)
        sa,sb=c.state_valid(a),c.state_valid(b)
        direct=c.edge_valid(a,b,step_rad=.005)
        item=dict(name=name,description=note,q_start=a.tolist(),q_goal=b.tolist(),scene=scene,
                  endpoint_validity=dict(start=sa.valid,goal=sb.valid),
                  direct_fine_valid=direct.valid,solvability='not_yet_confirmed' if not direct.valid else 'direct_fine_path_exists')
        cases.append(item)
        write_json(dest/f'{name}_direct_fine.json',direct.to_dict())
    add('A_clear','No nearby external obstacles; audited floor and self pairs still enabled')
    park(c);p=pose(-.18);c.set_obstacle_pose('obstacle_sphere',p)
    add('B_sphere','Sphere at actual fingertip FK position at joint1=-.18 rad')
    park(c);p=pose(.10);c.set_obstacle_pose('obstacle_cylinder',p)
    add('C_cylinder','Cylinder at actual fingertip FK position at joint1=.10 rad')
    park(c);p=pose(-.22);c.set_obstacle_pose('obstacle_sphere',p)
    p=pose(.22);c.set_obstacle_pose('obstacle_cylinder',p)
    add('D_two_obstacles','Sphere and cylinder at two distinct fingertip FK locations along the direct sweep')
    park(c);p=pose(0.);c.set_obstacle_pose('obstacle_box',p)
    add('E_box_detour','Box blocks the direct joint1 sweep; detour feasibility to be confirmed by fine validation')
    for case in cases:
        if not all(case['endpoint_validity'].values()):raise AssertionError((case['name'],case['endpoint_validity']))
        if case['name']!='A_clear' and case['direct_fine_valid']:raise AssertionError('Detour scene has no direct collision')
    park(c);c.state_valid(start)
    c.set_obstacle_pose('obstacle_sphere',c.data.geom_xpos[finger].copy())
    add('F_invalid_start','Place sphere in actual start fingertip collision geom centre')
    park(c);c.state_valid(goal)
    c.set_obstacle_pose('obstacle_sphere',c.data.geom_xpos[finger].copy())
    add('F_invalid_goal','Place sphere in actual goal fingertip collision geom centre')
    park(c);bad=goal.copy();bad[3]=c.robot.limits[3,1]+.01
    add('F_goal_limit','Goal joint4 exceeds actual MJCF upper hard limit by .01 rad',b=bad)
    if cases[-3]['endpoint_validity']!={'start':False,'goal':True} or cases[-2]['endpoint_validity']!={'start':True,'goal':False}:
        raise AssertionError('Failure endpoint fixtures must isolate start vs goal')
    payload=dict(training_or_dynamics=False,planning_seeds=[42,123,2026,7,11,19,31,73,101,314],
                 pilot_seeds=[42,123,2026],checker_options=dict(c.options),planner_config='configs/rrt_connect.yaml',
                 sampling='Seven arm MJCF limits; fixed Phase2A gripper; parked objects remain checked',scenes=cases)
    (ROOT/'configs/phase2b_scenes.yaml').write_text(yaml.safe_dump(payload,sort_keys=False,allow_unicode=True),encoding='utf-8')
    write_json(dest/'complete.json',payload)
    print([(x['name'],x['endpoint_validity'],x['direct_fine_valid']) for x in cases])


if __name__=='__main__':main_guard(main)
