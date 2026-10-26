"""Real MuJoCo planning regression; graph-only tests are explicitly separate."""
from dataclasses import replace
import copy

import mujoco
import numpy as np
import pytest

from src.planning.collision_checker import CollisionChecker
from src.planning.rrt_connect import RRTConnectPlanner,PlannerConfig,Node,Tree,reconstruct_path,clean_path,path_length
from scripts.phase2b_common import load_scenes,scene_checker
from scripts.phase2a_cases import configured_checker,park,external_distance


@pytest.fixture(scope='module')
def scenes():return {s['name']:s for s in load_scenes()['scenes']}


def test_start_equals_goal_single_node():
    c=CollisionChecker();park(c);r=RRTConnectPlanner(c,seed=42).plan(c.home,c.home)
    assert r.success and r.path==[c.home.tolist()] and r.path_length==0
    assert r.iterations==0 and r.final_validation_passed and r.raw_final_validation_passed


def test_unobstructed_connect_and_shortcut(scenes):
    s=scenes['A_clear'];c=scene_checker(s)
    r=RRTConnectPlanner(c,PlannerConfig(goal_bias=1.),42).plan(s['q_start'],s['q_goal'])
    assert r.success and r.final_validation_passed
    np.testing.assert_array_equal(r.path[0],s['q_start']);np.testing.assert_array_equal(r.path[-1],s['q_goal'])
    assert any(e['operation']=='CONNECT' and e['status']=='ADVANCED' for e in r.tree_events)
    assert r.path_length<=r.raw_path_length+1e-12
    for role,nodes in r.trees.items():
        for node in nodes[1:]:
            assert node['parent'] is not None
            assert np.linalg.norm(np.array(node['q'])-nodes[node['parent']]['q'])<=.35+1e-12
    assert r.state_valid_calls==sum(p['state_calls'] for p in r.phases.values())
    assert r.edge_valid_calls==sum(p['edge_calls'] for p in r.phases.values())
    assert r.geom_distance_calls==274*r.state_valid_calls


@pytest.mark.parametrize('name',['B_sphere','D_two_obstacles'])
def test_actual_detour_fine_checked(scenes,name):
    s=scenes[name];c=scene_checker(s)
    assert c.state_valid(s['q_start']).valid and c.state_valid(s['q_goal']).valid
    assert not c.edge_valid(s['q_start'],s['q_goal'],step_rad=.005).valid
    r=RRTConnectPlanner(c,seed=42).plan(s['q_start'],s['q_goal'])
    assert r.success,(name,r.failure_reason)
    assert r.minimum_sampled_clearance>.005
    assert r.final_validation_passed and r.raw_final_validation_passed
    assert r.raw_validation['step_rad']==r.smoothed_validation['step_rad']==.005
    assert any(e['tree']=='goal' and e['operation']=='EXTEND' for e in r.tree_events)
    verifier=scene_checker(s)
    for a,b in zip(r.path,r.path[1:]):assert verifier.edge_valid(a,b,step_rad=.005).valid
    assert r.path_length<=r.raw_path_length+1e-12
    assert all(e['edge_valid'] for e in r.shortcut_events if e['accepted'])
    np.testing.assert_array_equal(r.path[0],s['q_start']);np.testing.assert_array_equal(r.path[-1],s['q_goal'])


@pytest.mark.parametrize('name,reason',[('F_invalid_start','invalid_start:'),('F_invalid_goal','invalid_goal:'),('F_goal_limit','invalid_goal:joint_limit')])
def test_invalid_endpoints(scenes,name,reason):
    s=scenes[name];r=RRTConnectPlanner(scene_checker(s),seed=42).plan(s['q_start'],s['q_goal'])
    assert not r.success and r.failure_reason.startswith(reason) and not r.path


@pytest.mark.parametrize('q',[np.zeros(6),np.zeros(8),np.zeros((7,1)),np.full(7,np.nan),np.full(7,np.inf)])
def test_malformed_start_rejected(q):
    c=CollisionChecker();r=RRTConnectPlanner(c).plan(q,c.home)
    assert not r.success and r.failure_reason.startswith('invalid_start:') and not r.path


def test_graph_parent_trace_and_swapped_tree_direction():
    c=CollisionChecker();q=c.home.copy()
    def at(v):out=q.copy();out[0]=v;return out
    a=Tree('start',[Node(at(0),None),Node(at(.1),0),Node(at(.2),1)])
    b=Tree('goal',[Node(at(.3),None),Node(at(.2),0)])
    forward=reconstruct_path(a,2,b,1);swapped=reconstruct_path(b,1,a,2)
    np.testing.assert_array_equal(forward,swapped)
    np.testing.assert_array_equal(np.array(forward)[:,0],[0,.1,.2,.3])
    assert len(clean_path([q,q,q]))==1
    with pytest.raises(ValueError):reconstruct_path(a,1,b,1)
    a.nodes[1].parent=1
    with pytest.raises(ValueError):a.trace(1)


def test_iteration_and_time_budgets(scenes):
    s=scenes['A_clear'];c=scene_checker(s)
    r=RRTConnectPlanner(c,PlannerConfig(max_iterations=0)).plan(s['q_start'],s['q_goal'])
    assert not r.success and r.failure_reason=='iteration_budget'
    r=RRTConnectPlanner(c,PlannerConfig(max_planning_time_s=1e-9)).plan(s['q_start'],s['q_goal'])
    assert not r.success and r.failure_reason=='time_budget' and not r.path


def test_identical_seed_reproducible_without_budget_cutoff(scenes):
    s=scenes['B_sphere'];planner=RRTConnectPlanner(scene_checker(s),PlannerConfig(shortcut_attempts=8),seed=123)
    a=planner.plan(s['q_start'],s['q_goal']);b=planner.plan(s['q_start'],s['q_goal'])
    # Reproducibility includes a consistently rejected fine-grid shortcut.
    # A fixed seed is not a guarantee of successful planning.
    assert a.success == b.success and a.failure_reason == b.failure_reason
    assert a.raw_final_validation_passed and b.raw_final_validation_passed
    assert a.path==b.path and a.raw_path==b.raw_path and a.trees==b.trees
    assert a.state_valid_calls==b.state_valid_calls


def test_fine_validation_rejects_real_5mm_grazing_counterexample():
    # Original Phase2A zero-margin regression remains in test_phase2a_collision.
    # Here reproduce a narrow *5mm clearance violation*, keeping the planner margin.
    c=configured_checker(shapes={'obstacle_sphere':[.003]});park(c)
    mid=c.home.copy();mid[0]=.005;c.state_valid(mid)
    g=next(g for g in c.robot_geoms if c.name(g)=='panda_left_finger_collision_2')
    centre=c.data.geom_xpos[g].copy();direction=np.array([0.,0.,1.]);lo,hi=0.,.3
    for _ in range(45):
        off=(lo+hi)/2;c.set_obstacle_pose('obstacle_sphere',centre+direction*off)
        if external_distance(c,c.state_valid(mid))<.00499:lo=off
        else:hi=off
    c.set_obstacle_pose('obstacle_sphere',centre+direction*((lo+hi)/2))
    a,b=c.home.copy(),c.home.copy();a[0]=-.1;b[0]=.1
    assert c.edge_valid(a,b,step_rad=.02).valid
    assert not c.edge_valid(a,b,step_rad=.005).valid
    r=RRTConnectPlanner(c,PlannerConfig(max_extension_rad=.5,goal_bias=1.,shortcut_attempts=0),seed=42).plan(a,b)
    assert not r.success and r.failure_reason=='raw_fine_validation_failed'
    assert r.raw_path and not r.path and not r.final_validation_passed
    assert r.search_fine_discrepancy['first_invalid']['offending_geom_pairs']


def test_planning_does_not_modify_live_state(robot):
    m,d=robot.model,robot.data
    flag=mujoco.mjtState.mjSTATE_INTEGRATION;n=mujoco.mj_stateSize(m,flag)
    before=np.empty(n);after=np.empty(n);mujoco.mj_getState(m,d,before,flag)
    c=CollisionChecker(m);park(c);a=c.home.copy();b=a.copy();b[0]+=.1
    r=RRTConnectPlanner(c,PlannerConfig(goal_bias=1.),seed=42).plan(a,b)
    assert r.success
    mujoco.mj_getState(m,d,after,flag);np.testing.assert_array_equal(before,after)


def test_configuration_cannot_relax_margin_or_resolution():
    with pytest.raises(ValueError):PlannerConfig(search_step_rad=.1)
    with pytest.raises(ValueError):PlannerConfig(final_step_rad=.02)
    with pytest.raises(ValueError):PlannerConfig(max_planning_time_s=np.inf)
    with pytest.raises(ValueError):RRTConnectPlanner(CollisionChecker(options={'minimum_clearance_m':.001}))
