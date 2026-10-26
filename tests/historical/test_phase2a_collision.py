"""Real convex-geometry fixtures, isolation and discrete-edge regressions."""
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from src.common import ROOT
from src.planning.collision_checker import CollisionChecker, state_valid, edge_valid
from src.robot.joint_controller import JointPositionController
from scripts.phase2a_cases import build_cases, park, restore_scene, configured_checker
from scripts.phase2a_resolution import build_grazing_case


@pytest.fixture(scope='module')
def cases():
    from scripts._bootstrap import destination
    return {c['name']:c for c in build_cases(destination('phase2a_pytest_fixtures'))}


@pytest.fixture
def checker():
    c=CollisionChecker();park(c);return c


def test_clear_state_and_global_limits_not_teleop_envelope(checker):
    c=checker
    r=state_valid(c.home,c)
    assert r.valid and bool(r) and r.minimum_clearance > .005 and not r.minimum_is_lower_bound
    q=c.home.copy();q[0]+=.8
    assert state_valid(q,c).valid  # beyond home+.35 yet within MJCF


@pytest.mark.parametrize('j',range(7))
def test_each_real_joint_hard_limit(checker,j):
    q=checker.home.copy();q[j]=checker.robot.limits[j,1]+.001
    r=checker.state_valid(q)
    assert not r and not r.joint_limit_valid and r.rejection_reason=='joint_limit'
    assert r.distance_calls==0


@pytest.mark.parametrize('bad', [np.full(7,np.nan),np.full(7,np.inf),np.full(7,-np.inf),
                                np.zeros(6),np.zeros(8),np.zeros((7,1)),['bad']*7])
def test_bad_values_fail_closed_without_query(checker,bad):
    r=checker.state_valid(bad)
    assert not r.valid and not bool(r) and r.distance_calls==0 and r.minimum_clearance is None


@pytest.mark.parametrize('name',['B_sphere_collision','C_cylinder_collision','B2_box_collision','H_floor_collision'])
def test_external_shape_and_floor_collision(checker,cases,name):
    case=cases[name];restore_scene(checker,case['scene'])
    r=checker.state_valid(case['q'])
    assert not r.valid and r.external_collision
    assert any(p['category']=='external' and p['distance_m']<0 for p in r.offending_geom_pairs)
    assert any(p['checked_pair'] and p['distance_m']<0 for p in r.native_contacts)


def test_clearance_threshold_is_surface_distance(cases):
    case=cases['D_clearance_boundary']
    a=CollisionChecker();b=CollisionChecker(options={'minimum_clearance_m':.006})
    restore_scene(a,case['scene']);restore_scene(b,case['scene'])
    ra,rb=a.state_valid(case['q']),b.state_valid(case['q'])
    assert ra.valid and ra.minimum_clearance==pytest.approx(.0055,abs=1e-8)
    assert not rb.valid and rb.clearance_violation and not rb.external_collision
    assert rb.minimum_clearance==pytest.approx(ra.minimum_clearance,abs=1e-12)


def test_capped_distances_are_lower_bounds():
    c=CollisionChecker(options={'distance_query_max_m':.006})
    park(c);r=c.state_valid(c.home)
    assert r.valid and r.minimum_is_lower_bound and r.minimum_clearance==.006
    assert all(p['is_lower_bound'] for p in r.pair_results)
    with pytest.raises(ValueError):CollisionChecker(options={'distance_query_max_m':.005})


def test_self_filter_preserves_adjacent_and_nonadjacent(checker,cases):
    c=checker
    excluded={(p['body1'],p['body2']):p['exclusion_reason'] for p in c.excluded_pairs}
    assert excluded[('link7','hand')]=='fixed_weld_same_rigid_body'
    assert excluded[('link0','link1')]=='official_contact_exclusion'
    assert excluded[('hand','left_finger')]=='adjacent_welded_bodies_at_joint'
    assert any(p['body1']=='left_finger' and p['body2']=='right_finger' for p in c.pairs)
    assert any(p['body1']=='link2' and p['body2']=='link4' for p in c.pairs)
    r=c.state_valid(c.home)
    assert r.valid
    adjacent=[p for p in c.excluded_pairs if p['exclusion_reason']=='adjacent_welded_bodies_at_joint']
    assert any(mujoco.mj_geomDistance(c.model,c.data,p['id1'],p['id2'],.1,None)<0 for p in adjacent)
    case=cases['G_nonadjacent_self_collision'];restore_scene(c,case['scene'])
    r=c.state_valid(case['q'])
    assert not r.valid and r.self_collision
    rejected={(p['geom1'],p['geom2']) for p in r.offending_geom_pairs if p['category']=='self' and p['distance_m']<0}
    assert rejected
    assert any(p['checked_pair'] and ((p['geom1'],p['geom2']) in rejected or (p['geom2'],p['geom1']) in rejected)
               for p in r.native_contacts)


def test_valid_edge_and_all_sample_spacing(checker):
    c=checker;q=c.home.copy();q[0]+=.101
    r=edge_valid(c.home,q,c)
    assert r.valid and r.start_valid and r.end_valid
    assert r.state_calls==r.segments+1==7
    assert r.maximum_joint_step_rad<=.02
    assert len(r.samples)==7 and all(s['valid'] for s in r.samples)
    assert r.distance_calls==len(c.pairs)*r.state_calls


@pytest.mark.parametrize('which',['start','end','interior'])
def test_invalid_endpoints_and_interior(checker,cases,which):
    c=checker;case=cases['E_interior_collision'];restore_scene(c,case['scene'])
    a=case['q'] if which=='start' else case['q_start']
    b=case['q'] if which=='end' else case['q_end']
    r=c.edge_valid(a,b)
    assert not r.valid and r.first_invalid_q is not None and r.offending_geom_pairs
    assert r.state_calls==r.segments+1  # no early exit hides path minimum or endpoint status
    if which=='start':assert not r.start_valid and r.first_invalid_index==0
    elif which=='end':assert not r.end_valid
    else:assert r.start_valid and r.end_valid and 0<r.first_invalid_alpha<1


def test_edge_bad_resolution_inputs_budget_and_zero_length(checker):
    c=checker
    for step in (0,-.1,np.nan,np.inf):
        r=c.edge_valid(c.home,c.home,step_rad=step)
        assert not r.valid and r.state_calls==0
    assert not c.edge_valid([0]*6,c.home).valid
    q=c.home.copy();q[0]+=.1
    assert c.edge_valid(c.home,q,step_rad=1e-9).rejection_reason=='sample_budget_exceeded'
    assert c.edge_valid(c.home,c.home).valid


def test_resolution_sensitivity_with_real_grazing_collision():
    case,c=build_grazing_case()
    assert c is not None,'Fixture search did not find a real grazing case; do not fake it'
    coarse=c.edge_valid(case['q_start'],case['q_end'],step_rad=.02)
    fine=c.edge_valid(case['q_start'],case['q_end'],step_rad=.005)
    finer=c.edge_valid(case['q_start'],case['q_end'],step_rad=.00125)
    assert coarse.valid and not fine.valid and not finer.valid
    r=c.state_valid(fine.first_invalid_q)
    assert r.external_collision and any(p['distance_m']<0 for p in r.native_contacts if p['checked_pair'])


def test_live_data_and_model_are_untouched(robot):
    controller=JointPositionController(robot)
    controller.step(robot.positions()+.01)
    m,d=robot.model,robot.data
    flag=mujoco.mjtState.mjSTATE_INTEGRATION
    before=np.empty(mujoco.mj_stateSize(m,flag));after=before.copy()
    mujoco.mj_getState(m,d,before,flag)
    geom=m.geom_rgba.copy();masks=m.geom_contype.copy()
    c=CollisionChecker(m)
    assert c.model is not m and c.data is not d
    c.set_obstacle_pose('obstacle_sphere',[.4,.2,.5])
    c.state_valid(c.home);c.edge_valid(c.home,c.home+.01)
    mujoco.mj_getState(m,d,after,flag)
    np.testing.assert_array_equal(before,after)
    np.testing.assert_array_equal(m.geom_rgba,geom)
    np.testing.assert_array_equal(m.geom_contype,masks)


def test_repeated_input_resets_auxiliary_and_is_deterministic(checker,cases):
    c=checker
    before=c.state_valid(c.home).to_dict()
    c.state_valid(cases['G_nonadjacent_self_collision']['q'])
    c.data.qpos[c.fingers]=0;c.data.qvel[:]=42
    after=c.state_valid(c.home).to_dict()
    for key in ('elapsed_s','forward_s','distance_s'):
        before.pop(key);after.pop(key)
    assert before==after
    np.testing.assert_allclose(c.data.qpos[c.fingers],[.04,.04])
    assert c.data.time==0 and np.max(abs(c.data.qvel))==0


def test_invalid_scene_or_model_fails_closed(checker):
    c=checker;c.mocap_pos[0,0]=np.nan
    assert c.state_valid(c.home).rejection_reason.startswith('query_error:nonfinite_scene')
    c=CollisionChecker();c.model.geom_size[0,0]=np.nan
    assert 'nonfinite_model' in c.state_valid(c.home).rejection_reason
    c=CollisionChecker();c.model.geom_size[0,0]+=.1
    assert 'model_mutated' in c.state_valid(c.home).rejection_reason
    with pytest.raises(ValueError):c.set_obstacle_pose('obstacle_sphere',[0,np.nan,0])


def test_additional_planning_bounds_and_readonly_options(checker):
    c=checker;bounds=c.robot.limits.copy();bounds[0]=[-.1,.1]
    limited=CollisionChecker(options={'planning_joint_bounds_rad':bounds.tolist()})
    q=c.home.copy();q[0]=.2
    r=limited.state_valid(q)
    assert not r.valid and r.joint_limit_valid and r.rejection_reason=='planning_bounds'
    with pytest.raises(TypeError):limited.options['minimum_clearance_m']=0


def test_disabled_collision_masks_are_not_silently_accepted(robot):
    import copy
    model=copy.copy(robot.model)
    geom=next(i for i in range(model.ngeom) if model.geom_group[i]==3)
    model.geom_contype[geom]=0;model.geom_conaffinity[geom]=0
    with pytest.raises(ValueError):CollisionChecker(model)
    # A named external obstacle may not silently disappear when its masks are disabled.
    model=copy.copy(robot.model)
    geom=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'obstacle_sphere')
    model.geom_contype[geom]=0;model.geom_conaffinity[geom]=0
    with pytest.raises(ValueError):CollisionChecker(model)
