import time,inspect,ast
from pathlib import Path
import numpy as np
import pytest
import mujoco
from src.common import ROOT,config
from src.robot.kinematics import Kinematics
from src.robot.inverse_kinematics import DampedLeastSquaresIK
from src.robot.joint_controller import JointPositionController
from src.scene.obstacles import ObstacleScene
from src.visualization.viewer import LiveViewer
def test_import_and_model_structure(model):
    assert mujoco.__version__ and (model.nq,model.nv,model.nu)==(9,9,8)
def test_joint_mapping(robot):
    m=robot.model
    assert len(set(robot.joint_ids))==len(set(robot.actuator_ids))==7
    for j,p,v,a in zip(robot.joint_ids,robot.qpos_indices,robot.dof_indices,robot.actuator_ids):
        assert m.jnt_qposadr[j]==p and m.jnt_dofadr[j]==v and m.actuator_trnid[a,0]==j
def test_servo_range_and_semantics(robot):
    m=robot.model;np.testing.assert_allclose(robot.limits,m.actuator_ctrlrange[robot.actuator_ids]);assert np.all(m.actuator_biasprm[robot.actuator_ids,1]<0)
    assert m.nu==8 and m.actuator_ctrlrange[7].tolist()==[0,255]
def test_single_joint_real_dynamics(robot):
    q0=robot.positions();goal=q0.copy();goal[0]+=.12;c=JointPositionController(robot)
    for _ in range(1500):c.step(goal)
    assert robot.data.time==pytest.approx(3.) and abs(robot.positions()[0]-q0[0])>.1
    assert np.max(np.abs(robot.positions()[1:]-q0[1:]))<.02
def test_all_seven_joint_motion_is_finite(robot):
    q0=robot.positions();goal=q0+np.array([-.1,-.12,.1,-.12,.1,.12,.1]);c=JointPositionController(robot)
    for _ in range(2000):c.step(goal)
    assert np.isfinite(robot.data.qpos).all() and np.isfinite(robot.data.qvel).all()
    assert np.min(np.abs(robot.positions()-q0))>.05
    assert np.max(np.abs(robot.positions()-goal))<.02
def test_slew_limit_and_limit_rejection(robot):
    c=JointPositionController(robot);q=robot.positions();goal=q+.1;before=c.command_q.copy();c.step(goal)
    assert np.max(np.abs(c.command_q-before))<=c.max_velocity*robot.model.opt.timestep+1e-12
    with pytest.raises(ValueError):c.step(np.full(7,100))
    with pytest.raises(ValueError):c.step(np.full(7,np.nan))
def test_dynamic_controller_has_no_qpos_assignment():
    tree=ast.parse(inspect.getsource(__import__('src.robot.joint_controller',fromlist=['JointPositionController'])))
    for node in ast.walk(tree):
        if isinstance(node,(ast.Assign,ast.AugAssign,ast.AnnAssign)):
            targets=node.targets if isinstance(node,ast.Assign) else [node.target]
            assert all('qpos' not in ast.unparse(t) and 'qvel' not in ast.unparse(t) for t in targets)
def test_fk_pose_and_base_frame(model):
    k=Kinematics(model);x,R=k.forward(k.home);assert np.isfinite(x).all();np.testing.assert_allclose(R.T@R,np.eye(3),atol=1e-12);assert np.linalg.det(R)==pytest.approx(1.)
    np.testing.assert_allclose(k.world_to_base(x),x,atol=1e-12)
@pytest.mark.parametrize('epsilon',[1e-5,1e-6,1e-7])
def test_jacobian_central_difference(model,epsilon):
    k=Kinematics(model);rng=np.random.default_rng(config()['sampling_seed'])
    for _ in range(5):
        q=k.home+rng.uniform(-.35,.35,7);j=k.compute_position_jacobian(q);fd=k.finite_difference(q,epsilon)
        assert j.shape==(3,7) and np.linalg.norm(j-fd)/np.linalg.norm(fd)<1e-2
def test_ik_twenty_known_reachable_and_live_state_isolation(robot):
    s=DampedLeastSquaresIK(robot.model);k=Kinematics(robot.model);rng=np.random.default_rng(config()['sampling_seed']);qpos=robot.data.qpos.copy();qvel=robot.data.qvel.copy();ctrl=robot.data.ctrl.copy()
    for _ in range(20):
        target=k.forward(k.home+rng.uniform(-.35,.35,7))[0];initial=k.home+rng.uniform(-.06,.06,7);r=s.solve(target,initial)
        assert r.success and r.status=='converged' and r.final_error_m<.001
        assert np.all(r.q>=robot.limits[:,0]) and np.all(r.q<=robot.limits[:,1])
    np.testing.assert_array_equal(robot.data.qpos,qpos);np.testing.assert_array_equal(robot.data.qvel,qvel);np.testing.assert_array_equal(robot.data.ctrl,ctrl)
def test_ik_failure_and_bad_input(model):
    s=DampedLeastSquaresIK(model);q=s.kin.home
    assert s.solve([10,10,10],q).status=='unreachable_bound'
    assert s.solve([np.nan,0,0],q).status=='invalid_target'
    target=s.kin.forward(q+np.array([.3,-.25,.3,-.25,.3,.3,.25]))[0];r=s.solve(target,q,max_iterations=1);assert not r.success and r.status=='max_iterations' and np.isfinite(r.q).all()
def test_dls_near_extended_configuration_is_finite(model):
    s=DampedLeastSquaresIK(model);q=np.array([0,0,0,-.15,0,.15,0]);target=s.kin.forward(q)[0]+[.01,.01,.01];r=s.solve(target,q,max_iterations=30)
    assert np.isfinite(r.q).all() and np.isfinite(r.final_error_m)
def test_ik_real_dynamic_servo(robot):
    k=Kinematics(robot.model);target=k.forward(k.home+np.array([.1,-.12,.1,-.12,.1,.12,.1]))[0];solution=DampedLeastSquaresIK(robot.model).solve(target,robot.positions());assert solution.success;c=JointPositionController(robot)
    errors=[]
    for _ in range(2500):c.step(solution.q);errors.append(np.linalg.norm(robot.tool_pose()[0]-target))
    assert max(errors[-250:])<.01
def test_named_collision_geoms_and_real_negative_fixture(robot):
    s=ObstacleScene(robot);original=robot.positions();a=s.query();assert a['external_contact_count']==0 and a['minimum']['signed_distance_m']>0
    g=next(g for g in s.robot_ids if 'link4' in s.name(g));center=robot.data.geom_xpos[g].copy();s.move('obstacle_sphere',center);b=s.query();assert b['minimum']['signed_distance_m']<0 and b['external_contact_count']>0
    s.move('obstacle_sphere',[1.1,.8,.5]);c=s.query();assert c['external_contact_count']==0 and c['minimum']['signed_distance_m']>0;np.testing.assert_array_equal(robot.positions(),original)
    minimum=c['minimum'];segment=np.array(minimum['nearest_surface_segment']);assert np.linalg.norm(segment[3:]-segment[:3])==pytest.approx(minimum['signed_distance_m'],abs=1e-7)
def test_geom_distance_sphere_analytic_reference(robot):
    m,d=robot.model,robot.data;a=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'target_visual');b=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'obstacle_sphere');seg=np.zeros(6)
    distance=mujoco.mj_geomDistance(m,d,a,b,2.,seg);expected=np.linalg.norm(d.geom_xpos[a]-d.geom_xpos[b])-m.geom_size[a,0]-m.geom_size[b,0]
    assert distance==pytest.approx(expected,abs=1e-9)
def test_windows_native_passive_viewer_opens_and_closes(robot):
    view=LiveViewer(robot,True,speed=4);c=JointPositionController(robot);target=robot.positions()+np.array([.03,0,0,0,0,0,0])
    try:
        for _ in range(100):
            with view.lock():c.step(target)
            view.sync()
        assert view.was_running
    finally:status=view.close()
    assert status['launched'] and status['closed']
