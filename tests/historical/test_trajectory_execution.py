"""Real dynamic tests, plus analytic trajectory and explicit rejection cases."""
import ast,json,time
from pathlib import Path
import numpy as np
import mujoco
import pytest
from src.common import ROOT,write_json
from src.control.trajectory_generator import QuinticTrajectory
from src.control.validated_path import load_verified_path
from src.control.trajectory_executor import TrajectoryExecutor


def dest(label):
    p=ROOT/'results/runs'/f'phase2c_pytest_{label}_{time.time_ns()}';p.mkdir();return p


def test_quintic_endpoints_and_rest_boundaries():
    q=np.array([[0]*7,[.2,-.1,0,.3,0,0,.1],[.1,0,0,.1,0,0,0]])
    t=QuinticTrajectory(q,.1,.3)
    for i,b in enumerate(t.boundaries):
        s=t.sample(b);np.testing.assert_allclose(s.q_ref,q[i],atol=1e-14)
        np.testing.assert_allclose(s.dq_ref,0,atol=1e-14);np.testing.assert_allclose(s.ddq_ref,0,atol=1e-14)
    np.testing.assert_allclose(t.sample(-1).q_ref,q[0]);np.testing.assert_allclose(t.sample(t.duration+1).q_ref,q[-1])


def test_reference_stays_on_segments_and_respects_limits():
    q=np.random.default_rng(42).uniform(-1,1,(6,7));t=QuinticTrajectory(q,[.1,.15,.12,.1,.14,.15,.1],.3)
    for x in np.linspace(0,t.duration,2001):
        s=t.sample(x);a,b=q[s.segment],q[s.segment+1]
        assert np.all(s.q_ref>=np.minimum(a,b)-1e-12) and np.all(s.q_ref<=np.maximum(a,b)+1e-12)
        delta=b-a;u=np.dot(s.q_ref-a,delta)/np.dot(delta,delta)
        np.testing.assert_allclose(s.q_ref,a+u*delta,atol=1e-12)
        assert np.all(np.abs(s.dq_ref)<=t.vmax+1e-12) and np.all(np.abs(s.ddq_ref)<=t.amax+1e-12)


@pytest.mark.parametrize('path',[[],np.zeros((2,6)),np.full((2,7),np.nan)])
def test_bad_paths_rejected(path):
    with pytest.raises(ValueError):QuinticTrajectory(path)


def test_stationary_single_node_trajectory():
    q=np.arange(7)*.1;t=QuinticTrajectory([q]);assert t.duration==0
    np.testing.assert_array_equal(t.sample(10).q_ref,q)


@pytest.mark.parametrize('scene',['A_clear','B_sphere'])
def test_real_dynamic_following_and_only_ctrl_step(scene,monkeypatch):
    b=load_verified_path(scene);e=TrajectoryExecutor(b,.1,dest(scene));original=mujoco.mj_step;steps=[]
    def observed(m,d,*args,**kwargs):
        before=d.time;original(m,d,*args,**kwargs);steps.append((before,d.time,d.ctrl[e.robot.actuator_ids].copy()))
    monkeypatch.setattr(mujoco,'mj_step',observed)
    r=e.run()
    assert r['success'],r
    assert r['physics_steps']==len(steps)==r['logged_physics_steps']
    assert all(abs(y-x-.002)<1e-10 for x,y,_ in steps)
    assert r['minimum_actual_clearance_m']>.005 and r['safety_violation_steps']==0
    assert r['final_joint_max_error_rad']<=.02 and r['final_tcp_error_m']<=.010
    assert r['simulation_time_s']>=e.trajectory.duration+1.5+.5-1e-8
    with pytest.raises(RuntimeError):e.initialize(b.path[0])


def test_unverified_saved_path_is_rejected():
    b=load_verified_path('A_clear');b.evidence['verified']=False
    with pytest.raises(ValueError):TrajectoryExecutor(b,.1,dest('unverified'))
    with pytest.raises((ValueError,FileNotFoundError)):load_verified_path('B_sphere',123)


def test_start_mismatch_cannot_teleport():
    b=load_verified_path('A_clear');q=b.path[0].copy();q[0]+=.1
    e=TrajectoryExecutor(b,.1,dest('start_mismatch'),initial_q=q);r=e.run()
    assert not r['success'] and r['failure_reason']=='initial_configuration_mismatch' and r['physics_steps']==0
    np.testing.assert_array_equal(e.robot.positions(),q)


def test_actual_live_safety_violation_stops_reference_and_records_inertia():
    # Negative fixture only: deliberately move a live sphere into the robot.
    # Never included in the frozen-scene experimental completion rate.
    b=load_verified_path('B_sphere');e=TrajectoryExecutor(b,.1,dest('negative_obstacle_injection'))
    injected=False
    def inject():
        nonlocal injected
        if e.steps>=100 and not injected:
            g=next(g for g in b.checker.robot_geoms if b.checker.name(g)=='panda_left_finger_collision_2')
            obstacle=mujoco.mj_name2id(e.model,mujoco.mjtObj.mjOBJ_GEOM,'obstacle_sphere')
            mid=e.model.body_mocapid[e.model.geom_bodyid[obstacle]]
            e.data.mocap_pos[mid]=e.data.geom_xpos[g];injected=True
    r=e.run(stop_requested=inject)
    assert not r['success'] and r['failure_reason']=='dynamic_safety_violation'
    assert r['first_violation']['monitor']['offending_geom_pairs']
    assert r['simulation_time_s']>=r['first_violation']['time']+.49
    assert r['safety_violation_steps']>0 and r['physics_steps']>100
    assert (e.dest/'clearance.csv').stat().st_size>0


def test_viewer_close_intent_archives_and_integrates_protective_hold():
    b=load_verified_path('A_clear');e=TrajectoryExecutor(b,.1,dest('close_intent'))
    r=e.run(stop_requested=lambda:'viewer_closed' if e.steps>=100 else None)
    assert not r['success'] and r['failure_reason']=='viewer_closed'
    assert r['physics_steps']>=350 and (e.dest/'execution_summary.json').exists()
    assert r['logged_physics_steps']==r['physics_steps']


def test_runtime_loop_has_no_live_qpos_assignment():
    tree=ast.parse((ROOT/'src/control/trajectory_executor.py').read_text())
    method=next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name=='run')
    assert not any(isinstance(x,(ast.Assign,ast.AugAssign)) and 'qpos' in ast.unparse(x).split('=')[0] for x in ast.walk(method))
    old=ast.parse((ROOT/'src/robot/joint_controller.py').read_text())
    assert not any(isinstance(x,(ast.Assign,ast.AugAssign)) and 'qpos' in ast.unparse(x).split('=')[0] for x in ast.walk(old))


def test_actual_recorded_dynamic_video():
    import imageio.v2 as imageio
    run=Path(json.loads((ROOT/'results/phase2c_demo_latest.json').read_text())['run'])
    meta=json.loads((run/'video_validation.json').read_text())
    assert meta['final_success'] and meta['all_frames_same_live_data'] and meta['frame_times_monotonic']
    assert meta['raw_unique_frames']>100 and meta['maximum_joint_displacement_rad']>1.
    with imageio.get_reader(str(ROOT/'assets/videos/phase2c_dynamic_execution.mp4')) as reader:
        assert sum(1 for _ in reader)==meta['decoded_frames']==meta['frames']


def test_actual_native_viewer_close_preserves_results():
    run=Path(json.loads((ROOT/'results/phase2c_closed_viewer_latest.json').read_text())['run'])
    result=json.loads((run/'demo_complete.json').read_text())
    assert result['viewer_launched'] and result['viewer_closed']
    assert result['execution']['failure_reason']=='viewer_closed'
    assert result['execution']['physics_steps']==result['execution']['logged_physics_steps']>0
    assert (run/'tracking.csv').stat().st_size>0 and (run/'clearance.csv').stat().st_size>0
