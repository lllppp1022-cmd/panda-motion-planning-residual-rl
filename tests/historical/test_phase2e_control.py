"""Independent control tests; deliberately invalid models are isolated copies."""
import copy,json,time
from pathlib import Path
import numpy as np
import mujoco
import pytest
from src.common import ROOT,sha
from src.planning.collision_checker import CollisionChecker
from src.robot.panda_interface import PandaInterface
from src.robot.joint_controller import JointPositionController
from src.control.gravity_compensation import GravityCompensation
from src.control.compensated_joint_controller import CompensatedJointController
from src.control.predictive_safety_guard import PredictiveSafetyGuard
from src.control.phase2e_runtime import ControllerRun,config
from src.control.validated_path import load_verified_path


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def folder(label):
    p=ROOT/'results/runs'/f'phase2e_pytest_{label}_{time.time_ns()}';p.mkdir();return p
def ready_robot():
    c=CollisionChecker();c.state_valid(c.home);return c.robot


def test_pure_gravity_mapping_and_zero_velocity_query_does_not_mutate_live():
    robot=ready_robot();m,d=robot.model,robot.data;flag=mujoco.mjtState.mjSTATE_INTEGRATION
    n=mujoco.mj_stateSize(m,flag);before=np.empty(n);after=np.empty(n);d.qvel[:]=.1;mujoco.mj_getState(m,d,before,flag)
    gravity=GravityCompensation(m);tau=gravity.query(robot.positions(),d.qpos)
    mujoco.mj_getState(m,d,after,flag);np.testing.assert_array_equal(before,after)
    assert np.array_equal(gravity.data.qvel,np.zeros(m.nv))
    q=robot.positions();scratch=mujoco.MjData(m);mujoco.mj_copyData(scratch,m,d);scratch.qvel[:]=0
    scratch.ctrl[robot.actuator_ids]=q+tau/gravity.kp;mujoco.mj_forward(m,scratch)
    np.testing.assert_allclose(scratch.actuator_force[robot.actuator_ids],tau,atol=1e-9)
    assert tau[1]<0 and tau[3]>0


@pytest.mark.parametrize('field',['body_gravcomp','jnt_stiffness','actuator_gear','actuator_forcelimited'])
def test_unsupported_or_duplicate_compensation_rejected(field):
    robot=ready_robot();m=copy.copy(robot.model)
    if field=='body_gravcomp':m.body_gravcomp[3]=1.
    elif field=='jnt_stiffness':m.jnt_stiffness[3]=1.
    elif field=='actuator_gear':m.actuator_gear[0,0]=2.
    else:m.actuator_forcelimited[0]=False
    with pytest.raises((ValueError,AssertionError)):GravityCompensation(m)


def test_disabled_compensation_is_original_servo_and_switch_has_rate_limit():
    a=ready_robot();bdata=mujoco.MjData(a.model);mujoco.mj_copyData(bdata,a.model,a.data);b=PandaInterface(a.model,bdata)
    original=JointPositionController(a);comp=CompensatedJointController(b,enabled=False)
    target=a.positions();target[0]+=.02
    for _ in range(50):original.step(target);comp.step(target)
    np.testing.assert_array_equal(a.data.qpos,b.data.qpos);np.testing.assert_array_equal(a.data.qvel,b.data.qvel)
    comp.enabled=True;old=comp.command_q.copy();comp.step(target)
    assert np.max(np.abs(comp.command_q-old))<=.35*.002+1e-12


def test_compensation_position_offset_and_native_torque_limits():
    robot=ready_robot();control=CompensatedJointController(robot,max_offset_rad=.001)
    target=robot.positions()
    for _ in range(100):
        old=control.command_q.copy();control.step(target)
        assert np.max(abs(control.last['offset_rad']))<=.001
        assert np.max(abs(control.command_q-old))<=.0007+1e-12
        ids=robot.actuator_ids;force=robot.data.actuator_force[ids];bounds=robot.model.actuator_forcerange[ids]
        assert np.all(force>=bounds[:,0]-1e-10) and np.all(force<=bounds[:,1]+1e-10)
    assert control.last['compensation_clipped']
    before=robot.data.time
    with pytest.raises(ValueError):control.step(np.full(7,np.nan))
    assert robot.data.time==before


def test_shadow_rollout_adds_no_live_physics_steps_or_state_changes():
    run=ControllerRun(load_verified_path('A_clear'),'M1',.1,folder('shadow_isolation'),guard_mode='shadow')
    e=run.e;flag=mujoco.mjtState.mjSTATE_INTEGRATION;n=mujoco.mj_stateSize(e.model,flag)
    before=np.empty(n);after=np.empty(n);mujoco.mj_getState(e.model,e.data,before,flag)
    pred=run.guard.predict();mujoco.mj_getState(e.model,e.data,after,flag)
    np.testing.assert_array_equal(before,after);assert e.steps==0 and e.controller.physics_steps==0
    assert pred['live_physics_steps_added']==0 and not pred['warning'] and len(pred['samples'])==15


def test_hard_five_mm_threshold_and_m0_unchanged():
    b=load_verified_path('D_two_obstacles');e=ControllerRun(b,'M0',.15,folder('M0'))
    assert type(e.e.controller) is JointPositionController and e.guard is None
    assert e.e.checker.options['minimum_clearance_m']==.005
    with pytest.raises(ValueError):PredictiveSafetyGuard(e.e,{**config(),'execution_clearance_m':.004})


def test_static_pilot_improves_without_saturation_or_steady_oscillation():
    run=Path(read(ROOT/'results/phase2e_static_latest.json')['run']);assert read(run/'static_gate.json')['passed']
    for r in read(run/'complete.json')['results']:
        assert r['success'] and r['torque_saturation_steps']==0
        if r['method']=='M1':assert r['final_tcp_error_m']<1e-6


def test_shadow_gate_precedes_known_failure_and_checks_runtime():
    run=Path(read(ROOT/'results/phase2e_shadow_latest.json')['run']);gate=read(run/'shadow_gate.json')
    assert gate['passed'] and gate['max_joint_prediction_error_rad']<=1e-5 and gate['max_gap_prediction_error_m']<=.0001
    assert all(r['lead_s']>0 for r in gate['warning_leads'])


def test_historical_phase2d_artifacts_unchanged():
    from src.control.phase2e_provenance import verify_frozen_history
    audit=verify_frozen_history()
    assert audit['experimental_artifacts_all_unchanged'] and audit['test_assertions_unchanged']


def test_guard_prediction_fault_causes_archived_failure_not_silent_unprotected_run(monkeypatch):
    run=ControllerRun(load_verified_path('A_clear'),'M2',.1,folder('prediction_fault'))
    def fault():raise RuntimeError('intentional isolated predictor fault')
    monkeypatch.setattr(run.guard,'predict',fault)
    r=run.run();assert not r['success'] and r['failure_reason']=='predictive_guard_error'
    assert r['guard_fault'] and r['physics_steps']==r['logged_physics_steps']>0
    assert read(run.dest/'predictions.json')[0]['error']


def test_formal_cohort_contains_all_methods_and_preserves_non_success_outcomes():
    run=Path(read(ROOT/'results/phase2e_batch_latest.json')['run']);results=read(run/'complete.json')['results']
    assert len(results)==66
    assert all(sum(r['method']==m for r in results)==22 for m in ['M0','M1','M2'])
    for r in results:
        assert r['execution_clearance_m']==.005
        if r['outcome']=='protected_stop':assert not r['success']
        if r['success']:assert r['safety_violation_steps']==0 and r['contact_intervals']==0
    assert any(r['outcome']=='failure' for r in results) and any(r['outcome']=='protected_stop' for r in results)


def test_formal_raw_csv_metrics_and_torque_mapping_match():
    audit=read(ROOT/'results/phase2e_csv_audit.json')
    assert audit['actual_runs']==66 and audit['all_metrics_match']
    assert all(x['actual_signal_mapping_verified'] for x in audit['audits'])


def test_original_m0_reproduces_all_historical_conditions():
    audit=read(ROOT/'results/phase2e_m0_baseline_reproduction.json')
    assert audit['runs']==22 and audit['all_matched']


def test_both_actual_live_videos_decode_and_match_tracking():
    audit=read(ROOT/'results/phase2e_video_audit.json');assert audit['all_matched'] and len(audit['videos'])==2
    assert all(x['decoded_frames']==x['frames_matched_to_actual_tracking']>100 for x in audit['videos'])
