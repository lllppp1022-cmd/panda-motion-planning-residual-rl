"""Actual data-source integrity, control equivalence and fault isolation regressions."""
from pathlib import Path
import inspect,time,ast
import numpy as np
import pytest
from src.common import ROOT,sha,write_json
from scripts.phase3c_common import read,rows,sources
from src.training_observer.replay import RecordedEpisode
from src.training_observer.observer import ObservedEnv,EpisodeCapture


def validation():return read(ROOT/'results/phase3d_recording_validation.json')


def test_historical_fidelity_audit_and_no_fabricated_rgb():
    p=Path(read(ROOT/'results/phase3d_history_latest.json')['run']);a=read(p/'complete.json')
    assert a['groups']==4 and a['episodes']==32
    assert all(not x['original_rgb'] and not x['full_qpos'] and x['max_tcp_reconstruction_error_m']<1e-12 for x in a['records'])


def test_cross_rollout_episode_policy_version_is_per_decision():
    p=ROOT/'results/runs/phase3c_training_continue_1792662132222765100/R1_seed42';d=RecordedEpisode(p,0)
    try:
        assert d.actions[2304]['policy_version']==8 and d.actions[2305]['policy_version']==9
        assert d.actions[2304]['end_step']+1==d.actions[2305]['start_step']
    finally:d.close()


def test_actual_observer_on_off_parameters_and_physics_identical():
    v=validation();e=v['observer_equivalence'];assert e['passed']
    assert e['update_steps_off']==e['update_steps_on']==[256,512,768,1024]
    assert e['decision_samples_exact'] and e['tracking_exact'] and e['network_parameters_bitwise_equal'] and e['optimizer_state_bitwise_equal']
    assert e['on']['physics_total']==e['off']['physics_total']==28791
    a=np.load(Path(e['off']['run'])/'parameters.npz');b=np.load(Path(e['on']['run'])/'parameters.npz')
    assert all(np.array_equal(a[k],b[k]) for k in a.files)


def test_rgb_validation_all_actual_frames_and_full_states():
    v=validation();video=v['video'];assert video['total_frames']==video['decoded_frames']==579
    assert video['full_state_and_control_match'] and video['sha256']==sha(video['published'])
    assert len(video['episodes'])==4 and all(e['all_frames_match_actual_logs'] and e['joint_displacement_rad']>.01 for e in video['episodes'])
    assert not v['actual_task_failures'] and v['budget_interrupted_episodes']==1


def test_full_state_archive_includes_both_finger_joints():
    p=Path(validation()['observer_equivalence']['on']['run'])/'actual_training/episode_0000/integration_states.npz'
    z=np.load(p);assert z['qpos'].shape==(7505,9) and z['qvel'].shape==(7505,9) and z['ctrl'].shape==(7505,8)
    assert len(z['states'])==len(z['physics_step']) and np.all(np.diff(z['time'])>0)


def test_actual_rgb_frame_time_and_caption_are_aligned():
    d=RecordedEpisode(validation()['observer_equivalence']['on']['run'],0)
    try:
        index=d.raw_frames[1]['physics_step']-1;im=d.image(index)
        assert d.displayed_index==index and im.shape==(832,960,3)
        d.image(index+1);assert d.displayed_index==index
        d.set_rgb_mode(False);d.set_state(index+1)
        assert np.array_equal(d.data.qpos,d.full['qpos'][index+1]) and 'FULL RECORDED STATE' in d.fidelity
    finally:d.close()


def test_recorder_exception_does_not_break_physics_or_step_count(monkeypatch):
    dest=ROOT/'results/runs'/f'phase3d_pytest_fault_{time.time_ns()}'
    def failure(self,row):raise RuntimeError('INTENTIONAL TEST CAPTURE FAILURE')
    monkeypatch.setattr(EpisodeCapture,'capture',failure)
    env=ObservedEnv(sources('train'),dest,mode='evaluation',record=True,max_recorded=0)
    try:
        env.reset(seed=42)
        for _ in range(2):env.step(np.zeros(7))
    finally:env.active_decision=0;env.close()
    assert env.total_physics_steps==1051 and len(env.record_errors)==1 and env.capture_failed
    assert not env.result['errors'] and env.result['failure_reason']=='rl_external_truncation'
    write_json(dest/'intentional_fault_test.json',dict(expected_recording_fault=env.record_errors,physics_unaffected=True,actual_steps=env.total_physics_steps))


def test_live_preview_file_lock_fix_is_independently_executed():
    p=Path(read(ROOT/'results/phase3d_preview_fix_latest.json')['run']);r=read(p/'complete.json')
    assert r['passed'] and not r['errors'] and r['optimizer_updates']==0 and r['interface_steps']==32
    assert len(list((p/'actual_training/live_frames').glob('*.png')))>=30


def test_capture_has_no_extra_step_reset_predict_or_random_sampling():
    tree=ast.parse(inspect.getsource(EpisodeCapture))
    calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert not any(x in calls for x in ('mj_step','predict','reset','normal','randn','default_rng'))


def test_all_independent_checkpoints_and_baselines_evaluated():
    p=Path(read(ROOT/'results/phase3d_evaluation_latest.json')['run']);r=read(p/'complete.json')['results']
    assert {x['method'] for x in r}=={'M3','Integral',*[f'update{i:02d}' for i in range(5)]}
    assert all(x['success'] and not x['safety_violation_steps'] for x in r)


def test_previous_artifacts_are_read_only():
    p=Path(read(ROOT/'results/phase3d_history_latest.json')['run']);manifest=read(p/'history_sha256.json')
    assert all(sha(ROOT/name)==digest for name,digest in manifest.items())
