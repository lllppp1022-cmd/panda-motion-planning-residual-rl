"""Margin separation, real cohort, preserved baseline, and raw evidence tests."""
import csv,json,ast,time
from pathlib import Path
import numpy as np
import pytest
from src.common import ROOT,sha
from src.control.validated_path import load_verified_path
from src.control.trajectory_executor import TrajectoryExecutor
from src.planning.collision_checker import CollisionChecker
from src.planning.margin_study import MarginRequirements,checker_for,load_margin_path


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def planning():return Path(read(ROOT/'results/phase2d_planning_latest.json')['run'])
def dynamics():return read(Path(read(ROOT/'results/phase2d_dynamic_latest.json')['run'])/'complete.json')['results']


@pytest.mark.parametrize('margin',[.005,.008,.010,.015])
def test_independent_margins_and_unchanged_default(margin):
    req=MarginRequirements(margin);base=load_verified_path('D_two_obstacles')
    p=checker_for(base.checker,req.planning_clearance_m);e=checker_for(base.checker,req.execution_clearance_m)
    assert p.options['minimum_clearance_m']==margin and e.options['minimum_clearance_m']==.005
    assert CollisionChecker().options['minimum_clearance_m']==.005
    np.testing.assert_array_equal(p.mocap_pos,e.mocap_pos);np.testing.assert_array_equal(p.mocap_quat,e.mocap_quat)
    assert p.pairs==e.pairs and p.excluded_pairs==e.excluded_pairs


@pytest.mark.parametrize('kwargs',[{'planning_clearance_m':.004},{'planning_clearance_m':np.nan},{'planning_clearance_m':.1},{'execution_clearance_m':.008}])
def test_margin_requirements_fail_closed(kwargs):
    with pytest.raises(ValueError):MarginRequirements(**kwargs)


def test_all_saved_paths_have_complete_current_fine_audits():
    rows=read(planning()/'complete.json')['results'];assert len(rows)==12
    for r in rows:
        if not r['success']:continue
        b=load_margin_path(r['source_json']);assert b.evidence['verified']
        assert all(e['valid'] and e['maximum_joint_step_rad']<=.005+1e-12 for e in b.evidence['current_edges'])
        assert all(e['minimum_clearance']>r['planning_clearance_m'] for e in b.evidence['current_edges'])
        assert b.checker.options['minimum_clearance_m']==.005


def test_unvalidated_path_and_tampered_csv_rejected():
    # Restricted Windows temp root is not part of the project; preserve each
    # deliberately invalid fixture in an independent writable project folder.
    tmp_path=ROOT/'results/runs'/f'phase2d_tampered_fixture_{time.time_ns()}'
    tmp_path.mkdir()
    source=planning()/'margin08mm_seed42.json';r=read(source)
    r['result']['final_validation_passed']=False;invalid=tmp_path/'invalid.json'
    invalid.write_text(json.dumps(r),encoding='utf-8')
    with pytest.raises(ValueError,match='Rejected'):load_margin_path(invalid)
    r=read(source);tampered=tmp_path/'tampered.json';tampered.write_text(json.dumps(r),encoding='utf-8')
    (tmp_path/'tampered_shortcut_path.csv').write_text('node,q1\n0,nan\n')
    with pytest.raises(ValueError,match='hash mismatch'):load_margin_path(tampered)


def test_baseline_path_and_dynamic_failure_reproduced():
    old=load_verified_path('D_two_obstacles',42);new=load_margin_path(planning()/'margin05mm_seed42.json')
    np.testing.assert_array_equal(old.path,new.path)
    batch=Path(read(ROOT/'results/phase2c_batch_latest.json')['run'])
    historical=read(batch/'complete.json')['results']
    for r in dynamics():
        if r['planning_clearance_m']!=.005 or r['planning_seed']!=42:continue
        prior=next(x for x in historical if x['scene']=='D_two_obstacles' and x['reference_speed_rad_s']==r['reference_speed_rad_s'])
        assert not r['success'] and r['failure_reason']==prior['failure_reason']=='dynamic_safety_violation'
        np.testing.assert_allclose(r['first_violation']['time'],prior['first_violation']['time'],atol=1e-12)
        np.testing.assert_allclose(r['minimum_actual_clearance_m'],prior['minimum_actual_clearance_m'],atol=1e-12)
        assert r['simulation_time_s']-r['first_violation']['time']>=.5-1e-10


def test_original_scene_servo_and_execution_configuration_preserved():
    old=load_verified_path('D_two_obstacles',42);b=load_margin_path(planning()/'margin15mm_seed42.json')
    assert b.evidence['scene_configuration']==old.evidence['scene_configuration']
    np.testing.assert_array_equal(b.checker.model.actuator_gainprm,old.checker.model.actuator_gainprm)
    np.testing.assert_array_equal(b.checker.model.actuator_biasprm,old.checker.model.actuator_biasprm)
    np.testing.assert_array_equal(b.checker.robot.limits,old.checker.robot.limits)
    frozen=Path(read(ROOT/'results/phase2d_freeze_latest.json')['run'])
    assert all(sha(ROOT/p)==h for p,h in read(frozen/'historical_hashes.json').items())


def test_all_actual_metrics_recomputed_from_csv():
    audit=read(ROOT/'results/phase2d_csv_recomputation.json')
    assert audit['trials']==24 and audit['all_metrics_match']
    assert all(a['all_metrics_match'] for a in audit['audits'])
    assert len(dynamics())==24
    assert any(a['forbidden_contact_intervals']>0 for a in audit['audits'])


def test_local_decomposition_is_explicit_not_global_certificate():
    analysis=read(ROOT/'results/phase2d_failure_analysis.json');assert len(analysis['historical_failures'])==4
    for r in analysis['historical_failures']:
        parts=np.array(r['command_error_rad'])+r['gravity_quasistatic_error_estimate_rad']+np.array(r['velocity_damping_error_term_rad'])+r['residual_error_rad']
        np.testing.assert_allclose(parts,r['joint_error'],atol=1e-14)
        assert len(r['directional_distance_samples'])==7 and len(r['gradient_resolution_comparison'])==4
        assert r['prehold_acceleration_peak']['time_s']==.002
        assert r['prehold_acceleration_peak']['value_rad_s2']>r['motion_acceleration_peak']['value_rad_s2']


def test_actual_comparison_video_and_first_failure_annotation():
    import imageio.v2 as imageio
    run=Path(read(ROOT/'results/phase2d_acceptance.json')['video_run']);done=read(run/'complete.json')
    assert len(done['episodes'])==2
    a,b=done['episodes'];assert not a['execution']['success'] and b['execution']['success']
    assert a['video']['first_violation_frame'] is not None and b['video']['first_violation_frame'] is None
    for ep in done['episodes']:
        assert ep['video']['all_same_live_data'] and ep['video']['maximum_joint_displacement_rad']>.1
        assert ep['video']['execution_clearance_m']==.005
        assert read(run/ep['episode']/'demo_complete.json')['viewer_closed']
    with imageio.get_reader(str(run/'phase2d_margin_comparison.mp4')) as reader:
        assert sum(1 for _ in reader)==read(run/'comparison_validation.json')['frames']


def test_no_new_live_qpos_playback():
    for name in ['scripts/demo_margin_execution.py','scripts/run_phase2d_execution.py','src/planning/margin_study.py']:
        tree=ast.parse((ROOT/name).read_text())
        assert not any(isinstance(x,(ast.Assign,ast.AugAssign)) and 'qpos' in ast.unparse(x).split('=')[0] for x in ast.walk(tree))
