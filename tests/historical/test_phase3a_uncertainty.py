"""Isolation, actuator derivation, causal delay and real-integrator regressions."""
import ast,copy,json
from pathlib import Path
import mujoco
import numpy as np
import pytest
from src.common import ROOT
from src.planning.collision_checker import CollisionChecker
from src.robot.panda_interface import PandaInterface
from src.control.uncertain_models import model_pair,signature
from src.control.velocity_feedforward_controller import NominalFeedforwardController
from src.control.compensated_joint_controller import CompensatedJointController

def robot(model):
    d=mujoco.MjData(model);mujoco.mj_resetDataKeyframe(model,d,mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_KEY,'home'));mujoco.mj_forward(model,d)
    return PandaInterface(model,d)

def test_mass_model_isolation_and_inertia():
    c=CollisionChecker();before=signature(c.model);n,p,a=model_pair(c.model,{'name':'mass','mass_inertia_scale':1.2})
    assert signature(n)==before==signature(c.model) and signature(p)!=before
    ids=a['moving_body_ids'];assert np.allclose(p.body_mass[ids],1.2*n.body_mass[ids])
    assert np.allclose(p.body_inertia[ids],1.2*n.body_inertia[ids])
    assert np.array_equal(n.actuator_forcerange,p.actuator_forcerange)

@pytest.mark.parametrize('p',[{'name':'bad','kp_scale':0},{'name':'bad','mass_inertia_scale':float('nan')},{'name':'bad','action_delay_steps':-1},{'name':'bad','q_noise_std_rad':1}])
def test_invalid_physical_configs_rejected(p):
    with pytest.raises(ValueError):model_pair(CollisionChecker().model,p)

def test_gain_bias_mapping_and_unchanged_limits():
    n,p,_=model_pair(CollisionChecker().model,{'name':'gain','kp_scale':.8});r=robot(p);ids=r.actuator_ids
    assert np.allclose(p.actuator_gainprm[ids,0],.8*n.actuator_gainprm[ids,0])
    assert np.array_equal(p.actuator_biasprm[ids,2],n.actuator_biasprm[ids,2])
    assert np.allclose(p.actuator_biasprm[ids,1],-p.actuator_gainprm[ids,0])

def test_zero_extension_exact_original_m1():
    c=CollisionChecker();r1,r2=robot(copy.copy(c.model)),robot(copy.copy(c.model))
    old=CompensatedJointController(r1);new=NominalFeedforwardController(r2,c.model,'M1',{'name':'nominal'})
    for i in range(50):
        target=c.home.copy();target[0]+=.01*i/50;old.step(target);new.step(target)
        assert np.array_equal(r1.data.qpos,r2.data.qpos)
        assert np.array_equal(r1.data.ctrl,r2.data.ctrl)

def test_controller_never_queries_perturbed_gravity():
    c=CollisionChecker();n,p,_=model_pair(c.model,{'name':'mass','mass_inertia_scale':1.2});r=robot(p)
    controller=NominalFeedforwardController(r,n,'M1');target=r.positions();controller.step(target)
    assert np.array_equal(controller.gravity.model.body_mass,n.body_mass)
    assert not np.array_equal(controller.gravity.model.body_mass,p.body_mass)

def test_ff_units_sign_and_original_rate_limit():
    c=CollisionChecker();r=robot(c.model);controller=NominalFeedforwardController(r,c.model,'M3');q=r.positions();controller.step(q)
    target=q.copy();target[0]+=.0002;before=controller.command_q.copy();controller.step(target)
    assert controller.last['reference_velocity'][0]==pytest.approx(.1)
    assert controller.last['velocity_offset'][0]==pytest.approx(.01)
    assert np.max(abs(controller.command_q-before))<=.35*.002+1e-12
    assert r.data.time==pytest.approx(.004)

def test_exact_fifo_delay_and_paired_noise():
    c=CollisionChecker();rs=[robot(copy.copy(c.model)) for _ in range(2)]
    cs=[NominalFeedforwardController(r,c.model,m,{'name':'noise','q_noise_std_rad':.002,'action_delay_steps':10},3101) for r,m in zip(rs,['M1','M3'])]
    q=rs[0].positions();q[0]+=.01
    for i in range(12):
        for controller in cs:controller.step(q)
        assert np.array_equal(cs[0].last['noise'],cs[1].last['noise'])
        if i<10:assert cs[0].last['delayed_target'][0]==0
    assert cs[0].last['delayed_target'][0]>.009

def test_observation_delay_uses_past_position_only():
    c=CollisionChecker();r=robot(c.model);controller=NominalFeedforwardController(r,c.model,'M1',{'name':'delay','observation_delay_steps':3})
    initial=r.positions();past=[];q=initial.copy();q[0]=.1
    for i in range(9):
        past.append(r.positions());controller.step(q)
        assert np.array_equal(controller.last['observed_q'],initial if i<3 else past[i-3])

def test_damping_perturbation_does_not_modify_servo_or_geometry():
    c=CollisionChecker();n,p,_=model_pair(c.model,{'name':'damping','joint_damping_scale':1.5});r=robot(p)
    assert np.allclose(p.dof_damping[r.dof_indices],1.5*n.dof_damping[r.dof_indices])
    assert np.array_equal(p.actuator_biasprm,n.actuator_biasprm)
    assert np.array_equal(p.body_mass,n.body_mass) and np.array_equal(p.geom_size,n.geom_size)

def test_no_runtime_live_qpos_writes():
    tree=ast.parse((ROOT/'src/control/velocity_feedforward_controller.py').read_text())
    step=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='step')
    for node in ast.walk(step):
        if isinstance(node,(ast.Assign,ast.AugAssign)):
            for target in node.targets if isinstance(node,ast.Assign) else [node.target]:
                assert 'data.qpos' not in ast.unparse(target)

def test_static_dynamic_integrations_and_zero_ff():
    p=ROOT/'results/phase3a_static_latest.json'
    assert p.exists()
    records=json.loads((Path(json.loads(p.read_text())['run'])/'complete.json').read_text())
    assert records['passed'] and len(records['results'])==4
    assert all(r['physics_steps']>500 and r['live_qpos_writes_after_initialization']==0 and r['valid_control_task'] for r in records['results'])

def test_prepared_paths_different_and_validated():
    from scripts.run_phase3a import prepared,read,load_new
    plans=[p for p in read(prepared()/'plans.json') if p['success']]
    assert len(plans)==3
    bundles=[load_new(p['source_json']) for p in plans]
    assert len({b.evidence['path_csv_sha256'] for b in bundles})==3
    assert all(b.checker.options['minimum_clearance_m']==.005 for b in bundles)
    assert all(b.evidence['planning_minimum_clearance_m']>=.010 for b in bundles)

def test_formal_csv_recomputation_and_paired_fairness():
    import csv
    audit=json.loads((ROOT/'results/phase3a_csv_audit.json').read_text())
    assert audit['actual_runs']==audit['valid_recomputed']==120
    assert audit['paired_conditions']==60 and audit['all_pairs_fair'] and audit['all_metrics_match']
    assert audit['zero_live_qpos_writes'] and not audit['invalid_runs']
    with (ROOT/'results/phase3a_paired_fairness.csv').open(encoding='utf-8-sig') as f:rs=list(csv.DictReader(f))
    assert len(rs)==60 and all(r['noise_equal']==r['identical_physics']==r['identical_path']=='True' for r in rs)

def test_test_split_opened_after_pilot_without_tuning():
    read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
    pilot=Path(read(ROOT/'results/phase3a_pilot_latest.json')['run']);heldout=Path(read(ROOT/'results/phase3a_test_latest.json')['run'])
    gate=read(heldout/'heldout_opening.json')
    assert Path(gate['pilot'])==pilot and gate['no_tuning_after_opening']
    assert gate['opened_ns']>(pilot/'complete.json').stat().st_mtime_ns
    assert gate['source_hashes']==read(pilot/'complete.json')['source_hashes']
    assert all(r['split']=='test' for r in read(heldout/'complete.json')['results'])

def test_actual_live_video_decodable_and_matches_physics_csv():
    import imageio.v2 as imageio
    read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
    accepted=ROOT/'results/phase3a_acceptance.json'
    folder=Path(read(accepted)['video_run']) if accepted.exists() else Path(read(ROOT/'results/phase3a_video_latest.json')['run'])
    r=read(folder/'complete.json')
    assert len(r['episodes'])==3 and r['frames']>100 and r['only_live_integrations']
    for episode in r['episodes']:
        v=episode['validation'];assert v['viewer_launched'] and v['viewer_closed'] and not v['aborted']
        assert v['all_frames_match_actual_CSV'] and v['unique_rgb_frames']>10 and v['joint_displacement_rad']>.1
    with imageio.get_reader(str(folder/'phase3a_robustness_comparison.mp4')) as reader:
        assert sum(1 for _ in reader)==r['frames']

def test_entire_frozen_history_still_identical():
    from src.common import sha
    pointer=json.loads((ROOT/'results/phase3a_prepare_latest.json').read_text())
    manifest=json.loads((Path(pointer['run'])/'history_sha256.json').read_text(encoding='utf-8'))
    assert len(manifest)>1000
    assert all((ROOT/name).exists() and sha(ROOT/name)==value for name,value in manifest.items())

def test_tight_historical_path_failures_are_not_relabelled_as_success():
    import csv
    with (ROOT/'results/phase3a_controller_comparison.csv').open(encoding='utf-8-sig') as f:rs=list(csv.DictReader(f))
    failures=[r for r in rs if r['scene']=='D_two_obstacles' and r['perturbation'] in ('gain_low','mass_high','combined')]
    assert len(failures)==12
    assert all(r['success']=='False' and r['failure_reason']=='dynamic_safety_violation' and float(r['completion_progress'])<1 for r in failures)
    assert all(float(r['minimum_actual_clearance_m'])<.005 for r in failures)
