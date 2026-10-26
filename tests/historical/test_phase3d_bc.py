"""Causal BC labels, independently observable memory, split isolation and real control."""
import copy,time
from pathlib import Path
import mujoco
import numpy as np
from src.common import ROOT,sha
from src.bc_residual.core import ObservableIntegral,ShadowController,Policy,design
import torch
from src.experiments.integral_control import IntegralM3Controller
from src.rl.residual_controller import ResidualM3Controller
from src.robot.panda_interface import PandaInterface
from src.planning.collision_checker import CollisionChecker
from scripts.phase3c_common import read,rows


def make_robot(model):
    d=mujoco.MjData(model);mujoco.mj_resetDataKeyframe(model,d,mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_KEY,'home'));mujoco.mj_forward(model,d);return PandaInterface(model,d)


def test_observable_integral_matches_frozen_teacher_every_step():
    c=CollisionChecker();a=make_robot(copy.copy(c.model));b=make_robot(copy.copy(c.model));condition=dict(name='delayed_noise',q_noise_std_rad=.0002,observation_delay_steps=2)
    old=IntegralM3Controller(a,c.model,condition,42,lambda:False);new=ShadowController(IntegralM3Controller(b,c.model,condition,42,lambda:False),lambda:False,expert=True)
    for i in range(120):
        q=c.home.copy();q[0]+=.00005*i;old.step(q);new.step(q)
        assert np.array_equal(old.integral,new.memory.value) and np.array_equal(a.data.qpos,b.data.qpos) and np.array_equal(a.data.ctrl,b.data.ctrl)


def test_integral_clamp_antiwindup_and_reset_are_actual_rules():
    c=CollisionChecker();robot=make_robot(c.model);inner=ResidualM3Controller(robot,c.model,{'name':'nominal'},42,lambda:False);memory=ObservableIntegral()
    memory.value[:]=.00199;memory.advance(c.home+.1,inner,False);assert np.all(memory.value==.002)
    inner.last=dict(observed_q=c.home-.1,delayed_target=inner.command_q+1.);before=memory.value.copy();memory.advance(c.home,inner,False)
    assert np.array_equal(memory.value,before) and memory.blocked.all()
    memory.advance(c.home,inner,True);assert not memory.value.any()


def test_action_label_is_pre_slew_normalized_target_not_applied():
    c=CollisionChecker();robot=make_robot(c.model);controller=ResidualM3Controller(robot,c.model,{'name':'nominal'},42,lambda:False)
    target=np.ones(7);controller.set_residual(target);controller.step(c.home)
    assert np.array_equal(controller.last['residual_target_rad']/.002,target)
    assert np.allclose(controller.last['residual_rad']/.002,.01)
    assert not np.allclose(target,controller.last['residual_rad']/.002)


def test_dataset_split_causal_labels_and_real_cost():
    manifest=rows(ROOT/'results/phase3d_bc_dataset_manifest.csv');assert len(manifest)==28
    assert sum(r['split']=='train' for r in manifest)==24 and sum(r['split']=='validation' for r in manifest)==4
    assert all(r['split']!='test' and r['quality_accepted']=='True' for r in manifest)
    assert set(r['source'] for r in manifest if r['split']=='train').isdisjoint(r['source'] for r in manifest if r['split']=='validation')
    for r in manifest:
        z=np.load(r['samples_file']);assert len(z['labels'])==int(r['samples'])
        assert np.array_equal(z['observations85'][:,-7:],z['labels']) and np.all(np.diff(z['physics_step'])==25)


def test_train_only_normalizer_and_checkpoint_roundtrip():
    p=Path(read(ROOT/'results/phase3d_bc_training_latest.json')['run']);frozen=read(p/'frozen_training_design.json');xs=[]
    for r in frozen['training_files']:xs.append(np.load(r['samples_file'])['observations85'])
    x=np.concatenate(xs)
    for m in read(p/'complete.json')['results']:
        c=torch.load(m['best_checkpoint'],weights_only=True);d=m['dimension']
        assert np.array_equal(c['mean'].numpy(),x[:,:d].mean(axis=0))
        assert np.array_equal(c['std'].numpy(),np.maximum(x[:,:d].std(axis=0),.01))
        a=Policy(m['best_checkpoint']);b=Policy(m['best_checkpoint']);assert np.array_equal(a.predict(x[0]),b.predict(x[0]))


def test_bc78_cannot_read_extra_integral_or_physics_parameters():
    p=Path(read(ROOT/'results/phase3d_bc_training_latest.json')['run']);m=next(m for m in read(p/'complete.json')['results'] if m['dimension']==78)
    actor=Policy(m['best_checkpoint']);x=np.zeros(85,dtype=np.float32);y=x.copy();y[-7:]=1
    assert np.array_equal(actor.predict(x),actor.predict(y))
    import inspect
    assert all(name not in inspect.getsource(actor.predict) for name in ('body_mass','actuator_gainprm','future','teacher.integral'))


def test_bc_policy_action_bounds_finite():
    p=Path(read(ROOT/'results/phase3d_bc_training_latest.json')['run'])
    for m in read(p/'complete.json')['results']:
        actor=Policy(m['best_checkpoint']);a=actor.predict(np.ones(85,dtype=np.float32))
        assert a.shape==(7,) and np.isfinite(a).all() and np.max(abs(a))<=1


def test_final_opening_frozen_models_and_code():
    p=Path(read(ROOT/'results/phase3d_bc_eval_final_latest.json')['run']);f=read(p/'final_test_opening.json')
    assert f['no_further_training_or_selection'] and f['config_sha256']==sha(ROOT/'configs/phase3d_bc.json')
    assert all(sha(ROOT/name)==h for name,h in f['controller_code_sha256'].items())
    assert all(sha(v['path'])==v['sha256'] for v in f['checkpoints'].values())


def test_real_closed_loop_all_methods_and_unseen_tasks_retained():
    a=read(ROOT/'results/phase3d_bc_analysis.json');assert a['passed'] and len(a['all_evaluations'])==50
    final=[r for r in a['all_evaluations'] if r['split']=='final'];assert len(final)==30
    assert {r['method'] for r in final}==set(design()['methods'])
    assert {r['condition'] for r in final}=={'gain88','mass112','combo_test'}
    assert all(r['max_target_residual_rad']<=.002+1e-12 and r['max_active_residual_rate_rad_s']<=.010000001 for r in final)


def test_student_samples_come_from_own_closed_loop_states():
    cases=read(ROOT/'results/phase3d_bc_analysis.json')['all_evaluations']
    for r in cases:
        if r['method'] not in ('BC78','BC85'):continue
        d=Path(r['run_directory']);z=np.load(d/'bc_samples.npz');signals=rows(d/'control_signals.csv')
        for j,s in enumerate(z['physics_step']):
            own=np.array([float(signals[int(s)-1][f'observable_integral_{k}']) for k in range(1,8)])/.002
            assert np.max(abs(own-z['observations85'][j,-7:]))<1e-7


def test_actual_dynamic_video_matches_formal_execution():
    p=Path(read(ROOT/'results/phase3d_bc_video_latest.json')['run']);r=read(p/'complete.json')
    assert len(r['results'])==5 and r['frames']>100 and r['sha256']==sha(ROOT/'assets/videos/phase3d_bc_vs_integral.mp4')
    assert all(x['all_frames_match'] and x['tracking_exactly_matches_formal_evaluation'] and x['joint_displacement_rad']>.01 for x in r['results'])


def test_all_historical_files_frozen():
    p=Path(read(ROOT/'results/phase3d_bc_prepare_latest.json')['run']);manifest=read(p/'history_sha256.json')
    assert all(sha(ROOT/name)==value for name,value in manifest.items())
