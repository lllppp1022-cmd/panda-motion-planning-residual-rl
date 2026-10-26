"""Actual Gym/control/PPO release regressions, alongside all frozen 157 tests."""
import copy,json,time
from pathlib import Path
import mujoco
import numpy as np
import pytest
from src.common import ROOT
from src.rl.dependencies import activate
activate()
from src.rl.panda_residual_env import PandaResidualEnv,config
from src.rl.residual_controller import ResidualM3Controller
from src.control.velocity_feedforward_controller import NominalFeedforwardController
from src.robot.panda_interface import PandaInterface
from src.planning.collision_checker import CollisionChecker
from scripts.validate_phase3b import sources,read,run_episode

def dest(name):
    p=ROOT/'results/runs'/f'phase3b_pytest_{name}_{time.time_ns()}';p.mkdir();return p
def robot(model):
    d=mujoco.MjData(model);mujoco.mj_resetDataKeyframe(model,d,mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_KEY,'home'));mujoco.mj_forward(model,d)
    return PandaInterface(model,d)

def test_zero_controller_exact_frozen_m3_for_every_step():
    c=CollisionChecker();r1=robot(copy.copy(c.model));r2=robot(copy.copy(c.model))
    a=NominalFeedforwardController(r1,c.model,'M3',{'name':'noise','q_noise_std_rad':.0005},42)
    b=ResidualM3Controller(r2,c.model,{'name':'noise','q_noise_std_rad':.0005},42,lambda:False)
    for i in range(100):
        q=c.home.copy();q[0]=i*.0001;a.step(q);b.step(q)
        assert np.array_equal(r1.data.qpos,r2.data.qpos) and np.array_equal(r1.data.ctrl,r2.data.ctrl)

def test_nonzero_mapping_slew_and_native_limits():
    c=CollisionChecker();r=robot(c.model);controller=ResidualM3Controller(r,c.model,{'name':'nominal'},42,lambda:False)
    controller.set_residual(np.ones(7));before=controller.command_q.copy();controller.step(c.home)
    assert np.allclose(controller.last['residual_rad'],.01*.002)
    assert np.allclose(controller.last['desired_before_delay']-controller.last['base_target'],.00002)
    assert np.max(abs(controller.command_q-before))<=.35*.002+1e-12
    assert np.max(abs(r.data.actuator_force[r.actuator_ids]))<=87

@pytest.mark.parametrize('action',[np.ones(6),np.full(7,np.nan),np.full(7,np.inf)])
def test_invalid_action_rejected(action):
    c=CollisionChecker();controller=ResidualM3Controller(robot(c.model),c.model,{'name':'nominal'},42,lambda:False)
    with pytest.raises(ValueError):controller.set_residual(action)

def test_full_trajectory_zero_equivalence_gate():
    r=read(ROOT/'results/phase3b_zero_residual_equivalence.json')
    assert r['passed'] and len(r['results'])==2
    assert all(x['full_trace_bitwise_equal'] and x['physics_steps_old']==x['physics_steps_zero']>8000 for x in r['results'])

def test_gym_interface_and_observation_dimensions():
    r=read(Path(read(ROOT/'results/phase3b_interface_latest.json')['run'])/'complete.json')
    assert r['passed'] and r['gymnasium_check'] and r['sb3_check'] and r['observation_dimension']==78

def test_actual_25_steps_and_truncation_uses_real_restricted_hold():
    env=PandaResidualEnv(sources('train'),dest('truncate'),conditions=[{'name':'nominal'}],max_decisions=2,mode='interface_check')
    try:
        obs,info=env.reset(seed=42);assert obs.shape==(78,) and obs.dtype==np.float32 and info['reset_physics_steps']==750
        _,reward,term,trunc,info=env.step(np.zeros(7));assert not term and not trunc and info['physical_steps_this_decision']==25
        obs,reward,term,trunc,info=env.step(np.ones(7)*2)
        assert trunc and not term and info['reason']=='rl_external_truncation'
        assert info['action_clipped'] and info['physical_steps_this_decision']>=275
        assert env.observation_space.contains(obs) and np.isfinite(reward)
        assert env.result['live_qpos_writes_after_initialization']==0
    finally:env.close()

def test_nonzero_action_gate_includes_stopping_response():
    r=read(ROOT/'results/phase3b_action_safety_tests.json')
    assert r['passed'] and len(r['results'])==12 and r['residual_slew_rad_s']==.01
    assert all(x['success'] and not x['safety_violation_steps'] and not x['prohibited_contact_observations'] for x in r['results'])
    assert r['stop_response']['additional_physics_steps']>=250
    assert r['stop_response']['result']['failure_reason']=='rl_external_truncation'

def test_original_alternating_failures_archived_as_task_failures():
    import csv
    runs=sorted((ROOT/'results/runs').glob('phase3b_safety_*'))
    initial=next(read(p/'complete.json') for p in runs if (p/'complete.json').exists() and not read(p/'complete.json')['passed'])
    failures=[r for r in initial['results'] if not r['success']]
    assert len(failures)==4
    for r in failures:
        assert r['failure_reason']=='goal_not_stably_reached' and not r['safety_violation_steps']
        with (Path(r['run_directory'])/'decisions.csv').open(encoding='utf-8-sig') as f:last=list(csv.DictReader(f))[-1]
        assert last['terminated']=='True' and last['truncated']=='False' and float(last['reward_failure'])==-200

def test_reward_completed_lower_bound_exceeds_early_failure():
    c=config();r=c['reward'];maximum_step_loss=sum(r[k] for k in ('joint_max_penalty','velocity_max_penalty','action_penalty','smoothness_max_penalty','time_penalty'))
    assert r['success_bonus']-c['max_decisions_per_episode']*maximum_step_loss>r['progress_total']-r['failure_penalty']

def test_real_smoke_updated_both_networks_and_saved_reloaded():
    p=Path(read(ROOT/'results/phase3b_training_latest.json')['run']);r=read(p/'complete.json')
    assert r['actual_ppo_decisions']==1024 and r['training_step_integrations']>25000
    assert r['ppo_update_calls']==4 and r['optimizer_epochs']==20
    assert r['actor_delta_l2']>0 and r['critic_delta_l2']>0 and r['action_head_weight_norm']>0
    assert r['initial_actor_sha256']!=r['final_actor_sha256'] and r['initial_critic_sha256']!=r['final_critic_sha256']
    assert r['exact_reload_prediction'] and r['stochastic_action_nonzero_fraction']>.99 and r['completed_episodes']
    assert r['physical_steps_before_cleanup']==r['training_step_integrations']+r['reset_hold_integrations']
    assert list((p/'tensorboard').rglob('events.out.tfevents.*'))

def test_new_final_split_and_frozen_checkpoint():
    p=Path(read(ROOT/'results/phase3b_eval_final_latest.json')['run']);gate=read(p/'heldout_opening.json')
    r=read(p/'complete.json');assert len(r['results'])==6 and gate['no_further_training_or_tuning']
    assert r['final_model_sha256']==gate['final_model_sha256']
    assert all('b_final_box'==x['scene'] for x in r['results'])

def test_actor_has_no_true_physics_parameters():
    # Observation schema is explicit fixed sensor/reference/controller state only.
    import inspect
    source=inspect.getsource(PandaResidualEnv._observation)
    assert all(x not in source for x in ('body_mass','body_inertia','actuator_gainprm','physics_signature','minimum_clearance'))
    assert config()['execution_clearance_m']==.005

def test_residual_compensation_uses_nominal_mass_not_actual_mass():
    from src.control.uncertain_models import model_pair
    c=CollisionChecker();n,p,_=model_pair(c.model,{'name':'mass','mass_inertia_scale':1.05})
    controller=ResidualM3Controller(robot(p),n,{'name':'mass','mass_inertia_scale':1.05},42,lambda:False)
    controller.step(c.home)
    assert np.array_equal(controller.gravity.model.body_mass,n.body_mass)
    assert not np.array_equal(controller.gravity.model.body_mass,p.body_mass)

def test_frozen_prior_files_unchanged():
    from src.common import sha
    p=Path(read(ROOT/'results/phase3b_prepare_latest.json')['run']);manifest=read(p/'history_sha256.json')
    assert all((ROOT/name).exists() and sha(ROOT/name)==h for name,h in manifest.items())
