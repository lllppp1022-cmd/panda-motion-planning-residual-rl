"""Phase3C protocol, reward reconstruction and actual training regressions."""
import time
from pathlib import Path
import numpy as np
from src.common import ROOT,sha
from src.experiments.phase3c import DiagnosticEnv,design,reward_r1
from scripts.phase3c_common import read,sources

def test_historical_critic_and_optimizer_replay_exact():
    p=Path(read(ROOT/'results/phase3c_3b_audit_latest.json')['run']);r=read(p/'complete.json')
    assert r['recomputed_decisions']==1024 and r['all_components_match']
    assert all(u['loss_abs_difference']==0 and u['replayed_parameter_max_abs_error']==0 for u in r['updates'])
    assert 1400<r['updates'][1]['preoptimization_target_mse']<1500

def test_reward_same_prefix_discounted_rankings():
    from scripts.validate_phase3c import rankings
    r=rankings();assert len(r)==12 and all(x['same_prefix_completion_beats_stopping'] for x in r)

def test_r1_sensitivity_to_joint_and_tcp_error():
    row=dict(progress=.01,tcp_error_m=0.,**{f'q_actual_{j}':0. for j in range(1,8)},**{f'q_ref_{j}':0. for j in range(1,8)},
        **{f'dq_actual_{j}':0. for j in range(1,8)},**{f'dq_ref_{j}':0. for j in range(1,8)})
    good,_=reward_r1([row]*25,np.zeros(7),np.zeros(7),0,16,False,False,False)
    bad=dict(row,tcp_error_m=.001,q_actual_1=.001)
    worse,parts=reward_r1([bad]*25,np.zeros(7),np.zeros(7),0,16,False,False,False)
    assert worse<good and parts['joint']<0 and parts['tcp']<0

def test_new_paths_and_model_isolation():
    from scripts.run_phase3a import load_new
    paths=[load_new(sources(split)[0]) for split in ('train','validation','test')]
    assert len({b.evidence['path_csv_sha256'] for b in paths})==3
    assert all(b.evidence['planning_minimum_clearance_m']>=.02 and b.checker.options['minimum_clearance_m']==.005 for b in paths)

def test_gym_truncation_stream_flush_and_real_physics():
    folder=ROOT/'results/runs'/f'phase3c_pytest_truncate_{time.time_ns()}'
    env=DiagnosticEnv(sources('train'),folder,version='R1',max_decisions=1,mode='evaluation')
    try:
        obs,_=env.reset(seed=42);_,reward,term,trunc,info=env.step(np.zeros(7))
        assert trunc and not term and info['physical_steps_this_decision']>=275 and reward<0
    finally:env.close()
    from scripts.phase3c_common import rows
    assert len(rows(folder/'episode_0000/phase3c_decisions.csv'))==1
    assert len(rows(folder/'episode_0000/decisions.csv'))==1

def test_integral_and_probes_real_safety_gate():
    r=read(Path(read(ROOT/'results/phase3c_validation_latest.json')['run'])/'complete.json')
    assert r['passed'] and len(r['results'])==12
    assert all(x['success'] and not x['safety_violation_steps'] for x in r['results'])
    for split in ('c_train_sphere','c_validation_cylinder'):
        a=next(x for x in r['results'] if x['scene']==split and x['method']=='M3')
        b=next(x for x in r['results'] if x['scene']==split and x['method']=='CounterProbe')
        assert b['tcp_rmse_m']>a['tcp_rmse_m']

def test_common_budget_gate_and_independent_seeds():
    g=read(ROOT/'results/phase3c_2048_gate.json');assert g['all_four_passed']
    p=Path(read(ROOT/'results/phase3c_training_continue_latest.json')['run']);r=read(p/'complete.json')['results']
    assert {(x['version'],x['seed']) for x in r}=={(v,s) for v in ('R0','R1') for s in (42,123)}
    assert all(x['cumulative_decisions']==4096 and x['stage_decisions']==2048 for x in r)
    assert all(x['actor_delta_l2']>0 and x['critic_delta_l2']>0 for x in r)
    assert len({x['checkpoint_sha256'] for x in r})==4

def test_unchanged_historical_files():
    p=Path(read(ROOT/'results/phase3c_prepare_latest.json')['run']);m=read(p/'history_sha256.json')
    assert all((ROOT/name).exists() and sha(ROOT/name)==h for name,h in m.items())


def test_new_instrumentation_zero_residual_bitwise_physics():
    from src.rl.panda_residual_env import PandaResidualEnv
    from scripts.phase3c_common import rows
    base=ROOT/'results/runs'/f'phase3c_pytest_equivalence_{time.time_ns()}'
    environments=[PandaResidualEnv(sources('train'),base/'old',mode='evaluation'),
                  DiagnosticEnv(sources('train'),base/'new',version='R1',mode='evaluation')]
    for env in environments:
        try:
            env.reset(seed=42,options=dict(perturbation={'name':'gain90','kp_scale':.9},physical_seed=123))
            complete=False
            for _ in range(400):
                _,_,terminated,truncated,info=env.step(np.zeros(7))
                if terminated or truncated:
                    assert terminated and not truncated and info['success'];complete=True;break
            assert complete
        finally:env.close()
    left=rows(base/'old/episode_0000/tracking.csv');right=rows(base/'new/episode_0000/tracking.csv')
    assert len(left)==len(right)>7500
    for a,b in zip(left,right):
        assert a==b


def test_selected_reward_components_recomputed_from_actual_physics():
    r=read(ROOT/'results/phase3c_raw_log_reaudit.json');assert r['passed']
    assert len(r['stages'])==8
    for stage in r['stages']:
        assert sum(e['decisions'] for e in stage['episodes'])==2048
        assert all(e['reward_recompute_max_abs_error']<1e-9 and e['float32_sample_reward_max_abs_error']<2e-5 for e in stage['episodes'])
        assert all(e['max_active_residual_rate_rad_s']<=.010000001 and e['max_ctrl_rate_rad_s']<=.350000001 for e in stage['episodes'])
        assert all(b['gae_internal_max_abs_error']<3e-5 for b in stage['buffers'])


def test_physical_count_includes_reset_and_budget_failure_hold():
    r=read(ROOT/'results/phase3c_acceptance.json')
    for g in r['training']:
        assert g['decision_steps']==4096 and g['ppo_updates']==16
        assert g['total_physics_steps']==g['decision_physics_steps']+g['reset_physics_steps']+g['cleanup_physics_steps']
        assert g['cleanup_physics_steps']>=500 and g['budget_interrupted_episodes']==2


def test_final_protocol_frozen_and_all_seed_evaluations_retained():
    r=read(ROOT/'results/phase3c_acceptance.json');op=r['final_opening']
    assert op['no_tuning_after_opening'] and op['protocol_sha256']==sha(ROOT/'configs/phase3c_pilot.json')
    assert len(r['evaluation'])==42
    for group in r['training']:
        key=f"{group['reward']}_{group['training_seed']}"
        assert op['checkpoints'][key]['sha256']==group['checkpoint_sha256']==sha(group['checkpoint'])
    assert all(x['physical_seed']==9001 for x in r['evaluation'])


def test_checkpoints_reload_exact_and_actor_has_no_parameter_oracle():
    import torch,inspect
    from stable_baselines3 import PPO
    from src.rl.panda_residual_env import PandaResidualEnv
    source=inspect.getsource(PandaResidualEnv._observation)
    assert all(x not in source for x in ('body_mass','body_inertia','actuator_gainprm','physics_signature'))
    for g in read(ROOT/'results/phase3c_acceptance.json')['training']:
        a=PPO.load(g['checkpoint'],device='cpu');b=PPO.load(g['checkpoint'],device='cpu')
        observations=np.load(next(Path(g['run_directory']).glob('rollout_*.npz')))['observations'][:3,0]
        assert np.array_equal(a.predict(observations,deterministic=True)[0],b.predict(observations,deterministic=True)[0])
        assert torch.isfinite(a.policy.log_std).all()
