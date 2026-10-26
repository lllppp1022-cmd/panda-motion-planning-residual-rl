"""Actual archived Phase3D-C interface, physics and bounded training acceptance."""
from src.rl.dependencies import activate
activate()
from pathlib import Path
import json
import numpy as np
import torch
from stable_baselines3 import PPO
from src.common import ROOT,sha
from src.phase3dc.common import read,sources,prepared,csvrows,bc_checkpoint
from src.phase3dc.environment import DCEnv,design
from src.phase3dc.policy import make_model


def interface():return read(Path(read(ROOT/'results/phase3dc_interface_latest.json')['run'])/'complete.json')
def training():
    gate=read(ROOT/'results/phase3dc_2048_gate.json');stage='continue' if gate['all_four_passed'] else 'initial'
    return read(Path(read(ROOT/f'results/phase3dc_training_{stage}_latest.json')['run'])/'complete.json')['results']


def test_85d_normalization_identical_all_sb3_entry_points(tmp_path):
    torch.set_num_threads(1);env=DCEnv(sources('train'),tmp_path,mode='interface_check');m=make_model(env,42,bc_checkpoint(),'A');x=torch.as_tensor(np.linspace(-.2,.2,85)[None],dtype=torch.float32)
    with torch.no_grad():
        a,v,lp=m.policy(x,deterministic=True);d=m.policy.get_distribution(x);v2=m.policy.predict_values(x);v3,lp3,_=m.policy.evaluate_actions(x,a)
    assert torch.allclose(a,d.distribution.mean,atol=1e-7) and torch.equal(v,v2) and torch.equal(v,v3) and torch.equal(lp,lp3)
    assert env.observation_space.shape==(85,);env.close()


def test_bc_initialization_outputs_reload_and_dynamic_equivalence():
    a=interface();assert a['passed'] and a['exact_reload'] and a['action_max_abs_error']<2e-6
    assert all(r['maximum_state_ctrl_difference']<=2e-6 for r in a['equivalence'] if r['method']=='A')


def test_gymnasium_85d_env_real_step_checker(tmp_path):
    from stable_baselines3.common.env_checker import check_env
    env=DCEnv(sources('train')[:1],tmp_path,mode='interface_check',conditions=[{'name':'nominal'}])
    try:check_env(env,warn=True)
    finally:env.close()
    assert env.total_physics_steps>0 and env.total_decisions>=10


def test_integral20_zero_rl_bitwise_dynamics():
    assert all(r['bitwise_equivalent'] and r['maximum_state_ctrl_difference']==0 for r in interface()['equivalence'] if r['method']=='B')


def test_small_nonzero_correction_safe_and_not_fully_clipped():
    for r in interface()['nonzero_probes']:
        assert r['effective_fraction']>.1 and r['result']['success'] and r['result']['safety_violation_steps']==0 and r['result']['prohibited_contact_observations']==0


def test_new_geometry_splits_verified_and_distinct():
    sets=[set(map(str,sources(s))) for s in ('train','validation','test')]
    assert all(sets[i].isdisjoint(sets[j]) for i in range(3) for j in range(i+1,3))
    geometries=[read(path) for group in sets for path in group]
    assert len({json.dumps(r['scene'],sort_keys=True) for r in geometries})==5
    assert all(r['result']['success'] and r['result']['final_validation_passed'] for r in geometries)


def test_four_independent_runs_same_common_budget():
    rr=training();assert {(r['method'],r['seed']) for r in rr}=={(m,s) for m in ('A','B') for s in (42,123)}
    assert len({r['cumulative_decisions'] for r in rr})==1
    assert all(r['stage_decisions']==2048 for r in rr)
    assert next(r for r in rr if r['method']=='A' and r['seed']==42)['initial_actor_sha256']!=next(r for r in rr if r['method']=='B' and r['seed']==42)['initial_actor_sha256']


def test_actor_critic_updates_and_256_boundaries():
    for r in training():
        assert r['actor_delta_l2']>0 and r['critic_delta_l2']>0 and len(r['updates'])==8
        assert all(u['decision_steps']%256==0 and u['actor_parameter_delta_l2']>0 and u['critic_parameter_delta_l2']>0 for u in r['updates'])
        assert all(np.isfinite(u['value_loss']) and np.isfinite(u['approx_kl']) for u in r['updates'])


def test_true_physics_accounting_and_failure_archives():
    for r in training():
        assert r['total_physics']==r['reset_physics']+r['decision_physics']+r['cleanup_physics']
        folders=list((Path(r['run_directory'])/'actual_training').glob('episode_*'))
        assert sum(len(csvrows(f/'tracking.csv')) for f in folders)==r['total_physics']
        assert all(read(f/'episode_physics_summary.json')['live_qpos_writes_after_initialization']==0 for f in folders)


def test_actions_safety_and_original_native_limits_not_bypassed():
    for r in training():
        for f in (Path(r['run_directory'])/'actual_training').glob('episode_*'):
            rows=csvrows(f/'tracking.csv');u=np.array([[float(v[f'ctrl_{j}']) for j in range(1,8)] for v in rows]);assert np.max(abs(np.diff(u,axis=0)))<=.35*.002+1e-9
            extra=csvrows(f/'dc_control.csv');target=np.array([json.loads(v['combined_target']) for v in extra]);actual=np.array([json.loads(v['applied']) for v in extra])
            assert np.max(abs(target))<=.002+1e-12 and np.max(abs(actual))<=.002+1e-12
            force=np.array([json.loads(v['force']) for v in extra]);assert np.all(abs(force)<=np.array([87,87,87,87,12,12,12])+1e-8)


def test_frozen_final_opening_and_all_methods_present():
    d=Path(read(ROOT/'results/phase3dc_evaluation_final_latest.json')['run']);r=read(d/'complete.json');opening=read(d/'final_test_opening.json')
    assert opening['no_more_selection'] and len(r['results'])==48
    assert {x['evaluation_method'] for x in r['results']}=={'M3','Integral500','Integral20','BC85','A_seed42','A_seed123','B_seed42','B_seed123'}
    for name,h in opening['code_hashes'].items():assert sha(ROOT/name)==h


def test_training_rgb_frames_match_actual_live_logs():
    for r in training():
        assert not r['record_errors']
        for f in (Path(r['run_directory'])/'actual_training').glob('episode_*'):
            meta=read(f/'rgb_frames.json');z=np.load(f/'integration_states.npz');t=csvrows(f/'tracking.csv')
            for frame in meta:
                idx=frame['physics_step']-1;assert np.array_equal(z['qpos'][idx],frame['qpos']) and abs(float(t[idx]['time'])-frame['time'])<1e-12
                assert 'ACTUAL STOCHASTIC TRAINING' in frame['source']


def test_historical_data_and_bc_models_not_modified():
    manifest=read(prepared()/'history_sha256.json');assert all(sha(ROOT/name)==h for name,h in manifest.items())
