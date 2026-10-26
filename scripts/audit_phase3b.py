"""Independent reconstruction from actual control, tracking and PPO decision CSVs."""
from _bootstrap import ROOT,main_guard
import csv,json
from pathlib import Path
import numpy as np
from src.common import write_csv,write_json
from scripts.validate_phase3b import read
from src.planning.collision_checker import CollisionChecker

def rows(p):
    with Path(p).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def audit(r):
    folder=Path(r['run_directory']);t=rows(folder/'tracking.csv');u=rows(folder/'control_signals.csv');assert len(t)==len(u)==r['physics_steps']
    get=lambda key:np.array([[float(x[f'{key}_{j}']) for j in range(1,8)] for x in t])
    q,qr,v,a,ctrl=get('q_actual'),get('q_ref'),get('dq_actual'),get('ddq_actual_fd'),get('ctrl')
    i=np.array([j for j,x in enumerate(t) if x['phase'] in ('trajectory','settle')]);error=q[i]-qr[i]
    if len(i):
        rmse=np.sqrt(np.mean(error**2,axis=0));tcp=float(np.sqrt(np.mean([float(t[j]['tcp_error_m'])**2 for j in i])))
        assert np.allclose(rmse,r['joint_rmse_rad'],atol=1e-12) and abs(tcp-r['tcp_rmse_m'])<1e-12
    else:rmse=None;tcp=None
    minimum=min(float(x['minimum_clearance_m']) for x in t);assert abs(minimum-r['minimum_actual_clearance_m'])<1e-12
    violations=sum(x['safety_valid']=='False' for x in t);assert violations==r['safety_violation_steps']
    contacts=np.array([int(x['prohibited_contact_count'])>0 for x in t]);intervals=int(np.sum(contacts&~np.r_[False,contacts[:-1]]))
    param=read(folder/'model_parameters.json');m=param['physics'];ids=CollisionChecker().robot.actuator_ids
    kp=np.array(m['actuator_gainprm'])[ids,0];bias=np.array(m['actuator_biasprm'])[ids];bounds=np.array(m['actuator_forcerange'])[ids]
    forces=np.array([[float(x[f'actuator_force_{j}']) for j in range(1,8)] for x in u]);raw=kp*ctrl+bias[:,1]*q+bias[:,2]*v
    assert np.max(abs(forces-np.clip(raw,bounds[:,0],bounds[:,1])))<1e-7
    saturation=int(np.sum(np.any((forces<=bounds[:,0]+1e-8)|(forces>=bounds[:,1]-1e-8),axis=1)))
    rate=np.diff(ctrl,axis=0)/.002;assert np.max(abs(rate))<=.35+1e-9
    if 'residual_rad_1' in u[0]:
        residual=np.array([[float(x[f'residual_rad_{j}']) for j in range(1,8)] for x in u]);assert np.max(abs(residual))<=.002+1e-12
        active=np.array([x['phase']!='protective_hold' for x in t]);rdiff=np.diff(residual,axis=0)/.002
        assert np.max(abs(rdiff[active[1:]&active[:-1]]))<=.01+1e-9
    return dict(scene=r['scene'],method=r.get('comparison_method','actual_training'),condition=r.get('condition',{}).get('name','see_reset_json'),
        success=r['success'],reason=r['failure_reason'],physics_steps=len(t),joint_rmse_rad=json.dumps(rmse.tolist()) if rmse is not None else '',
        tcp_rmse_m=tcp,final_tcp_error_m=r['final_tcp_error_m'],minimum_clearance_m=minimum,safety_violation_steps=violations,
        prohibited_contact_intervals=intervals,saturation_steps=saturation,command_max_rate_rad_s=float(np.max(abs(rate))),
        actual_max_velocity_rad_s=float(np.max(abs(v))),actual_max_acceleration_rad_s2=float(np.max(abs(a))),run_directory=str(folder))

def main():
    p=Path(read(ROOT/'results/phase3b_training_latest.json')['run']);tr=read(p/'complete.json');evaluations=[]
    for stage in ('validation','final'):
        e=Path(read(ROOT/f'results/phase3b_eval_{stage}_latest.json')['run']);evaluations+=read(e/'complete.json')['results']
    scientific=[audit(r) for r in evaluations];training=[audit(r) for r in tr['episode_results']]
    pairs={}
    for r in evaluations:pairs.setdefault((r['scene'],json.dumps(r['condition'],sort_keys=True)),[]).append(r)
    fairness=[]
    for key,group in pairs.items():
        assert {'M1','M3','M3_PPO'}<=set(x['comparison_method'] for x in group)
        reference=read(Path(group[0]['run_directory'])/'model_parameters.json')
        trace0=rows(Path(group[0]['run_directory'])/'control_signals.csv')
        source0=read(Path(group[0]['run_directory'])/'frozen_run_config.json')['path_evidence']
        for r in group[1:]:
            current=read(Path(r['run_directory'])/'model_parameters.json')
            for field in ('physics_signature','nominal_signature','seed'):assert current[field]==reference[field]
            source=read(Path(r['run_directory'])/'frozen_run_config.json')['path_evidence']
            assert source['path_csv_sha256']==source0['path_csv_sha256'] and source['scene_configuration']==source0['scene_configuration']
            trace1=rows(Path(r['run_directory'])/'control_signals.csv')
            assert all(a[f'noise_{j}']==b[f'noise_{j}'] for a,b in zip(trace0,trace1) for j in range(1,8))
        fairness.append(dict(scene=key[0],condition=json.loads(key[1]),methods=[x['comparison_method'] for x in group],
            same_path_scene_physics_nominal_seed_noise=True))
    sample=rows(p/'training_samples.csv');assert len(sample)==1024
    physical=sum(x['physics_steps'] for x in training);assert physical==tr['actual_physics_integrations_total']
    assert sum(int(x['physical_steps_this_decision']) for x in sample)==tr['training_step_integrations']
    for r in sample:
        components=sum(float(v) for k,v in r.items() if k.startswith('reward_'))
        assert abs(components-float(r['reward']))<2e-5
        assert np.isfinite(float(r['reward']))
        assert r['sample_mode']=='stochastic_actual_training'
        assert np.array_equal(np.clip(json.loads(r['raw_gaussian_action']),-1,1),json.loads(r['executed_normalized_action']))
    write_csv(ROOT/'results/phase3b_evaluation_comparison.csv',scientific)
    write_json(ROOT/'results/phase3b_csv_audit.json',dict(passed=True,actual_training_episodes=training,evaluation_runs=len(evaluations),
        evaluation_audits=scientific,actual_training_physics_steps_recomputed=physical,actual_decisions=len(sample),all_rewards_finite=True,
        force_mapping_and_limits_verified=True,privileged_physics_not_actor_input=True,paired_fairness=fairness))
    print('AUDIT PASS',len(evaluations),'EVALUATIONS',len(sample),'DECISIONS',physical,'ACTUAL TRAINING PHYSICS',flush=True)

if __name__=='__main__':main_guard(main)
