"""Recompute every scientific metric from actual physics CSVs and verify fairness."""
from _bootstrap import ROOT,main_guard
import csv,json
from pathlib import Path
import numpy as np
from src.common import write_json,write_csv,sha

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def all_results():
    out=[]
    for stage in ('pilot','test'):
        pointer=ROOT/f'results/phase3a_{stage}_latest.json'
        if pointer.exists():out+=read(Path(read(pointer)['run'])/'complete.json')['results']
    return out

def audit_one(r):
    dest=Path(r['run_directory']);t=rows(dest/'tracking.csv');u=rows(dest/'control_signals.csv')
    assert len(t)==len(u)==r['physics_steps'] and all(a['physics_step']==b['physics_step'] for a,b in zip(t,u))
    q=np.array([[float(x[f'q_actual_{j}']) for j in range(1,8)] for x in t]);qr=np.array([[float(x[f'q_ref_{j}']) for j in range(1,8)] for x in t])
    v=np.array([[float(x[f'dq_actual_{j}']) for j in range(1,8)] for x in t]);acc=np.array([[float(x[f'ddq_actual_fd_{j}']) for j in range(1,8)] for x in t])
    ctrl=np.array([[float(x[f'ctrl_{j}']) for j in range(1,8)] for x in t]);indices=[i for i,x in enumerate(t) if x['phase'] in ('trajectory','settle')]
    err=q[indices]-qr[indices];tcp=np.array([float(t[i]['tcp_error_m']) for i in indices]);gap=np.array([float(x['minimum_clearance_m']) for x in t])
    contact=np.array([int(x['prohibited_contact_count'])>0 for x in t]);intervals=int(np.count_nonzero(contact&~np.r_[False,contact[:-1]]))
    violations=sum(x['safety_valid']=='False' for x in t);tail=t[-250:]
    goal=np.array(r['goal_q']);tcpgoal=np.array([float(x) for x in [r.get('goal_tcp_x',0)]]) # final FK error from immutable run summary checked below via final actual/reference FK
    # Physical model includes actual masses but FK geometry is independently recomputed.
    import mujoco
    from src.robot.panda_interface import PandaInterface
    m=mujoco.MjModel.from_binary_path(str(dest/'perturbed_physics_model.mjb'));d=mujoco.MjData(m);robot=PandaInterface(m,d)
    d.qpos[:]=m.key_qpos[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_KEY,'home')]
    d.qpos[robot.qpos_indices]=q[-1];mujoco.mj_forward(m,d);actual=robot.tool_pose()[0]
    d.qpos[robot.qpos_indices]=goal;mujoco.mj_forward(m,d);tcpgoal=robot.tool_pose()[0];finaltcp=float(np.linalg.norm(actual-tcpgoal))
    # Actual fingers can deviate. Reconstruct final full qpos from deterministic finger state is not in old tracking CSV;
    # use last actual TCP already logged for exact full-state endpoint and nominal goal FK (fingers don't move TCP site).
    finaltcp=float(np.linalg.norm(np.array([float(t[-1][f'tcp_actual_{j}']) for j in range(3)])-tcpgoal))
    stable=all(max(abs(float(x[f'q_actual_{j}'])-goal[j-1]) for j in range(1,8))<=.02
        and np.linalg.norm(np.array([float(x[f'tcp_actual_{j}']) for j in range(3)])-tcpgoal)<=.01
        and max(abs(float(x[f'dq_actual_{j}'])) for j in range(1,8))<=.02 for x in tail)
    success=bool(not violations and len(tail)==250 and stable and r['failure_reason']=='')
    rmse=np.sqrt(np.mean(err*err,axis=0)) if len(err) else np.zeros(7)
    tcp_rmse=float(np.sqrt(np.mean(tcp*tcp))) if len(tcp) else None
    assert success==r['success']
    assert abs(min(gap)-r['minimum_actual_clearance_m'])<1e-12
    assert np.allclose(rmse,r['joint_rmse_rad'],atol=1e-12) if len(err) else r['joint_rmse_rad'] is None
    if tcp_rmse is not None:assert abs(tcp_rmse-r['tcp_rmse_m'])<1e-12
    assert abs(finaltcp-r['final_tcp_error_m'])<1e-10
    params=read(dest/'model_parameters.json');phys=params['physics'];nom=params['nominal']
    assert sha(dest/'perturbed_physics_model.mjb')==params['physics_mjb_sha256']
    assert sha(dest/'nominal_controller_model.mjb')==params['nominal_mjb_sha256']
    assert nom['actuator_forcerange']==phys['actuator_forcerange'] and nom['jnt_range']==phys['jnt_range']
    ids=robot.actuator_ids;kp=np.array(phys['actuator_gainprm'])[ids,0];bias=np.array(phys['actuator_biasprm'])[ids]
    raw=kp*(ctrl-q)+bias[:,2]*v;bounds=np.array(phys['actuator_forcerange'])[ids]
    forces=np.array([[float(x[f'actuator_force_{j}']) for j in range(1,8)] for x in u])
    assert np.max(abs(forces-np.clip(raw,bounds[:,0],bounds[:,1])))<1e-7
    saturation=np.sum(np.any((forces<=bounds[:,0]+1e-8)|(forces>=bounds[:,1]-1e-8),axis=1))
    assert saturation==r['saturation_steps']
    cost=np.mean([float(x['algorithm_s']) for x in u])*1e6
    assert abs(cost-r['controller_extra_us_per_step'])<1e-8
    rate=np.diff(ctrl,axis=0)/.002
    assert np.max(abs(rate))<=.35+1e-9
    runconfig=read(dest/'frozen_run_config.json');evidence=runconfig['path_evidence']
    assert evidence['verified'] and evidence['path_csv_sha256']==sha(evidence['path_csv'])
    first=lambda condition: next((float(x['time']) for x in t if condition(x)),None)
    return dict(scene=r['scene'],split=r['split'],method=r['method'],perturbation=r['perturbation'],disturbance_seed=r['disturbance_seed'],
        status=r['status'],success=success,valid_control_task=r['valid_control_task'],failure_reason=r['failure_reason'],
        completion_progress=max(float(x['progress']) for x in t if x['phase']!='protective_hold'),
        joint_rmse_rad=json.dumps(rmse.tolist()),joint_rmse_all_rad=float(np.sqrt(np.mean(err*err))) if len(err) else None,
        joint_max_error_rad=float(np.max(abs(err))) if len(err) else None,tcp_rmse_m=tcp_rmse,final_tcp_error_m=finaltcp,
        minimum_actual_clearance_m=float(min(gap)),hard_violation=bool(violations),hard_violation_steps=violations,
        prohibited_contact_intervals=intervals,first_violation_time_s=first(lambda x:x['safety_valid']=='False'),
        first_contact_time_s=first(lambda x:int(x['prohibited_contact_count'])>0),
        max_velocity_rad_s=float(np.max(abs(v))),max_acceleration_rad_s2=float(np.max(abs(acc))),saturation_steps=int(saturation),
        simulation_time_s=float(t[-1]['time']),wall_time_s=r['wall_time_s'],controller_extra_us_per_step=cost,
        command_max_rate_rad_s=float(np.max(abs(rate))),run_directory=str(dest),path_sha256=evidence['path_csv_sha256'],
        nominal_signature=r['nominal_signature'],physics_signature=r['physics_signature'],
        planned_minimum_clearance_m=r['planning_minimum_clearance_m'],source='independently recomputed actual physics CSV')

def main():
    results=all_results();audits=[];invalid=[]
    for i,r in enumerate(results):
        if not r.get('valid_control_task',False):invalid.append(r);continue
        audits.append(audit_one(r))
        if (i+1)%10==0:print('AUDIT',i+1,len(results),flush=True)
    pairs={};fair=[]
    for a in audits:pairs.setdefault((a['scene'],a['perturbation'],a['disturbance_seed']),{})[a['method']]=a
    for key,pair in pairs.items():
        assert set(pair)=={'M1','M3'}
        a,b=pair['M1'],pair['M3']
        for field in ('path_sha256','physics_signature','nominal_signature'):assert a[field]==b[field]
        ua,ub=rows(Path(a['run_directory'])/'control_signals.csv'),rows(Path(b['run_directory'])/'control_signals.csv')
        common=min(len(ua),len(ub));noise_equal=all(x[f'noise_{j}']==y[f'noise_{j}'] for x,y in zip(ua,ub) for j in range(1,8))
        assert noise_equal
        fair.append(dict(scene=key[0],perturbation=key[1],seed=key[2],common_steps=common,noise_equal=noise_equal,
            identical_physics=True,identical_path=True,M1_success=a['success'],M3_success=b['success'],M1_tcp_rmse_m=a['tcp_rmse_m'],M3_tcp_rmse_m=b['tcp_rmse_m']))
    write_csv(ROOT/'results/phase3a_controller_comparison.csv',audits);write_csv(ROOT/'results/phase3a_paired_fairness.csv',fair)
    nominal=[]
    for a in audits:
        n=next(x for x in audits if x['scene']==a['scene'] and x['method']==a['method'] and x['perturbation']=='nominal')
        nominal.append(dict(scene=a['scene'],split=a['split'],method=a['method'],perturbation=a['perturbation'],disturbance_seed=a['disturbance_seed'],
            nominal_success=n['success'],perturbed_success=a['success'],nominal_tcp_rmse_m=n['tcp_rmse_m'],perturbed_tcp_rmse_m=a['tcp_rmse_m'],
            nominal_final_tcp_error_m=n['final_tcp_error_m'],perturbed_final_tcp_error_m=a['final_tcp_error_m'],
            perturbed_hard_violation=a['hard_violation'],completion_progress=a['completion_progress'],run_directory=a['run_directory']))
    write_csv(ROOT/'results/phase3a_nominal_vs_perturbed.csv',nominal)
    write_json(ROOT/'results/phase3a_csv_audit.json',dict(actual_runs=len(results),valid_recomputed=len(audits),invalid_runs=invalid,
        paired_conditions=len(fair),all_pairs_fair=True,all_metrics_match=True,zero_live_qpos_writes=all(r.get('live_qpos_writes_after_initialization')==0 for r in results)))
    print('ALL CSV AUDITS PASSED',len(audits),'PAIRS',len(fair),flush=True)

if __name__=='__main__':main_guard(main)
