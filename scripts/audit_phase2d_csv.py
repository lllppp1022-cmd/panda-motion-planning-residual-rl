"""Independently recompute every formal execution metric from raw CSV."""
from _bootstrap import main_guard
import csv,json
from pathlib import Path
import numpy as np
from src.common import ROOT,write_json,write_csv
from src.planning.margin_study import load_margin_path


def recompute(r):
    path=Path(r['run_directory'])/'tracking.csv'
    with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    q=np.array([[float(x[f'q_actual_{j}']) for j in range(1,8)] for x in rows])
    ref=np.array([[float(x[f'q_ref_{j}']) for j in range(1,8)] for x in rows])
    vel=np.array([[float(x[f'dq_actual_{j}']) for j in range(1,8)] for x in rows])
    active=np.array([x['phase'] in ('trajectory','settle') for x in rows]);errors=q[active]-ref[active]
    tcp_error=np.array([float(x['tcp_error_m']) for x in rows])
    gaps=np.array([float(x['minimum_clearance_m']) for x in rows]);valid=np.array([x['safety_valid']=='True' for x in rows])
    native=np.array([int(x['native_contact_count']) for x in rows]);prohibited=np.array([int(x['prohibited_contact_count']) for x in rows])
    bundle=load_margin_path(r['source_planning_json']);bundle.checker.state_valid(bundle.path[-1]);goal_tcp=bundle.checker.robot.tool_pose()[0]
    last_tcp=np.array([float(rows[-1][f'tcp_actual_{j}']) for j in range(3)])
    first_invalid=next((float(x['time']) for x,v in zip(rows,valid) if not v),None)
    def intervals(binary):return int(np.sum(binary & ~np.r_[False,binary[:-1]]))
    check=dict(joint_rmse_rad=np.sqrt(np.mean(errors*errors,axis=0)),joint_max_abs_error_rad=np.max(np.abs(errors),axis=0),
        tcp_rmse_m=float(np.sqrt(np.mean(tcp_error[active]**2))),tcp_max_error_m=float(np.max(tcp_error[active])),
        minimum_actual_clearance_m=float(np.min(gaps)),simulation_time_s=float(rows[-1]['time']),physics_steps=len(rows),
        safety_violation_steps=int(np.sum(~valid)),prohibited_contact_observations=int(np.sum(prohibited)),
        native_contact_observations=int(np.sum(native)),final_joint_max_error_rad=float(np.max(np.abs(q[-1]-bundle.path[-1]))),
        final_tcp_error_m=float(np.linalg.norm(last_tcp-goal_tcp)),
        path_max_deviation_rad=max(float(x['path_deviation_rad']) for x,a in zip(rows,active) if a))
    for key,v in check.items():np.testing.assert_allclose(v,r[key],atol=1e-12,rtol=1e-10,err_msg=key)
    tail=rows[-250:]
    stable_tail=len(tail)==250 and all(x['phase']=='settle' for x in tail)
    if stable_tail:
        qtail=q[-250:];vtail=vel[-250:]
        actual_tcp=np.array([[float(x[f'tcp_actual_{j}']) for j in range(3)] for x in tail])
        stable_tail=bool(np.max(np.abs(qtail-bundle.path[-1]))<=.02 and np.max(np.abs(vtail))<=.02 and np.max(np.linalg.norm(actual_tcp-goal_tcp,axis=1))<=.010)
    reconstructed_success=bool(valid.all() and not np.any(prohibited) and stable_tail)
    assert reconstructed_success==r['success']
    if r['first_violation']:
        assert abs(first_invalid-r['first_violation']['time'])<1e-10
        hold=[x for x in rows if x['phase']=='protective_hold']
        assert len(hold)>=250 and len({x['trajectory_time'] for x in hold})==1
    contact_starts=[float(x['time']) for i,x in enumerate(rows) if prohibited[i]>0 and (i==0 or prohibited[i-1]==0)]
    return dict(planning_clearance_m=r['planning_clearance_m'],planning_seed=r['planning_seed'],speed_rad_s=r['reference_speed_rad_s'],
        all_metrics_match=True,reconstructed_success=reconstructed_success,first_violation_time_s=first_invalid,
        forbidden_contact_intervals=intervals(prohibited>0),first_forbidden_contact_time_s=contact_starts[0] if contact_starts else None,
        minimum_actual_clearance_m=check['minimum_actual_clearance_m'],source_tracking_csv=str(path),recomputed=check)


def main():
    run=Path(json.loads((ROOT/'results/phase2d_dynamic_latest.json').read_text())['run'])
    results=json.loads((run/'complete.json').read_text())['results'];audits=[]
    for r in results:
        audits.append(recompute(r));print('AUDIT',r['planning_clearance_m'],r['planning_seed'],r['reference_speed_rad_s'],flush=True)
    write_json(ROOT/'results/phase2d_csv_recomputation.json',dict(audits=audits,all_metrics_match=all(x['all_metrics_match'] for x in audits),
        trials=len(audits),contact_intervals=sum(x['forbidden_contact_intervals'] for x in audits)))
    write_csv(ROOT/'results/phase2d_csv_recomputation.csv',[{k:v for k,v in a.items() if k!='recomputed'} for a in audits])
    print('All',len(audits),'raw CSV executions recomputed',flush=True)


if __name__=='__main__':main_guard(main)
