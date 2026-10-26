"""Independent CSV reconstruction for every actual Phase2E execution."""
from _bootstrap import main_guard
import csv,json
from pathlib import Path
import numpy as np
from src.common import ROOT,write_json,write_csv


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def truth(x):return str(x).lower() in ('true','1','1.0')


def audit(r):
    folder=Path(r['run_directory'])
    with (folder/'tracking.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    q=np.array([[float(x[f'q_actual_{j}']) for j in range(1,8)] for x in rows]);ref=np.array([[float(x[f'q_ref_{j}']) for j in range(1,8)] for x in rows])
    ctrl=np.array([[float(x[f'ctrl_{j}']) for j in range(1,8)] for x in rows]);v=np.array([[float(x[f'dq_actual_{j}']) for j in range(1,8)] for x in rows])
    active=np.array([x['phase'] in ('trajectory','settle') for x in rows]);err=q[active]-ref[active]
    gaps=np.array([float(x['minimum_clearance_m']) for x in rows]);bad=np.array([not truth(x['safety_valid']) for x in rows])
    contacts=np.array([int(x['prohibited_contact_count'])>0 for x in rows]);intervals=int(np.sum(contacts & ~np.r_[False,contacts[:-1]]))
    tcp=np.array([float(x['tcp_error_m']) for x in rows]);time=np.array([float(x['time']) for x in rows]);phase=np.array([x['phase'] for x in rows])
    values=dict(minimum_actual_clearance_m=float(np.min(gaps)),physics_steps=len(rows),simulation_time_s=float(time[-1]),
        safety_violation_steps=int(np.sum(bad)),contact_intervals=intervals,
        joint_rmse_rad=np.sqrt(np.mean(err*err,axis=0)),joint_max_abs_error_rad=np.max(abs(err),axis=0),
        tcp_rmse_m=float(np.sqrt(np.mean(tcp[active]**2))),tcp_max_error_m=float(np.max(tcp[active])))
    for k,value in values.items():np.testing.assert_allclose(value,r[k],atol=1e-12,rtol=1e-10,err_msg=k)
    firstbad=float(time[np.flatnonzero(bad)[0]]) if bad.any() else None
    firstcontact=float(time[np.flatnonzero(contacts)[0]]) if contacts.any() else None
    assert firstbad==r['first_hard_violation_time_s'] and firstcontact==r['first_contact_time_s']
    with (folder/'control_signals.csv').open(encoding='utf-8-sig',newline='') as f:signals=list(csv.DictReader(f))
    assert len(signals)==len(rows)
    spec=read(folder/'frozen_run_config.json');arm=spec['actuators']['arm_joints'];kp=np.array([x['gain'][0] for x in arm]);kd=np.array([-x['bias'][2] for x in arm])
    limits=np.array([x['forcerange'] for x in arm]);tau=np.array([[float(x[f'tau_actual_{j}']) for j in range(1,8)] for x in signals])
    raw=kp*(ctrl-q)-kd*v;np.testing.assert_allclose(tau,np.clip(raw,limits[:,0],limits[:,1]),atol=1e-8,rtol=1e-10)
    assert np.all(tau>=limits[:,0]-1e-9) and np.all(tau<=limits[:,1]+1e-9)
    sat=sum(any(truth(x[f'torque_saturated_{j}']) for j in range(1,8)) for x in signals)
    assert sat==r['torque_saturation_steps']
    initial_ctrl=np.array(spec['path_evidence']['scene_configuration']['q_start'])
    rates=np.diff(np.vstack([initial_ctrl,ctrl]),axis=0)/.002
    np.testing.assert_allclose(np.max(abs(rates)),r['max_ctrl_rate_rad_s'],atol=1e-12)
    np.testing.assert_allclose(np.sum(abs(rates),axis=0)*.002,r['control_total_variation_rad'],atol=1e-10)
    assert np.max(abs(rates))<=.35+1e-9
    effective=sum(truth(x['effective_rate_limited']) for x in signals);assert effective==r['effective_rate_limit_steps']
    stop_time=r['first_violation']['time'] if r['first_violation'] else None
    post= time>=stop_time-1e-9 if stop_time is not None else np.zeros(len(time),dtype=bool)
    steady=(len(rows)>=250 and np.all(phase[-250:]=='settle') and np.max(abs(q[-250:]-np.array(r['goal_q'])))<=.02
            and np.max(abs(v[-250:]))<=.02 and np.max(tcp[-250:])<=.010)
    assert bool(not bad.any() and not contacts.any() and steady)==r['success']
    return dict(run_directory=str(folder),method=r['method'],all_metrics_match=True,actual_signal_mapping_verified=True,
        hard_violation_time_s=firstbad,first_contact_time_s=firstcontact,contact_intervals=intervals,
        post_stop_min_gap_m=float(np.min(gaps[post])) if post.any() else None,
        post_stop_peak_velocity_rad_s=float(np.max(abs(v[post]))) if post.any() else None,
        stabilized_after_stop=bool(post.any() and np.max(abs(v[-1]))<=.02),effective_rate_limit_steps=effective)


def main():
    run=Path(read(ROOT/'results/phase2e_batch_latest.json')['run']);results=read(run/'complete.json')['results'];audits=[]
    for i,r in enumerate(results):
        audits.append(audit(r));print('CSV AUDIT',i+1,len(results),r['method'],r['outcome'],flush=True)
    write_json(ROOT/'results/phase2e_csv_audit.json',dict(actual_runs=len(audits),all_metrics_match=True,audits=audits))
    write_csv(ROOT/'results/phase2e_csv_audit.csv',audits)
    cohort=read(run/'cohort_before_execution.json');old_c=read(Path(read(ROOT/'results/phase2c_batch_latest.json')['run'])/'complete.json')['results']
    old_d=read(Path(read(ROOT/'results/phase2d_dynamic_latest.json')['run'])/'complete.json')['results'];reproductions=[]
    for r in results:
        if r['method']!='M0':continue
        uid=Path(r['run_directory']).name.split('_M0_')[0]
        spec=next(p['spec'] for p in cohort['unique_paths'] if p['uid']==uid)
        if spec['kind']=='phase2c':
            old=next(x for x in old_c if x['scene']==spec['scene'] and x['reference_speed_rad_s']==r['reference_speed_rad_s'])
        else:old=next(x for x in old_d if x['source_planning_json']==spec['source_json'] and x['reference_speed_rad_s']==r['reference_speed_rad_s'])
        assert r['success']==old['success'] and r['failure_reason']==old['failure_reason']
        for key in ['simulation_time_s','physics_steps','minimum_actual_clearance_m','final_q','joint_rmse_rad']:
            np.testing.assert_allclose(r[key],old[key],atol=1e-12,rtol=1e-10,err_msg=key)
        reproductions.append(dict(run=r['run_directory'],historical_source=old['run_directory'],matched=True))
    write_json(ROOT/'results/phase2e_m0_baseline_reproduction.json',dict(runs=len(reproductions),all_matched=True,comparisons=reproductions))
    videos=[]
    import imageio.v2 as imageio
    for kind,name in [('controller','phase2e_controller_comparison.mp4'),('guard','phase2e_safety_guard.mp4')]:
        folder=Path(read(ROOT/f'results/phase2e_{kind}_video_latest.json')['run']);done=read(folder/'complete.json');matched=0
        for ep in done['episodes']:
            case=folder/ep['method'];meta=read(case/'demo_complete.json');frames=read(case/'video_frames.json');lookup={x['physics_step']:x for x in frames};count=0
            assert meta['viewer_launched'] and meta['viewer_closed'] and ep['video']['all_same_live_data']
            with (case/'tracking.csv').open(encoding='utf-8-sig',newline='') as f:
                for row in csv.DictReader(f):
                    frame=lookup.get(int(row['physics_step']))
                    if frame is None:continue
                    assert frame['time']==float(row['time'])
                    for field in ['q_actual','q_ref','ctrl']:
                        np.testing.assert_array_equal([float(row[f'{field}_{j}']) for j in range(1,8)],frame[field])
                    count+=1
            assert count==len(frames);matched+=count
        with imageio.get_reader(str(ROOT/'assets/videos'/name)) as reader:decoded=sum(1 for _ in reader)
        assert decoded==matched==done['frames']
        videos.append(dict(kind=kind,decoded_frames=decoded,frames_matched_to_actual_tracking=matched,native_viewers_closed=True,all_same_live_data=True))
    write_json(ROOT/'results/phase2e_video_audit.json',dict(all_matched=True,videos=videos))
    print('M0 baseline and both live videos verified',flush=True)


if __name__=='__main__':main_guard(main)
