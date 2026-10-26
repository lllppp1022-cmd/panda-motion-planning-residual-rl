"""Staged static -> shadow -> active pilot -> bounded M0/M1/M2 cohort."""
from _bootstrap import destination,main_guard
import argparse,concurrent.futures,csv,hashlib,json
from pathlib import Path
import numpy as np
from src.common import ROOT,write_json,write_csv,sha
from src.control.validated_path import load_verified_path,ValidatedPath
from src.control.phase2e_runtime import ControllerRun,config
from src.planning.margin_study import load_margin_path
from src.planning.collision_checker import CollisionChecker
from scripts.phase2a_cases import park,snapshot_scene


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def static_bundle(name,folder):
    c=CollisionChecker();park(c)
    q=c.home.copy()
    if name=='D_start':q=load_verified_path('D_two_obstacles').path[0].copy()
    if name=='D_critical_reference':q=np.array(read(ROOT/'results/phase2d_failure_analysis.json')['historical_failures'][0]['q_ref'])
    state=c.state_valid(q)
    if not state.valid:raise ValueError('Invalid static probe')
    folder.mkdir(parents=True);csvpath=folder/'static_reference.csv';source=folder/'static_reference.json'
    write_csv(csvpath,[dict(node=0,**{f'q{i+1}':v for i,v in enumerate(q)})])
    write_json(source,dict(type='new static diagnostic, NOT an RRT path',case=name,q=q,scene=snapshot_scene(c),validation=state.to_dict()))
    evidence=dict(scene='static_'+name,seed=0,verified=True,source_json=str(source),source_json_sha256=sha(source),
        path_csv=str(csvpath),path_csv_sha256=sha(csvpath),planning_minimum_clearance_m=state.minimum_clearance,
        scene_configuration=dict(scene=snapshot_scene(c),q_start=q.tolist(),q_goal=q.tolist()))
    return ValidatedPath(q[None,:],c,evidence)


def load_spec(spec):
    return load_verified_path(spec['scene'],42) if spec['kind']=='phase2c' else load_margin_path(spec['source_json'])


def cohort():
    conditions=[];unique={};plan=Path(read(ROOT/'results/phase2d_planning_latest.json')['run'])
    specs=[dict(kind='phase2c',scene=s,group='Phase2C',planning_seed=42,planning_clearance_m=.005) for s in ['A_clear','B_sphere','C_cylinder','D_two_obstacles']]
    for r in read(plan/'complete.json')['results']:
        if not r['success']:continue
        specs.append(dict(kind='phase2d',scene='D_two_obstacles',group='Phase2D',planning_seed=r['planning_seed'],
            planning_clearance_m=r['planning_clearance_m'],source_json=r['source_json']))
    for spec in specs:
        b=load_spec(spec);payload=json.dumps(dict(path=b.path.tolist(),scene=b.evidence['scene_configuration']),sort_keys=True).encode()
        identity=hashlib.sha256(payload).hexdigest();uid='path_'+identity[:12]
        if identity not in unique:unique[identity]=dict(uid=uid,spec=spec,identity=identity)
        conditions.append(dict(**spec,path_uid=uid,identity=identity))
    return conditions,list(unique.values())


def job(args):
    spec,method,speed,dest,mode=args
    r=ControllerRun(load_spec(spec),method,speed,Path(dest),guard_mode=mode).run()
    return r


def gate_shadow(results,dest):
    comparisons=[];leads=[]
    for r in results:
        folder=Path(r['run_directory']);predictions=read(folder/'predictions.json')
        with (folder/'tracking.csv').open(encoding='utf-8-sig',newline='') as f:actual={int(x['physics_step']):x for x in csv.DictReader(f)}
        for p in predictions:
            for sample in p['samples']:
                row=actual.get(sample['target_physics_step'])
                if row is None or row['phase']=='protective_hold':continue
                q=np.array([float(row[f'q_actual_{j}']) for j in range(1,8)])
                comparisons.append(dict(run=str(folder),method=r['method'],observed_time=p['observed_time'],target_time=sample['time'],
                    q_max_error_rad=float(np.max(np.abs(q-sample['q']))),
                    gap_error_m=abs(float(row['minimum_clearance_m'])-sample['minimum_gap_m'])))
        if r['first_hard_violation_time_s'] is not None:
            leads.append(dict(method=r['method'],source=folder,warning=r['first_warning_time_s'],violation=r['first_hard_violation_time_s'],
                lead_s=None if r['first_warning_time_s'] is None else r['first_hard_violation_time_s']-r['first_warning_time_s']))
    maximum_q=max(x['q_max_error_rad'] for x in comparisons);maximum_gap=max(x['gap_error_m'] for x in comparisons)
    calls=[p['computation_s'] for r in results for p in read(Path(r['run_directory'])/'predictions.json')]
    gate=dict(comparisons=len(comparisons),max_joint_prediction_error_rad=maximum_q,max_gap_prediction_error_m=maximum_gap,
        p95_prediction_time_s=float(np.percentile(calls,95)),warning_leads=leads,
        passed=bool(maximum_q<=1e-5 and maximum_gap<=.0001 and all(x['lead_s'] is not None and x['lead_s']>0 for x in leads)
             and np.percentile(calls,95)<.10),
        caution='Nominal model consistency only; no continuous safety or disturbance guarantee; not hard realtime')
    write_csv(dest/'shadow_prediction_validation.csv',comparisons);write_json(dest/'shadow_gate.json',gate);return gate


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['static','shadow','active_pilot','batch'],required=True)
    a=p.parse_args();dest=destination('phase2e_'+a.stage);results=[]
    if a.stage=='static':
        for name in config()['static_cases']:
            bundle=static_bundle(name,dest/(name+'_reference'))
            for method in ['M0','M1']:
                r=ControllerRun(bundle,method,.10,dest/(name+'_'+method)).run();r['static_case']=name
                with (Path(r['run_directory'])/'tracking.csv').open(encoding='utf-8-sig',newline='') as f:tail=list(csv.DictReader(f))[-250:]
                q=np.array([[float(x[f'q_actual_{j}']) for j in range(1,8)] for x in tail])
                r['steady_joint_peak_to_peak_rad']=np.ptp(q,axis=0)
                r['steady_velocity_max_rad_s']=max(abs(float(x[f'dq_actual_{j}'])) for x in tail for j in range(1,8))
                write_json(Path(r['run_directory'])/'phase2e_summary.json',r);results.append(r)
                print('STATIC',name,method,r['success'],r['final_tcp_error_m'],flush=True)
        stable=all(r['success'] and not r['torque_saturation_steps'] and max(r['steady_joint_peak_to_peak_rad'])<1e-4 for r in results)
        improved=all(next(r for r in results if r['static_case']==name and r['method']=='M1')['final_tcp_error_m']<next(r for r in results if r['static_case']==name and r['method']=='M0')['final_tcp_error_m'] for name in config()['static_cases'])
        write_json(dest/'static_gate.json',dict(passed=stable and improved,stable=stable,all_cases_improved=improved))
    elif a.stage=='shadow':
        static=Path(read(ROOT/'results/phase2e_static_latest.json')['run'])
        if not read(static/'static_gate.json')['passed']:raise ValueError('Static compensation gate failed')
        plan=Path(read(ROOT/'results/phase2d_planning_latest.json')['run'])
        specs=[dict(kind='phase2c',scene='D_two_obstacles'),dict(kind='phase2d',source_json=str(plan/'margin08mm_seed123.json'))]
        for i,spec in enumerate(specs):
            for method in ['M0','M1']:
                r=job((spec,method,.15,str(dest/f'case{i}_{method}_shadow'),'shadow'));results.append(r)
                print('SHADOW',i,method,r['outcome'],r['first_warning_time_s'],r['first_hard_violation_time_s'],flush=True)
        gate=gate_shadow(results,dest);print('GATE',gate,flush=True)
    else:
        shadow=Path(read(ROOT/'results/phase2e_shadow_latest.json')['run'])
        if not read(shadow/'shadow_gate.json')['passed']:raise ValueError('Shadow validation gate failed; no active protection authorized by evidence')
        if a.stage=='active_pilot':
            plan=Path(read(ROOT/'results/phase2d_planning_latest.json')['run'])
            specs=[dict(kind='phase2c',scene='D_two_obstacles'),dict(kind='phase2d',source_json=str(plan/'margin08mm_seed123.json'))]
            for i,spec in enumerate(specs):
                r=job((spec,'M2',.15,str(dest/f'case{i}_M2_active'),'active'));results.append(r)
                print('ACTIVE PILOT',i,r['outcome'],r['first_warning_time_s'],r['minimum_actual_clearance_m'],flush=True)
        else:
            active=Path(read(ROOT/'results/phase2e_active_pilot_latest.json')['run'])
            if not (active/'complete.json').exists():raise ValueError('Active pilot must finish before formal batch')
            if not all(not r.get('guard_fault') and (r['success'] or r['safely_stopped_without_goal']) for r in read(active/'complete.json')['results']):
                raise ValueError('Active pilot did not reach a goal or a stable safe protective stop; refusing expansion')
            conditions,paths=cohort();jobs=[]
            write_json(dest/'cohort_before_execution.json',dict(conditions=conditions,unique_paths=paths,
                actual_executions=len(paths)*6,condition_rows=len(conditions)*6,deterministic_repeats=1,
                duplicate_policy='identical path AND scene reused with explicit actual execution id; never count as independent trials',
                methods=['M0','M1','M2'],speeds=[.10,.15],workers=3,parameter_perturbations_executed=False,
                config_sha256=sha(ROOT/'configs/phase2e_controllers.yaml')))
            for path in paths:
                for method in ['M0','M1','M2']:
                    for speed in [.10,.15]:jobs.append((path['spec'],method,speed,str(dest/f"{path['uid']}_{method}_v{speed:.2f}"),None))
            with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
                futures={pool.submit(job,j):j for j in jobs}
                for future in concurrent.futures.as_completed(futures):
                    j=futures[future]
                    try:r=future.result()
                    except Exception as exc:r=dict(method=j[1],reference_speed_rad_s=j[2],outcome='worker_exception',success=False,failure_reason=repr(exc),run_directory=j[3])
                    results.append(r);write_json(dest/'partial_results.json',results)
                    print('BATCH',Path(j[3]).name,r['outcome'],r.get('minimum_actual_clearance_m'),flush=True)
    write_json(dest/'complete.json',dict(stage=a.stage,results=results))
    write_json(ROOT/f'results/phase2e_{a.stage}_latest.json',dict(run=dest));print('COMPLETE',dest,flush=True)


if __name__=='__main__':main_guard(main)
