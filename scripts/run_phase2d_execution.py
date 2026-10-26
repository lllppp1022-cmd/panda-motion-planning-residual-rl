"""Execute the complete bounded planning cohort; process isolation, no RL."""
from _bootstrap import destination,main_guard
import concurrent.futures,json,shutil
from pathlib import Path
from src.common import ROOT,write_json,write_csv,sha
from src.planning.margin_study import load_margin_path
from src.control.trajectory_executor import TrajectoryExecutor


def execute_job(job):
    row,speed,dest=job;bundle=load_margin_path(row['source_json'])
    if bundle.checker.options['minimum_clearance_m']!=.005:raise RuntimeError('Execution threshold changed')
    e=TrajectoryExecutor(bundle,speed,Path(dest));r=e.run()
    r.update(planning_seed=row['planning_seed'],planning_clearance_m=row['planning_clearance_m'],execution_clearance_m=.005,
        deterministic_repeat=1,source_planning_json=row['source_json'],path_length_rad=row['path_length_rad'],
        planning_time_s=row['planning_time_s'],first_violation_time_s=None if not r['first_violation'] else r['first_violation']['time'],
        triggered_restricted_hold=bool(r['failure_reason']),planning_minus_actual_gap_m=None if r['minimum_actual_clearance_m'] is None else r['planning_minimum_clearance_m']-r['minimum_actual_clearance_m'])
    write_json(Path(dest)/'study_execution_summary.json',r);return r


def main():
    plan=Path(json.loads((ROOT/'results/phase2d_planning_latest.json').read_text())['run'])
    rows=json.loads((plan/'complete.json').read_text())['results'];dest=destination('phase2d_dynamic')
    jobs=[]
    for row in rows:
        if not row['success']:continue
        for speed in [.10,.15]:
            run=dest/f"margin{row['planning_clearance_m']*1000:02.0f}mm_seed{row['planning_seed']}_v{speed:.2f}"
            jobs.append((row,speed,str(run)))
    write_json(dest/'budget_decision_before_execution.json',dict(planning_run=str(plan),planning_trials=len(rows),
        pilot_seeds=[42,123,2026],planning_seed_expansion_executed=False,planned_dynamic_trials=len(jobs),
        repetitions_per_path_speed=1,maximum_workers=3,maximum_simulation_s_per_execution=120,maximum_wall_s_per_execution=180,
        rationale='Execute every fine-validated path at both frozen speeds before expanding the seed cohort; bounded small-sample feasibility study.',
        source_sha256={p:sha(ROOT/p) for p in ['scripts/run_phase2d_execution.py','src/planning/margin_study.py','src/control/trajectory_executor.py']}))
    results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(execute_job,j):j for j in jobs}
        for future in concurrent.futures.as_completed(futures):
            job=futures[future]
            try:r=future.result()
            except Exception as exc:
                r=dict(planning_clearance_m=job[0]['planning_clearance_m'],planning_seed=job[0]['planning_seed'],
                    reference_speed_rad_s=job[1],success=False,failure_reason='worker_exception:'+repr(exc),run_directory=job[2])
            results.append(r);results.sort(key=lambda x:(x['planning_clearance_m'],x['planning_seed'],x['reference_speed_rad_s']))
            write_csv(dest/'phase2d_dynamic_execution_results.csv',results)
            print('DYNAMIC',r['planning_clearance_m'],r['planning_seed'],r['reference_speed_rad_s'],r['success'],r['failure_reason'],
                  'gap',r.get('minimum_actual_clearance_m'),'sim',r.get('simulation_time_s'),flush=True)
    write_json(dest/'complete.json',dict(results=results,planning_run=str(plan),all_planned_jobs=len(results)==len(jobs)))
    shutil.copy2(dest/'phase2d_dynamic_execution_results.csv',ROOT/'results/phase2d_dynamic_execution_results.csv')
    write_json(ROOT/'results/phase2d_dynamic_latest.json',dict(run=dest));print('COMPLETE',dest,flush=True)


if __name__=='__main__':main_guard(main)
