"""Bounded and frozen uncertainty pilot; train/validation before heldout dynamics."""
if __package__:
    from scripts._bootstrap import ROOT,destination,main_guard
else:
    from _bootstrap import ROOT,destination,main_guard
import argparse,copy,csv,concurrent.futures,json,time
from pathlib import Path
import mujoco
import numpy as np
from src.common import write_json,write_csv,sha
from src.control.validated_path import ValidatedPath,load_verified_path
from src.control.trajectory_executor import TrajectoryExecutor,CSVStream
from src.control.uncertain_models import model_pair,signature
from src.control.velocity_feedforward_controller import NominalFeedforwardController
from src.planning.collision_checker import CollisionChecker
from src.planning.rrt_connect import RRTConnectPlanner,PlannerConfig
from scripts.phase2a_cases import park,snapshot_scene,restore_scene

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def design():return read(ROOT/'configs/phase3a_pilot.json')
def prepared():return Path(read(ROOT/'results/phase3a_prepare_latest.json')['run'])
def frozen_sources():
    names=['configs/phase3a_pilot.json','src/control/uncertain_models.py','src/control/velocity_feedforward_controller.py',
           'scripts/run_phase3a.py','src/control/trajectory_executor.py','src/robot/joint_controller.py','src/planning/collision_checker.py']
    return {p:sha(ROOT/p) for p in names}
def check_lock():
    cfg=read(prepared()/'frozen_design.json')
    if cfg['source_hashes']!=frozen_sources():raise ValueError('Frozen design/source changed; refuse mixed experimental provenance')
    return cfg

def freeze_history(dest):
    paths=set()
    for folder in ('src','configs','docs','tests'):
        paths.update(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'phase3' not in p.name.lower() and p.name not in ('uncertain_models.py','velocity_feedforward_controller.py') and not p.name.startswith(('18_','19_')))
    paths.update((ROOT/'models').rglob('*.xml'))
    paths.update(p for p in (ROOT/'results').glob('*') if p.is_file() and ('phase2' in p.name or p.name=='model_provenance.json'))
    for label in ('phase2e_batch_latest.json','phase2e_static_latest.json','phase2e_shadow_latest.json','phase2e_controller_video_latest.json','phase2e_guard_video_latest.json'):
        paths.update(p for p in Path(read(ROOT/'results'/label)['run']).rglob('*') if p.is_file())
    paths.update((ROOT/'assets/videos').glob('phase2*.mp4'))
    manifest={str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}
    write_json(dest/'history_sha256.json',manifest)
    return len(manifest)

def prepare(dest):
    count=freeze_history(dest);cfg=design()
    write_json(dest/'frozen_design.json',dict(config=cfg,source_hashes=frozen_sources(),history_files=count,
        frozen_before_planning_and_dynamics=True,created_time_ns=time.time_ns()))
    write_json(ROOT/'results/phase3a_prepare_latest.json',dict(run=dest))
    plans=[]
    for layout in cfg['layouts']:
        c=CollisionChecker(options={'minimum_clearance_m':cfg['planning_clearance_m']});park(c)
        a,b=c.home.copy(),c.home.copy();a[0]=layout['start_joint1_rad'];b[0]=layout['goal_joint1_rad']
        a[1]=b[1]=layout['joint2_rad'];cross=a.copy();cross[0]=layout['crossing_angle_rad']
        c.state_valid(cross)
        finger=next(g for g in c.robot_geoms if 'left_finger_fingertip_pad_collision_1' in c.name(g))
        c.set_obstacle_pose(layout['obstacle'],c.data.geom_xpos[finger].copy())
        scene=snapshot_scene(c);direct=c.edge_valid(a,b,step_rad=.005).to_dict()
        accepted=False
        for seed in cfg['planning_seeds'][:cfg['maximum_planning_attempts_per_layout']]:
            planner=RRTConnectPlanner(c,PlannerConfig(),seed);r=planner.plan(a,b)
            source=dest/f"{layout['name']}_seed{seed}.json"
            record=dict(layout=layout,scene=scene,q_start=a,q_goal=b,planning_seed=seed,
                model_signature=signature(c.model),planning_clearance_m=cfg['planning_clearance_m'],
                checker_options=dict(c.options),direct=direct,result=r.to_dict())
            write_json(source,record);plans.append(dict(layout=layout['name'],split=layout['split'],seed=seed,
                success=r.success,final_validation_passed=r.final_validation_passed,source_json=str(source),
                failure_reason=r.failure_reason,minimum_clearance_m=r.minimum_sampled_clearance,planning_s=r.planning_time))
            print('PLAN',layout['name'],seed,r.success,r.failure_reason,flush=True)
            if r.success and r.final_validation_passed:
                write_csv(source.with_suffix('.csv'),[dict(node=i,**{f'q{j+1}':v for j,v in enumerate(q)}) for i,q in enumerate(r.path)])
                accepted=True;break
        if not accepted:print('UNSOLVED WITHIN BUDGET',layout['name'],flush=True)
    write_json(dest/'plans.json',plans);write_csv(dest/'planning_attempts.csv',plans)

def load_new(source):
    source=Path(source);record=read(source);result=record['result'];q=np.array(result['path'],dtype=float)
    if not result['success'] or not result['final_validation_passed']:raise ValueError('Unvalidated plan')
    c=CollisionChecker(options={'minimum_clearance_m':record['planning_clearance_m']});restore_scene(c,record['scene'])
    if signature(c.model)!=record['model_signature']:raise ValueError('Model signature changed')
    if q.ndim!=2 or q.shape[1]!=7 or not np.isfinite(q).all():raise ValueError('Invalid path array')
    if not np.array_equal(q[0],record['q_start']) or not np.array_equal(q[-1],record['q_goal']):raise ValueError('Endpoints changed')
    with source.with_suffix('.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    if not np.array_equal(q,np.array([[float(x[f'q{j}']) for j in range(1,8)] for x in rows])):raise ValueError('Path CSV differs')
    audits=[c.state_valid(x).to_dict() for x in q]+[c.edge_valid(a,b,step_rad=.005).to_dict() for a,b in zip(q,q[1:])]
    if not all(x['valid'] for x in audits):raise ValueError('Independent fine revalidation failed')
    c=CollisionChecker(options={'minimum_clearance_m':.005});restore_scene(c,record['scene'])
    evidence=dict(scene=record['layout']['name'],split=record['layout']['split'],seed=record['planning_seed'],verified=True,
        source_json=str(source),source_json_sha256=sha(source),path_csv=str(source.with_suffix('.csv')),
        path_csv_sha256=sha(source.with_suffix('.csv')),planning_minimum_clearance_m=result['minimum_sampled_clearance'],
        scene_configuration=dict(scene=record['scene'],q_start=q[0].tolist(),q_goal=q[-1].tolist()),
        current_fine_audits=audits,model_signature=record['model_signature'])
    return ValidatedPath(q,c,evidence)

def bundle(spec):
    return load_verified_path('D_two_obstacles',42) if spec['kind']=='historical' else load_new(spec['source_json'])

class UncertainRun:
    def __init__(self,b,method,perturbation,seed,dest):
        self.dest=Path(dest);self.dest.mkdir(parents=True,exist_ok=True);self.method=method;self.perturbation=perturbation;self.seed=seed
        n,p,self.parameters=model_pair(b.checker.model,perturbation)
        check=CollisionChecker(p,options={'minimum_clearance_m':.005});restore_scene(check,b.evidence['scene_configuration']['scene'])
        if check.pairs!=b.checker.pairs:raise ValueError('Changed collision pairs')
        self.e=TrajectoryExecutor(ValidatedPath(b.path,check,b.evidence),design()['speed_rad_s'],self.dest)
        self.nominal=n
        self.e.controller=NominalFeedforwardController(self.e.robot,n,method,perturbation,seed,lambda:bool(self.e.failure_reason))
        self.e.options['maximum_wall_s']=design()['maximum_wall_s_per_execution']
        mujoco.mj_saveModel(n,str(self.dest/'nominal_controller_model.mjb'),None)
        mujoco.mj_saveModel(self.e.model,str(self.dest/'perturbed_physics_model.mjb'),None)
        write_json(self.dest/'model_parameters.json',dict(**self.parameters,nominal_mjb_sha256=sha(self.dest/'nominal_controller_model.mjb'),
            physics_mjb_sha256=sha(self.dest/'perturbed_physics_model.mjb'),seed=seed,
            isolation='controller receives nominal model only; physical model parameters used solely by simulator and monitor'))
        self.signals=CSVStream(self.dest/'control_signals.csv');self.control_cost=0.;self.saturation=0;self.rate_count=0;self.ff_clipped=0
    def run(self,display=None,stop_requested=None):
        e=self.e
        def observe(ex,row):
            last=e.controller.last;self.control_cost+=last['algorithm_s'];self.rate_count+=int(last['rate_limited']);self.ff_clipped+=int(last['ff_clipped'])
            ids=e.robot.actuator_ids;force=e.data.actuator_force[ids];bounds=e.model.actuator_forcerange[ids]
            saturated=np.any((force<=bounds[:,0]+1e-8)|(force>=bounds[:,1]-1e-8));self.saturation+=int(saturated)
            signal=dict(time=row['time'],physics_step=row['physics_step'],algorithm_s=last['algorithm_s'],saturated=bool(saturated),rate_limited=last['rate_limited'])
            for key in ('observed_q','noise','reference_velocity','gravity_nm','gravity_offset','velocity_offset','desired_before_delay','delayed_target'):
                for i,v in enumerate(last[key]):signal[f'{key}_{i+1}']=float(v)
            for i,v in enumerate(force):signal[f'actuator_force_{i+1}']=float(v)
            self.signals.append(signal)
            if display:display(self,row)
        try:r=e.run(on_step=observe,stop_requested=stop_requested)
        finally:self.signals.close()
        invalid=bool(r['errors']) or not np.isfinite(e.data.qpos).all() or not np.isfinite(e.data.qvel).all() or np.any(e.data.warning.number)
        r.update(method=self.method,perturbation=self.perturbation['name'],disturbance_seed=self.seed,
            status='invalid_model_or_runtime' if invalid else 'success' if r['success'] else 'user_abort' if r['failure_reason']=='viewer_closed' else 'failure',
            valid_control_task=not invalid,nominal_signature=self.parameters['nominal_signature'],physics_signature=self.parameters['physics_signature'],
            saturation_steps=self.saturation,effective_rate_limit_steps=self.rate_count,ff_clipped_steps=self.ff_clipped,
            controller_extra_us_per_step=1e6*self.control_cost/max(1,e.steps),split=e.bundle.evidence.get('split','historical'),
            warning_counts=e.data.warning.number.tolist(),model_parameter_file=str(self.dest/'model_parameters.json'),
            controller_sources=frozen_sources(),supervisory_observation_scope=design()['observation_scope'])
        write_json(self.dest/'phase3a_summary.json',r);return r

def job(args):
    spec,method,perturbation,seed,dest=args
    return UncertainRun(bundle(spec),method,perturbation,seed,Path(dest)).run()

def stage_static(dest):
    # Real holding trajectories at two distinct legal configurations, obstacles parked.
    results=[]
    for i in range(2):
        c=CollisionChecker();park(c);q=c.home.copy();q[0]=.2*i;q[1]=-.15*i
        s=c.state_valid(q)
        if not s.valid:raise ValueError('Invalid static target')
        source=dest/f'static{i}.json';path=source.with_suffix('.csv');write_json(source,dict(q=q,scene=snapshot_scene(c)));write_csv(path,[{f'q{j+1}':v for j,v in enumerate(q)}])
        b=ValidatedPath(q[None,:],c,dict(scene=f'static{i}',split='static',seed=0,verified=True,source_json=str(source),source_json_sha256=sha(source),
            path_csv=str(path),path_csv_sha256=sha(path),planning_minimum_clearance_m=s.minimum_clearance,scene_configuration=dict(scene=snapshot_scene(c))))
        for method in ('M1','M3'):
            r=UncertainRun(b,method,{'name':'nominal'},3101,dest/f'static{i}_{method}').run();results.append(r)
            print('STATIC',i,method,r['success'],r['final_tcp_error_m'],flush=True)
    write_json(dest/'complete.json',dict(results=results,passed=all(r['success'] and not r['saturation_steps'] for r in results)))

def cohort(stage):
    specs=[dict(kind='historical',split='historical') ] if stage=='pilot' else []
    for p in read(prepared()/'plans.json'):
        if p['success'] and ((p['split'] in ('train','validation')) if stage=='pilot' else p['split']=='test'):
            specs.append(dict(kind='new',source_json=p['source_json'],split=p['split']))
    return specs

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',choices=['prepare','static','pilot','test'],required=True)
    parser.add_argument('--workers',type=int,choices=[1,2,3],default=3);a=parser.parse_args();dest=destination('phase3a_'+a.stage)
    if a.stage=='prepare':prepare(dest);print('COMPLETE',dest,flush=True);return
    lock=check_lock()
    if a.stage=='static':stage_static(dest)
    else:
        static=Path(read(ROOT/'results/phase3a_static_latest.json')['run'])
        if not read(static/'complete.json')['passed']:raise ValueError('Static gate failed')
        if a.stage=='test':
            pilot=Path(read(ROOT/'results/phase3a_pilot_latest.json')['run'])
            if not (pilot/'complete.json').exists():raise ValueError('Pilot incomplete')
            write_json(dest/'heldout_opening.json',dict(source_hashes=frozen_sources(),pilot=str(pilot),opened_ns=time.time_ns(),no_tuning_after_opening=True))
        jobs=[]
        for spec in cohort(a.stage):
            name='historical_D' if spec['kind']=='historical' else read(spec['source_json'])['layout']['name']
            for p in design()['perturbations']:
                seeds=design()['disturbance_seeds'][:1] if p['name']=='nominal' else design()['disturbance_seeds']
                for seed in seeds:
                    for method in design()['methods']:jobs.append((spec,method,p,seed,str(dest/f"{name}_{p['name']}_{seed}_{method}")))
        if len(jobs)>design()['maximum_paired_dynamic_runs']:raise ValueError('Budget exceeded')
        write_json(dest/'cohort_before_execution.json',dict(jobs=jobs,config=design(),source_hashes=lock['source_hashes'],workers=a.workers))
        results=[]
        def accept(j,r):
            results.append(r);write_json(dest/'partial.json',results)
            print('RUN',len(results),len(jobs),Path(j[-1]).name,r.get('status'),r.get('failure_reason'),r.get('minimum_actual_clearance_m'),flush=True)
        if a.workers==1:
            for j in jobs:accept(j,job(j))
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
                futures={pool.submit(job,j):j for j in jobs}
                for f in concurrent.futures.as_completed(futures):
                    j=futures[f]
                    try:r=f.result()
                    except Exception as exc:r=dict(status='invalid_model_or_runtime',valid_control_task=False,error=repr(exc),run_directory=j[-1],method=j[1],perturbation=j[2]['name'],disturbance_seed=j[3])
                    accept(j,r)
        write_json(dest/'complete.json',dict(results=results,stage=a.stage,source_hashes=frozen_sources()))
    write_json(ROOT/f'results/phase3a_{a.stage}_latest.json',dict(run=dest));print('COMPLETE',dest,flush=True)

if __name__=='__main__':main_guard(main)
