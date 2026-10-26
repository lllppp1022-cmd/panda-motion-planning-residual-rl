"""Actual ctrl + mj_step execution; only initialize() may write live qpos."""
import copy,csv,json,time
from contextlib import nullcontext
from dataclasses import asdict
import numpy as np
import mujoco
import yaml
from src.common import ROOT,write_json,sha
from src.robot.panda_interface import PandaInterface
from src.robot.joint_controller import JointPositionController
from src.control.trajectory_generator import QuinticTrajectory
from src.control.validated_path import ValidatedPath


class CSVStream:
    def __init__(self,path):self.f=open(path,'w',encoding='utf-8-sig',newline='');self.writer=None
    def append(self,row):
        row={k:json.dumps(v,ensure_ascii=False,allow_nan=False) if isinstance(v,(dict,list)) else v for k,v in row.items()}
        if self.writer is None:self.writer=csv.DictWriter(self.f,fieldnames=list(row));self.writer.writeheader()
        self.writer.writerow(row)
    def close(self):self.f.close()


def options():return yaml.safe_load((ROOT/'configs/trajectory_execution.yaml').read_text(encoding='utf-8'))


class TrajectoryExecutor:
    def __init__(self,verified,speed,dest,*,initial_q=None):
        if not isinstance(verified,ValidatedPath) or not verified.evidence.get('verified'):raise ValueError('Verified saved path required')
        e=verified.evidence
        if sha(e['path_csv'])!=e['path_csv_sha256'] or sha(e['source_json'])!=e['source_json_sha256']:
            raise ValueError('Validated artifact changed')
        self.bundle=verified;self.options=options();self.dest=dest;dest.mkdir(parents=True,exist_ok=True)
        if speed not in self.options['reference_speeds_rad_s']:raise ValueError('Use a frozen reference speed')
        self.speed=float(speed);self.trajectory=QuinticTrajectory(verified.path,speed,self.options['reference_acceleration_rad_s2'],self.options['minimum_segment_duration_s'])
        self.model=copy.copy(verified.checker.model);self.data=mujoco.MjData(self.model)
        self.robot=PandaInterface(self.model,self.data);self.dt=float(self.model.opt.timestep)
        if abs(self.dt-.002)>1e-12:raise ValueError('Unaudited physics timestep')
        self.checker=verified.checker
        self.reference_data=mujoco.MjData(self.model);self.reference_robot=PandaInterface(self.model,self.reference_data)
        self.initialize(verified.path[0] if initial_q is None else initial_q)
        self.controller=JointPositionController(self.robot,max_velocity=self.options['command_slew_rad_s'])
        self.first_violation=None;self.rows=0;self.steps=0;self.command_limited=0;self.violations=0;self.prohibited_contacts=0
        self.tracking=[];self.last_row=None;self.last_clearance=None;self.failure_reason='';self.success=False
        self.expected_q=self.data.qpos.copy();self.gripper_deviation=0.
        self.integration_lock=nullcontext

    def initialize(self,q):
        """One explicit pre-simulation initialization, never callable after stepping."""
        if hasattr(self,'controller'):raise RuntimeError('Reinitialization after initialization is forbidden')
        q=self.robot.validate(q);m,d=self.model,self.data
        mujoco.mj_resetDataKeyframe(m,d,self.checker.home_id)
        d.qpos[self.robot.qpos_indices]=q
        d.qpos[self.checker.fingers]=self.checker.options['gripper_opening_m']
        d.mocap_pos[:]=self.checker.mocap_pos;d.mocap_quat[:]=self.checker.mocap_quat
        d.ctrl[self.robot.actuator_ids]=q
        mujoco.mj_forward(m,d)
        write_json(self.dest/'initialization.json',dict(mode='EXPLICIT PRE-SIMULATION INITIALIZATION ONLY',q=q,sim_time=d.time,
                   physics_steps=0,planned_start=self.bundle.path[0],not_a_home_to_start_motion=True))

    def _monitor(self):
        # Reuse full Phase2A state check every physics step. It assumes fixed
        # fingers. ALSO query all the identical audited pairs on live data,
        # capturing the actual (not exactly fixed) dynamically moving fingers.
        q=self.robot.positions();audit=self.checker.state_valid(q)
        m,d=self.model,self.data;maximum=self.checker.options['distance_query_max_m']
        distances=[float(mujoco.mj_geomDistance(m,d,p['id1'],p['id2'],maximum,None)) for p in self.checker.pairs]
        finite=np.isfinite(distances).all();minimum=min(distances) if finite else None
        threshold=self.checker.options['minimum_clearance_m']+self.checker.options['numerical_tolerance_m']
        offending=[dict(**p,distance_m=v) for p,v in zip(self.checker.pairs,distances) if not np.isfinite(v) or v<=threshold]
        contacts=[];forbidden=0
        for ct in d.contact:
            key=tuple(sorted((int(ct.geom1),int(ct.geom2))))
            checked=key in self.checker._pair_set
            contacts.append(dict(geom1=self.checker.name(ct.geom1),geom2=self.checker.name(ct.geom2),
                                 distance_m=float(ct.dist),audited_pair=checked))
            if checked and ct.dist<=self.checker.options['numerical_tolerance_m']:forbidden+=1
        self.gripper_deviation=max(self.gripper_deviation,float(np.max(np.abs(d.qpos[self.checker.fingers]-.04))))
        valid=audit.valid and finite and not offending and forbidden==0
        return dict(valid=bool(valid),minimum_clearance_m=minimum,minimum_is_lower_bound=bool(minimum is not None and minimum>=maximum),
                    fixed_gripper_audit_valid=audit.valid,fixed_gripper_minimum_m=audit.minimum_clearance,
                    offending_geom_pairs=offending or audit.offending_geom_pairs,native_contacts=contacts,
                    prohibited_contacts=forbidden,reason='' if valid else audit.rejection_reason or 'actual_live_clearance_or_contact')

    def _tcp_ref(self,q):
        # Independent reference FK, never the data rendered by the video.
        self.reference_data.qpos[:]=self.data.qpos
        self.reference_data.qpos[self.robot.qpos_indices]=q
        mujoco.mj_forward(self.model,self.reference_data)
        return self.reference_robot.tool_pose()[0]

    def _path_deviation(self,q):
        path=self.bundle.path
        if len(path)==1:return float(np.linalg.norm(q-path[0]))
        a,b=path[:-1],path[1:];delta=b-a
        u=np.clip(np.sum((q-a)*delta,axis=1)/np.maximum(np.sum(delta*delta,axis=1),1e-30),0,1)
        return float(np.min(np.linalg.norm(q-(a+u[:,None]*delta),axis=1)))

    def run(self,on_step=None,stop_requested=None):
        o=self.options;d=self.data;started=time.perf_counter();phase='prehold';clock=0.;elapsed=0.;stable=0.;protect=0.
        failure_q=None;failure_ref=None;init_stable=0.;previous_v=self.robot.velocities()
        streams={n:CSVStream(self.dest/f'{n}.csv') for n in ['tracking','clearance','events']}
        errors=[];min_gap=None;min_bound=False;max_velocity=0.;max_acceleration=0.;native_count=0
        def fail(reason,monitor=None):
            nonlocal phase,protect,failure_q,failure_ref
            if not self.failure_reason:
                self.failure_reason=reason;failure_q=self.robot.positions();failure_ref=failure_q.copy();phase='protective_hold';protect=0.
                self.first_violation=dict(time=float(d.time),q=failure_q.tolist(),reason=reason,
                    monitor=monitor,trajectory_time=clock)
                streams['events'].append(dict(time=d.time,event='FAILURE_STOP_ADVANCING_REFERENCE',detail=self.first_violation))
        try:
            write_json(self.dest/'frozen_run_config.json',dict(options=o,speed=self.speed,physics_dt_s=self.dt,control_frequency_hz=1/self.dt,
                monitor_frequency_hz=1/self.dt,trajectory_duration_s=self.trajectory.duration,durations_s=self.trajectory.durations,
                q_path=self.bundle.path,path_evidence=self.bundle.evidence,actuators=self.robot.inspection(),
                source_hashes={p:sha(ROOT/p) for p in ['src/control/trajectory_executor.py','src/control/trajectory_generator.py',
                    'src/control/validated_path.py','configs/trajectory_execution.yaml']},source='actual ctrl + mj_step; no live qpos writes after initialization'))
            initial=self._monitor()
            if np.max(np.abs(self.robot.positions()-self.bundle.path[0]))>o['initial_joint_tolerance_rad']:
                self.failure_reason='initial_configuration_mismatch';return self._finish(started,errors,min_gap,min_bound,max_velocity,max_acceleration,native_count)
            if not initial['valid']:
                self.failure_reason='invalid_initial_actual_state';self.first_violation=dict(time=0.,q=self.robot.positions().tolist(),monitor=initial)
                return self._finish(started,errors,min_gap,min_bound,max_velocity,max_acceleration,native_count)
            while True:
                if time.perf_counter()-started>o['maximum_wall_s'] or d.time>o['maximum_simulation_s']:
                    fail('execution_budget')
                if not np.array_equal(d.qpos,self.expected_q):fail('unexpected_external_live_state_edit')
                if stop_requested:
                    reason=stop_requested()
                    if reason:fail(str(reason))
                if phase=='prehold':ref=self.trajectory.sample(0.)
                elif phase=='trajectory':ref=self.trajectory.sample(min(self.trajectory.duration,clock+self.dt))
                elif phase=='settle':ref=self.trajectory.sample(self.trajectory.duration)
                else:
                    ref=self.trajectory.sample(clock)
                    ref=type(ref)(clock,failure_ref,np.zeros(7),np.zeros(7),ref.segment,ref.progress)
                desired=ref.q_ref
                self.command_limited+=int(np.any(np.abs(desired-self.controller.command_q)>o['command_slew_rad_s']*self.dt+1e-12))
                before=float(d.time)
                with self.integration_lock():self.controller.step(desired)
                self.steps+=1;self.expected_q=d.qpos.copy()
                if abs(d.time-before-self.dt)>1e-9:raise RuntimeError('MuJoCo integration time reset or mismatch')
                if phase=='trajectory':clock=ref.t
                elapsed+=self.dt
                q,v=self.robot.positions(),self.robot.velocities();acc=(v-previous_v)/self.dt;previous_v=v.copy()
                monitor=self._monitor();self.last_clearance=monitor
                if monitor['minimum_clearance_m'] is not None and (min_gap is None or monitor['minimum_clearance_m']<min_gap):
                    min_gap=monitor['minimum_clearance_m'];min_bound=monitor['minimum_is_lower_bound']
                native_count+=len(monitor['native_contacts']);self.prohibited_contacts+=monitor['prohibited_contacts']
                self.violations+=int(not monitor['valid'])
                actual_tcp=self.robot.tool_pose()[0];reference_tcp=self._tcp_ref(desired)
                max_velocity=max(max_velocity,float(np.max(np.abs(v))));max_acceleration=max(max_acceleration,float(np.max(np.abs(acc))))
                error=q-desired;tcp_error=float(np.linalg.norm(actual_tcp-reference_tcp))
                row=dict(time=float(d.time),physics_step=self.steps,phase=phase,trajectory_time=clock,segment=ref.segment,progress=ref.progress,
                    tcp_error_m=tcp_error,path_deviation_rad=self._path_deviation(q),minimum_clearance_m=monitor['minimum_clearance_m'],
                    command_rate_limited=self.command_limited,native_contact_count=len(monitor['native_contacts']),
                    prohibited_contact_count=monitor['prohibited_contacts'],safety_valid=monitor['valid'])
                for i in range(7):
                    for k,value in [('q_ref',desired[i]),('dq_ref',ref.dq_ref[i]),('ddq_ref',ref.ddq_ref[i]),('q_actual',q[i]),
                                    ('dq_actual',v[i]),('ddq_actual_fd',acc[i]),('qacc_mujoco',d.qacc[self.robot.dof_indices[i]]),
                                    ('ctrl',d.ctrl[self.robot.actuator_ids[i]])]:row[f'{k}_{i+1}']=float(value)
                for i in range(3):row[f'tcp_ref_{i}']=float(reference_tcp[i]);row[f'tcp_actual_{i}']=float(actual_tcp[i])
                self.last_row=row;self.rows+=1;streams['tracking'].append(row)
                streams['clearance'].append(dict(time=float(d.time),physics_step=self.steps,phase=phase,**monitor))
                if phase in ('trajectory','settle'):self.tracking.append(dict(error=error,tcp=tcp_error,path=row['path_deviation_rad']))
                if not monitor['valid']:fail('dynamic_safety_violation',monitor)
                if np.max(np.abs(error))>o['tracking_abort_rad']:fail('tracking_error_abort',monitor)
                if np.max(np.abs(v))>o['velocity_abort_rad_s']:fail('actual_velocity_abort',monitor)
                if phase=='prehold':
                    ok=np.max(np.abs(q-self.bundle.path[0]))<=o['initial_joint_tolerance_rad'] and np.max(np.abs(v))<=o['stable_velocity_rad_s']
                    init_stable=init_stable+self.dt if ok else 0.
                    if elapsed>=o['prehold_s']-1e-9:
                        if init_stable<o['stable_hold_s']-1e-9:fail('initial_hold_not_stable',monitor)
                        else:
                            phase='trajectory';elapsed=0.;streams['events'].append(dict(time=d.time,event='START_TRAJECTORY',detail='prehold passed'))
                elif phase=='trajectory' and clock>=self.trajectory.duration-1e-9:phase='settle';elapsed=0.
                elif phase=='settle':
                    ok=np.max(np.abs(q-self.bundle.path[-1]))<=o['goal_joint_tolerance_rad'] and tcp_error<=o['goal_tcp_tolerance_m'] and np.max(np.abs(v))<=o['stable_velocity_rad_s']
                    stable=stable+self.dt if ok else 0.
                    if stable>=o['stable_hold_s']-1e-9:self.success=True;phase='completed'
                    elif elapsed>=o['final_settle_budget_s']:fail('goal_not_stably_reached',monitor)
                elif phase=='protective_hold':protect=float(d.time)-self.first_violation['time']
                if on_step:
                    try:on_step(self,row)
                    except Exception as exc:
                        errors.append(repr(exc));on_step=None
                if self.success or (phase=='protective_hold' and protect>=o['protective_hold_s']-1e-9):break
                if self.steps>int(o['maximum_simulation_s']/self.dt)+int(o['protective_hold_s']/self.dt)+2:break
        except (Exception,KeyboardInterrupt) as exc:
            self.failure_reason=self.failure_reason or 'execution_exception';errors.append(repr(exc));self.success=False
        finally:
            for stream in streams.values():stream.close()
        return self._finish(started,errors,min_gap,min_bound,max_velocity,max_acceleration,native_count)

    def _finish(self,started,errors,min_gap,min_bound,max_velocity,max_acceleration,native_count):
        q=self.robot.positions();final_tcp=self.robot.tool_pose()[0];goal_tcp=self._tcp_ref(self.bundle.path[-1])
        err=np.array([x['error'] for x in self.tracking])
        result=dict(scene=self.bundle.evidence['scene'],seed=self.bundle.evidence['seed'],reference_speed_rad_s=self.speed,
            success=self.success,failure_reason=self.failure_reason,physics_steps=self.steps,logged_physics_steps=self.rows,
            simulation_time_s=float(self.data.time),wall_time_s=time.perf_counter()-started,
            trajectory_duration_s=self.trajectory.duration,minimum_actual_clearance_m=min_gap,minimum_is_lower_bound=min_bound,
            planning_minimum_clearance_m=self.bundle.evidence['planning_minimum_clearance_m'],safety_violation_steps=self.violations,
            prohibited_contact_observations=self.prohibited_contacts,native_contact_observations=native_count,
            command_rate_limit_steps=self.command_limited,actual_max_velocity_rad_s=max_velocity,actual_max_acceleration_fd_rad_s2=max_acceleration,
            final_q=q,goal_q=self.bundle.path[-1],final_joint_max_error_rad=float(np.max(np.abs(q-self.bundle.path[-1]))),
            final_tcp_error_m=float(np.linalg.norm(final_tcp-goal_tcp)),joint_rmse_rad=np.sqrt(np.mean(err*err,axis=0)) if len(err) else None,
            joint_max_abs_error_rad=np.max(np.abs(err),axis=0) if len(err) else None,
            tcp_rmse_m=float(np.sqrt(np.mean([r['tcp']**2 for r in self.tracking]))) if self.tracking else None,
            tcp_max_error_m=max([r['tcp'] for r in self.tracking],default=None),path_max_deviation_rad=max([r['path'] for r in self.tracking],default=None),
            actual_finger_max_deviation_from_planning_m=self.gripper_deviation,first_violation=self.first_violation,
            errors=errors,live_qpos_writes_after_initialization=0,controller='Unmodified upstream position servos; ctrl targets radians; original JointPositionController',
            run_directory=str(self.dest),source='actual live MjData / ctrl + mj_step')
        self.result=result;write_json(self.dest/'execution_summary.json',result);return result
