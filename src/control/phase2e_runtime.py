"""Independent M0/M1/M2 wiring and extra telemetry; original executor untouched."""
import json,time
from pathlib import Path
import numpy as np
import yaml
from src.common import ROOT,write_json,sha
from src.control.trajectory_executor import TrajectoryExecutor,CSVStream
from src.control.compensated_joint_controller import CompensatedJointController
from src.control.predictive_safety_guard import PredictiveSafetyGuard


def config():return yaml.safe_load((ROOT/'configs/phase2e_controllers.yaml').read_text(encoding='utf-8'))


class ControllerRun:
    def __init__(self,bundle,method,speed,dest,guard_mode=None):
        if method not in ('M0','M1','M2'):raise ValueError('Expected M0/M1/M2')
        self.cfg=config();self.method=method;self.dest=Path(dest);self.e=TrajectoryExecutor(bundle,speed,self.dest)
        self.guard_mode=guard_mode or ('active' if method=='M2' else 'off')
        if self.guard_mode not in ('off','shadow','active'):raise ValueError('Bad guard mode')
        if method=='M0' and self.guard_mode=='active':raise ValueError('M0 must remain original')
        if method!='M0':self.e.controller=CompensatedJointController(self.e.robot,self.cfg['command_slew_rad_s'],self.cfg['compensation_max_offset_rad'])
        self.guard=PredictiveSafetyGuard(self.e,self.cfg) if self.guard_mode!='off' else None
        self.predictions=[];self.telemetry_rows=[];self.previous_ctrl=self.e.controller.command_q.copy();self.previous_rate=np.zeros(7)
        self.first_hard_violation=None;self.first_contact=None;self.stop_q=None;self.last_contact=False;self.contact_intervals=0
        self.guard_fault=None

    def run(self,display=None,external_stop=None):
        e=self.e;m=e.model;ids=e.robot.actuator_ids;kp=m.actuator_gainprm[ids,0];kd=-m.actuator_biasprm[ids,2];limits=m.actuator_forcerange[ids]
        signals=CSVStream(self.dest/'control_signals.csv')
        write_json(self.dest/'phase2e_config.json',dict(method=self.method,guard_mode=self.guard_mode,config=self.cfg,
             model_actuators=e.robot.inspection(),gravity_audit=e.controller.gravity.audit() if self.method!='M0' else None,
             source_sha256={p:sha(ROOT/p) for p in ['src/control/gravity_compensation.py','src/control/compensated_joint_controller.py',
             'src/control/predictive_safety_guard.py','src/control/phase2e_runtime.py','configs/phase2e_controllers.yaml']}))
        def observe(ex,row):
            ctrl=e.data.ctrl[ids].copy();q=e.robot.positions();vel=e.robot.velocities();raw=kp*(ctrl-q)-kd*vel;tau=e.data.actuator_force[ids].copy()
            rate=(ctrl-self.previous_ctrl)/e.dt;acc=(rate-self.previous_rate)/e.dt;self.previous_ctrl=ctrl;self.previous_rate=rate
            info=e.controller.last if self.method!='M0' else {}
            out=dict(time=row['time'],physics_step=row['physics_step'],phase=row['phase'],method=self.method,
                algorithm_s=info.get('algorithm_s',0.),compensation_clipped=info.get('compensation_clipped',False),
                position_clipped=info.get('position_clipped',False),effective_rate_limited=info.get('effective_rate_limited',bool(np.max(abs(rate))>=.35-1e-10)))
            for j in range(7):
                for name,value in [('tau_raw',raw[j]),('tau_actual',tau[j]),('ctrl_rate',rate[j]),('ctrl_acceleration',acc[j]),
                    ('gravity_nm',info.get('gravity_nm',np.zeros(7))[j]),('gravity_offset_rad',info.get('offset_rad',np.zeros(7))[j]),
                    ('torque_saturated',bool(raw[j]<=limits[j,0]+1e-9 or raw[j]>=limits[j,1]-1e-9))]:out[f'{name}_{j+1}']=float(value) if not isinstance(value,bool) else value
            signals.append(out);self.telemetry_rows.append(out)
            if not row['safety_valid'] and self.first_hard_violation is None:self.first_hard_violation=row['time']
            contact=row['prohibited_contact_count']>0
            if contact and self.first_contact is None:self.first_contact=row['time']
            if contact and not self.last_contact:self.contact_intervals+=1
            self.last_contact=contact
            if e.first_violation and self.stop_q is None:self.stop_q=np.array(e.first_violation['q'])
            if self.guard:
                try:
                    prediction=self.guard.observe()
                    if prediction:self.predictions.append(prediction)
                except Exception as exc:
                    self.guard_fault=repr(exc)
                    self.predictions.append(dict(error=self.guard_fault,observed_time=row['time']))
            if display:display(self,row)
        def stop():
            if external_stop:
                reason=external_stop()
                if reason:return reason
            if self.guard_mode=='active' and self.guard_fault and not e.first_violation:return 'predictive_guard_error'
            if self.guard_mode=='active' and self.guard and self.guard.last and self.guard.last['warning'] and not e.first_violation:
                return 'predictive_guard_stop'
            return None
        try:r=e.run(on_step=observe,stop_requested=stop)
        finally:signals.close();write_json(self.dest/'predictions.json',self.predictions)
        for key in ['errors']:
            if r[key]:r['telemetry_or_display_errors']=r[key]
        rates=np.array([[x[f'ctrl_rate_{j}'] for j in range(1,8)] for x in self.telemetry_rows])
        torques=np.array([[x[f'tau_actual_{j}'] for j in range(1,8)] for x in self.telemetry_rows])
        saturation=sum(any(x[f'torque_saturated_{j}'] for j in range(1,8)) for x in self.telemetry_rows)
        active_stop=r['failure_reason']=='predictive_guard_stop';aborted=r['failure_reason'] in ('viewer_closed','user_abort')
        r.update(method=self.method,guard_mode=self.guard_mode,controller_variant={'M0':'original_position_servo','M1':'gravity_compensated_position_servo','M2':'gravity_compensated_servo_predictive_hold'}[self.method],
            guard_fault=self.guard_fault,
            outcome='success' if r['success'] else 'user_aborted' if aborted else 'protected_stop' if active_stop else 'failure',
            execution_clearance_m=.005,first_warning_time_s=None if not self.guard else self.guard.first_warning_time,
            first_hard_violation_time_s=self.first_hard_violation,first_contact_time_s=self.first_contact,
            guard_action_time_s=r['first_violation']['time'] if active_stop else None,
            post_stop_joint_displacement_rad=None if self.stop_q is None else float(np.linalg.norm(e.robot.positions()-self.stop_q)),
            safely_stopped_without_goal=bool(active_stop and r['safety_violation_steps']==0 and self.contact_intervals==0 and max(abs(e.robot.velocities()))<=.02),
            contact_intervals=self.contact_intervals,torque_saturation_steps=saturation,
            peak_actual_torques_nm=np.max(abs(torques),axis=0) if len(torques) else None,
            max_ctrl_rate_rad_s=float(np.max(abs(rates))) if len(rates) else None,
            control_total_variation_rad=(np.sum(abs(rates),axis=0)*e.dt) if len(rates) else None,
            max_ctrl_acceleration_rad_s2=max((abs(x[f'ctrl_acceleration_{j}']) for x in self.telemetry_rows for j in range(1,8)),default=None),
            gravity_algorithm_s=getattr(e.controller,'algorithm_s',0.),guard_algorithm_s=0. if not self.guard else self.guard.elapsed_s,
            guard_calls=0 if not self.guard else self.guard.calls,
            effective_rate_limit_steps=sum(x['effective_rate_limited'] for x in self.telemetry_rows),
            legacy_ref_rate_limit_counter_note='Original counter compares q_ref to command; use effective_rate_limit_steps for compensated control',
            compensation_clipped_steps=sum(x['compensation_clipped'] for x in self.telemetry_rows),
            position_clipped_steps=sum(x['position_clipped'] for x in self.telemetry_rows))
        write_json(self.dest/'phase2e_summary.json',r);return r
