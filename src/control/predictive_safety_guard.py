"""Experimental shadow dynamics look-ahead; never steps or edits live data."""
import copy,time
import numpy as np
import mujoco
from src.robot.panda_interface import PandaInterface
from src.robot.joint_controller import JointPositionController
from src.control.compensated_joint_controller import CompensatedJointController


class PredictiveSafetyGuard:
    def __init__(self,executor,config):
        self.executor=executor;self.cfg=dict(config);self.model=copy.copy(executor.model);self.data=mujoco.MjData(self.model)
        self.robot=PandaInterface(self.model,self.data);self.dt=float(self.model.opt.timestep)
        self.controller=(CompensatedJointController(self.robot,max_velocity=config['command_slew_rad_s'],
            max_offset_rad=config['compensation_max_offset_rad']) if isinstance(executor.controller,CompensatedJointController)
            else JointPositionController(self.robot,max_velocity=config['command_slew_rad_s']))
        for key in ['guard_period_s','prediction_horizon_s','prediction_collision_period_s']:
            if config[key]<=0 or abs(round(config[key]/self.dt)*self.dt-config[key])>1e-12:raise ValueError('Guard times must be positive physics-step multiples')
        if config['execution_clearance_m']!=.005 or config['prediction_warning_margin_m']<.005:raise ValueError('5mm hard threshold cannot be lowered')
        self.last_time=-1.;self.last=None;self.first_warning_time=None;self.calls=0;self.elapsed_s=0.

    def predict(self):
        e=self.executor;started=time.perf_counter();m,d=self.model,self.data
        mujoco.mj_copyData(d,m,e.data)
        self.controller.command_q[:]=e.controller.command_q
        initial_time=float(d.time);n=int(round(self.cfg['prediction_horizon_s']/self.dt))
        stride=int(round(self.cfg['prediction_collision_period_s']/self.dt))
        minimum=e.last_clearance['minimum_clearance_m'] if e.last_clearance else .1
        samples=[];pair_at_min=None;bad_state=False
        for step in range(1,n+1):
            # Original execution advances quintic reference after its prehold.
            ref=e.trajectory.sample(max(0.,initial_time+step*self.dt-e.options['prehold_s']))
            self.controller.step(ref.q_ref)
            q=self.robot.positions()
            if not np.isfinite(q).all() or np.any(q<self.robot.limits[:,0]) or np.any(q>self.robot.limits[:,1]):bad_state=True;break
            if step%stride==0 or step==n:
                values=[float(mujoco.mj_geomDistance(m,d,p['id1'],p['id2'],.1,None)) for p in e.checker.pairs]
                if not np.isfinite(values).all():bad_state=True;break
                i=int(np.argmin(values));gap=values[i]
                if gap<minimum:minimum=gap;pair_at_min=e.checker.pairs[i]
                samples.append(dict(target_physics_step=e.steps+step,time=float(d.time),q=q.tolist(),minimum_gap_m=gap,
                                    closest_pair=e.checker.pairs[i]))
        warning=bad_state or minimum<=self.cfg['prediction_warning_margin_m']
        if warning and self.first_warning_time is None:self.first_warning_time=initial_time
        elapsed=time.perf_counter()-started;self.elapsed_s+=elapsed;self.calls+=1;self.last_time=initial_time
        self.last=dict(observed_physics_step=e.steps,observed_time=initial_time,horizon_s=self.cfg['prediction_horizon_s'],
            predicted_minimum_gap_m=minimum,warning=bool(warning),invalid_prediction=bad_state,closest_pair=pair_at_min,
            computation_s=elapsed,samples=samples,live_physics_steps_added=0,
            source='mj_copyData to independent shadow; same controller; continued original reference; discrete predicted geometry')
        return self.last

    def observe(self):
        if self.executor.first_violation:return None
        if self.executor.data.time-self.last_time>=self.cfg['guard_period_s']-1e-9:return self.predict()
        return None
