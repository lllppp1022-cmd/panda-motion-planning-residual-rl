"""Audited M3 order plus post-feedforward bounded residual, original servo integration."""
from collections import deque
import time
import numpy as np
from src.robot.joint_controller import JointPositionController
from src.control.gravity_compensation import GravityCompensation

class ResidualM3Controller(JointPositionController):
    def __init__(self,robot,nominal_model,perturbation,seed,hold_requested,cap_rad=.002,before_step=None,residual_slew_rad_s=.01):
        super().__init__(robot,max_velocity=.35)
        if not np.isfinite(cap_rad) or not 0<=cap_rad<=.002:raise ValueError('Unvalidated residual cap')
        self.gravity=GravityCompensation(nominal_model);self.perturbation=dict(perturbation);self.rng=np.random.default_rng(seed)
        self.obs_delay=int(perturbation.get('observation_delay_steps',0));self.action_delay=int(perturbation.get('action_delay_steps',0))
        self.observations=deque([robot.data.qpos.copy() for _ in range(self.obs_delay+1)],maxlen=self.obs_delay+1)
        self.velocities=deque([robot.velocities() for _ in range(self.obs_delay+1)],maxlen=self.obs_delay+1)
        self.actions=deque([robot.positions() for _ in range(self.action_delay+1)],maxlen=self.action_delay+1)
        self.previous_reference=None;self.hold_requested=hold_requested;self.before_step=before_step
        self.cap_rad=cap_rad;self.residual=np.zeros(7);self.last={};self.algorithm_s=0.
        if not np.isfinite(residual_slew_rad_s) or residual_slew_rad_s<=0:raise ValueError('Positive residual slew required')
        self.residual_slew_rad_s=residual_slew_rad_s;self.applied_residual=np.zeros(7)
    def set_residual(self,normalized):
        x=np.asarray(normalized,dtype=float)
        if x.shape!=(7,) or not np.isfinite(x).all():raise ValueError('Seven finite normalized residuals required')
        self.residual=np.clip(x,-1,1)*self.cap_rad
    def step(self,target):
        if self.before_step:self.before_step()
        began=time.perf_counter();ref=self.robot.validate(target);dt=float(self.robot.model.opt.timestep)
        observation=self.robot.data.qpos.copy();noise_std=self.perturbation.get('q_noise_std_rad',0.)
        noise=self.rng.normal(0,noise_std,7) if noise_std else np.zeros(7)
        observation[self.robot.qpos_indices]=np.clip(observation[self.robot.qpos_indices]+noise,self.robot.limits[:,0],self.robot.limits[:,1])
        self.observations.append(observation);observed=self.observations[0].copy()
        self.velocities.append(self.robot.velocities());observed_velocity=self.velocities[0].copy()
        holding=self.hold_requested()
        dq=np.zeros(7) if self.previous_reference is None or holding else (ref-self.previous_reference)/dt
        self.previous_reference=ref.copy();g=self.gravity.query(observed[self.robot.qpos_indices],observed)
        gravity_offset=np.clip(g/self.gravity.kp,-.02,.02);raw_ff=self.gravity.kd/self.gravity.kp*dq;ff=np.clip(raw_ff,-.02,.02)
        raw=ref+gravity_offset+ff
        base=np.clip(raw,self.robot.limits[:,0],self.robot.limits[:,1])
        if holding:self.applied_residual[:]=0.
        else:self.applied_residual+=np.clip(self.residual-self.applied_residual,-self.residual_slew_rad_s*dt,self.residual_slew_rad_s*dt)
        residual=self.applied_residual.copy()
        desired=np.clip(base+residual,self.robot.limits[:,0],self.robot.limits[:,1])
        self.actions.append(desired.copy());delayed=self.actions[0].copy()
        limited=np.any(np.abs(delayed-self.command_q)>.35*dt+1e-12)
        cost=time.perf_counter()-began;self.algorithm_s+=cost
        self.last=dict(observed_q=observed[self.robot.qpos_indices],observed_velocity=observed_velocity,noise=noise,
            reference_velocity=dq,gravity_nm=g,base_target=base,residual_rad=residual,residual_target_rad=self.residual.copy(),desired_before_delay=desired,
            delayed_target=delayed,rate_limited=bool(limited),algorithm_s=cost)
        return super().step(delayed)
