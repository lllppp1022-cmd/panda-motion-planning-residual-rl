"""M1/M3 using nominal parameters only and explicitly delayed observations/actions."""
from collections import deque
import time
import numpy as np
from src.robot.joint_controller import JointPositionController
from src.control.gravity_compensation import GravityCompensation

class NominalFeedforwardController(JointPositionController):
    def __init__(self,robot,nominal_model,method='M1',perturbation=None,seed=3101,hold_requested=None):
        super().__init__(robot,max_velocity=.35)
        if method not in ('M1','M3'):raise ValueError('M1/M3 only')
        self.gravity=GravityCompensation(nominal_model) # never reads physical parameter arrays
        self.method=method;self.perturbation=dict(perturbation or {});self.rng=np.random.default_rng(seed)
        self.obs_delay=int(self.perturbation.get('observation_delay_steps',0));self.action_delay=int(self.perturbation.get('action_delay_steps',0))
        self.observations=deque([robot.data.qpos.copy() for _ in range(self.obs_delay+1)],maxlen=self.obs_delay+1)
        self.actions=deque([robot.positions() for _ in range(self.action_delay+1)],maxlen=self.action_delay+1)
        self.previous_reference=None;self.hold_requested=hold_requested or (lambda:False);self.last={};self.algorithm_s=0.
    def step(self,target):
        began=time.perf_counter();ref=self.robot.validate(target);dt=float(self.robot.model.opt.timestep)
        observation=self.robot.data.qpos.copy()
        noise_std=self.perturbation.get('q_noise_std_rad',0.)
        noise=self.rng.normal(0,noise_std,7) if noise_std else np.zeros(7)
        observation[self.robot.qpos_indices]=np.clip(observation[self.robot.qpos_indices]+noise,self.robot.limits[:,0],self.robot.limits[:,1])
        self.observations.append(observation);observed=self.observations[0].copy()
        dq=np.zeros(7) if self.previous_reference is None or self.hold_requested() else (ref-self.previous_reference)/dt
        self.previous_reference=ref.copy()
        g=self.gravity.query(observed[self.robot.qpos_indices],observed)
        gravity_offset=np.clip(g/self.gravity.kp,-.02,.02)
        raw_ff=self.gravity.kd/self.gravity.kp*dq if self.method=='M3' else np.zeros(7)
        ff=np.clip(raw_ff,-.02,.02)
        raw=ref+gravity_offset+ff;desired=np.clip(raw,self.robot.limits[:,0],self.robot.limits[:,1])
        self.actions.append(desired.copy());delayed=self.actions[0].copy()
        limited=np.any(np.abs(delayed-self.command_q)>.35*dt+1e-12)
        cost=time.perf_counter()-began;self.algorithm_s+=cost
        self.last=dict(observed_q=observed[self.robot.qpos_indices],noise=noise,reference_velocity=dq,gravity_nm=g,
            gravity_offset=gravity_offset,velocity_offset=ff,unlimited_velocity_offset=raw_ff,desired_before_delay=desired,
            delayed_target=delayed,rate_limited=bool(limited),ff_clipped=bool(np.any(ff!=raw_ff)),
            position_clipped=bool(np.any(desired!=raw)),algorithm_s=cost)
        return super().step(delayed)
