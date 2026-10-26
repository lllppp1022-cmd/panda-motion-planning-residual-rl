"""Limited gravity feedforward through original position-servo inputs only."""
import time
import numpy as np
from src.robot.joint_controller import JointPositionController
from src.control.gravity_compensation import GravityCompensation


class CompensatedJointController(JointPositionController):
    def __init__(self,robot,max_velocity=.35,max_offset_rad=.02,enabled=True):
        super().__init__(robot,max_velocity);self.gravity=GravityCompensation(robot.model)
        if not np.isfinite(max_offset_rad) or max_offset_rad<=0:raise ValueError('Positive finite compensation cap required')
        self.max_offset_rad=float(max_offset_rad);self.enabled=bool(enabled)
        self.last={};self.rate_limited_steps=0;self.algorithm_s=0.

    def step(self,target):
        started=time.perf_counter();ref=self.robot.validate(target)
        tau=self.gravity.query(self.robot.positions(),self.robot.data.qpos) if self.enabled else np.zeros(7)
        raw=tau/self.gravity.kp;offset=np.clip(raw,-self.max_offset_rad,self.max_offset_rad)
        desired=np.clip(ref+offset,self.robot.limits[:,0],self.robot.limits[:,1])
        limited=bool(np.any(np.abs(desired-self.command_q)>self.max_velocity*self.robot.model.opt.timestep+1e-12))
        self.rate_limited_steps+=int(limited);elapsed=time.perf_counter()-started;self.algorithm_s+=elapsed
        self.last=dict(gravity_nm=tau,raw_offset_rad=raw,offset_rad=offset,corrected_target_rad=desired,
            compensation_clipped=bool(np.any(raw!=offset)),position_clipped=bool(np.any(desired!=ref+offset)),
            effective_rate_limited=limited,algorithm_s=elapsed,enabled=self.enabled)
        # Original implementation still performs ctrl write, rate limit, mj_step,
        # and post-integration mj_forward. No live qpos/qvel mutation here.
        return super().step(desired)
