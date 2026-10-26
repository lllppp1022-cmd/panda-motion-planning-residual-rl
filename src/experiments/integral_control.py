"""Independent limited error-integral baseline over unchanged nominal M3."""
import numpy as np
from src.rl.residual_controller import ResidualM3Controller

class IntegralM3Controller(ResidualM3Controller):
    def __init__(self,robot,nominal,perturbation,seed,hold_requested,ki=.3):
        super().__init__(robot,nominal,perturbation,seed,hold_requested,cap_rad=.002,residual_slew_rad_s=.01)
        self.ki=ki;self.integral=np.zeros(7);self.antiwindup_steps=0;self.integral_clamp_steps=0
    def step(self,target):
        ref=self.robot.validate(target);dt=float(self.robot.model.opt.timestep)
        if self.hold_requested():self.integral[:]=0.
        else:
            q=self.last['observed_q'] if self.last else self.robot.positions()
            change=self.ki*(ref-q)*dt;blocked=np.zeros(7,dtype=bool)
            if self.last:
                blocked=abs(self.last['delayed_target']-self.command_q)>.35*dt+1e-12
                force=self.robot.data.actuator_force[self.robot.actuator_ids]
                bounds=self.gravity.force_limits
                blocked|=(force<=bounds[:,0]+1e-8)|(force>=bounds[:,1]-1e-8)
            self.antiwindup_steps+=int(np.any(blocked));change[blocked]=0.
            raw=self.integral+change;self.integral=np.clip(raw,-.002,.002)
            self.integral_clamp_steps+=int(np.any(raw!=self.integral))
        self.set_residual(self.integral/.002);return super().step(ref)
