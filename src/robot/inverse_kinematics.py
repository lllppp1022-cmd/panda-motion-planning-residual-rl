from dataclasses import dataclass
import numpy as np
from src.common import config
from src.robot.kinematics import Kinematics
@dataclass
class IKResult:
    status:str
    success:bool
    q:np.ndarray
    final_error_m:float|None
    iterations:int
    errors_m:list
    reason:str
class DampedLeastSquaresIK:
    def __init__(self,model):self.kin=Kinematics(model);self.options=config()['ik']
    def solve(self,target,q_initial,max_iterations=None):
        q=self.kin.robot.validate(q_initial);target=np.asarray(target,dtype=float)
        initial=self.kin.forward(q)[0]
        if target.shape!=(3,) or not np.isfinite(target).all():return IKResult('invalid_target',False,q,None,0,[],'Expected three finite world-frame position coordinates; error not evaluated')
        error=float(np.linalg.norm(target-initial));base=self.kin.data.xpos[self.kin.robot.base_body_id]
        if np.linalg.norm(target-base)>self.kin.conservative_reach_bound():return IKResult('unreachable_bound',False,q,error,0,[error],'Target beyond conservative chain-length bound')
        options=self.options;max_iterations=options['max_iterations'] if max_iterations is None else max_iterations
        if max_iterations<1:raise ValueError('max_iterations must be positive')
        margin=options['joint_margin_rad'];bounds=self.kin.robot.limits;errors=[];status='max_iterations'
        for iteration in range(max_iterations):
            x=self.kin.forward(q)[0];e=target-x;error=float(np.linalg.norm(e));errors.append(error)
            if error<options['position_tolerance_m']:return IKResult('converged',True,q,error,iteration,errors,'Position-only tolerance reached; no collision/pose guarantee')
            j=self.kin.compute_position_jacobian(q)
            try:dq=j.T@np.linalg.solve(j@j.T+options['damping_m_per_rad']**2*np.eye(3),e)
            except np.linalg.LinAlgError:status='numerical_failure';break
            if not np.isfinite(dq).all():status='numerical_failure';break
            dq*=min(1.,options['max_joint_step_rad']/max(float(np.max(np.abs(dq))),1e-15))
            accepted=False
            for scale in [1.,.5,.25,.125,.0625]:
                candidate=np.clip(q+scale*dq,bounds[:,0]+margin,bounds[:,1]-margin)
                if np.linalg.norm(target-self.kin.forward(candidate)[0])<error-1e-12:q=candidate;accepted=True;break
            if not accepted:status='stalled';break
        error=float(np.linalg.norm(target-self.kin.forward(q)[0]));errors.append(error)
        return IKResult(status,False,q,error,len(errors)-1,errors,'No convergence; not automatically classified unreachable')
