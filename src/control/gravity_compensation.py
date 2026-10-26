"""Pure-gravity query on independent model/data, with an audited servo map."""
import copy,time
import mujoco
import numpy as np
from src.robot.panda_interface import PandaInterface


class GravityCompensation:
    def __init__(self,model):
        self.model=copy.copy(model);m=self.model;self.data=mujoco.MjData(m);self.robot=PandaInterface(m,self.data)
        ids=self.robot.actuator_ids
        if np.any(m.body_gravcomp!=0) or np.any(m.jnt_stiffness!=0):
            raise ValueError('Existing gravity compensation or springs require a new audit; refusing duplicate compensation')
        if m.opt.density!=0 or m.opt.viscosity!=0:raise ValueError('Fluid forces require separate audit')
        if np.any(m.actuator_dyntype[ids]!=mujoco.mjtDyn.mjDYN_NONE) or np.any(m.actuator_gaintype[ids]!=mujoco.mjtGain.mjGAIN_FIXED):
            raise ValueError('Unsupported actuator dynamics/gain mapping')
        if np.any(m.actuator_biastype[ids]!=mujoco.mjtBias.mjBIAS_AFFINE):raise ValueError('Affine position servo required')
        self.kp=m.actuator_gainprm[ids,0].copy();self.kd=-m.actuator_biasprm[ids,2].copy()
        if not np.allclose(m.actuator_gainprm[ids,1:],0) or not np.allclose(m.actuator_biasprm[ids,0],0):raise ValueError('Unsupported affine constants')
        if not np.all(m.actuator_forcelimited[ids]) or not np.all(m.actuator_ctrllimited[ids]):raise ValueError('Native force/control limits required')
        if not np.allclose(m.actuator_gear[ids],np.tile([1,0,0,0,0,0],(7,1))):raise ValueError('Unit direct-joint gear required')
        if not np.isfinite(m.opt.gravity).all():raise ValueError('Nonfinite gravity')
        self.force_limits=m.actuator_forcerange[ids].copy();self.calls=0;self.elapsed_s=0.

    def query(self,q,full_qpos=None):
        began=time.perf_counter();q=self.robot.validate(q);m,d=self.model,self.data
        mujoco.mj_resetDataKeyframe(m,d,mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_KEY,'home'))
        if full_qpos is not None:
            full_qpos=np.asarray(full_qpos,dtype=float)
            if full_qpos.shape!=(m.nq,) or not np.isfinite(full_qpos).all():raise ValueError('Finite full model qpos required')
            d.qpos[:]=full_qpos
        d.qpos[self.robot.qpos_indices]=q;d.qvel[:]=0.;d.qacc[:]=0.
        mujoco.mj_forward(m,d)
        if np.max(np.abs(d.qfrc_passive[self.robot.dof_indices]))>1e-8:
            raise ValueError('Unaudited passive force at rest; refusing to label bias alone as required gravity torque')
        gravity=d.qfrc_bias[self.robot.dof_indices].copy()
        if not np.isfinite(gravity).all():raise ValueError('Nonfinite gravity force')
        self.calls+=1;self.elapsed_s+=time.perf_counter()-began
        return gravity

    def audit(self):
        return dict(kp=self.kp,kd=self.kd,gear=self.model.actuator_gear[self.robot.actuator_ids],force_limits=self.force_limits,
            dof_addresses=self.robot.dof_indices,qpos_addresses=self.robot.qpos_indices,
            body_gravcomp=self.model.body_gravcomp,joint_stiffness=self.model.jnt_stiffness,
            mapping='tau=clip(Kp*(ctrl-q)-Kd*qvel, original force limits); ctrl is joint radians',
            gravity_query='independent MjData; explicitly zero velocity; no springs/body gravcomp/fluid forces')
