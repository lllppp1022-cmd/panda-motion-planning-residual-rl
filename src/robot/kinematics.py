import mujoco
import numpy as np
from src.robot.panda_interface import PandaInterface
class Kinematics:
    """All qpos writes are confined to this independent scratch MjData."""
    def __init__(self,model):
        self.model=model;self.data=mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model,self.data,mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_KEY,'home'))
        self.robot=PandaInterface(model,self.data);self.home=self.robot.positions();mujoco.mj_forward(model,self.data)
    def set_q(self,q):
        q=self.robot.validate(q);self.data.qpos[self.robot.qpos_indices]=q;self.data.qvel[:]=0;mujoco.mj_forward(self.model,self.data)
    def forward(self,q):self.set_q(q);return self.robot.tool_pose()
    def compute_position_jacobian(self,q):
        self.set_q(q);jp=np.zeros((3,self.model.nv));jr=np.zeros_like(jp)
        mujoco.mj_jacSite(self.model,self.data,jp,jr,self.robot.site_id)
        return jp[:,self.robot.dof_indices].copy()
    def finite_difference(self,q,epsilon=1e-6):
        if not np.isfinite(epsilon) or epsilon<=0:raise ValueError('Finite positive epsilon required')
        q=self.robot.validate(q);j=np.empty((3,7))
        for k in range(7):
            delta=np.zeros(7);delta[k]=epsilon
            j[:,k]=(self.forward(q+delta)[0]-self.forward(q-delta)[0])/(2*epsilon)
        self.set_q(q);return j
    def world_to_base(self,position):
        b=self.robot.base_body_id;R=self.data.xmat[b].reshape(3,3);return R.T@(np.asarray(position)-self.data.xpos[b])
    def conservative_reach_bound(self):
        # Triangle inequality along fixed body translations; valid for this fixed-base hinge chain.
        b=int(self.model.site_bodyid[self.robot.site_id]);bound=float(np.linalg.norm(self.model.site_pos[self.robot.site_id]))
        while b!=self.robot.base_body_id:
            bound+=float(np.linalg.norm(self.model.body_pos[b]));b=int(self.model.body_parentid[b])
        return bound
