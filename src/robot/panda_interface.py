"""Discover actual model addresses and actuator transmissions, not assumed nq/nu."""
import mujoco
import numpy as np
from src.common import config
class PandaInterface:
    def __init__(self,model,data):
        self.model=model;self.data=data;self.cfg=config();self.names=self.cfg['arm_joint_names']
        self.joint_ids=np.array([mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,n) for n in self.names]);assert np.all(self.joint_ids>=0)
        assert np.all(model.jnt_type[self.joint_ids]==mujoco.mjtJoint.mjJNT_HINGE)
        self.qpos_indices=model.jnt_qposadr[self.joint_ids].copy();self.dof_indices=model.jnt_dofadr[self.joint_ids].copy()
        self.actuator_ids=[]
        for j in self.joint_ids:
            a=np.flatnonzero((model.actuator_trntype==mujoco.mjtTrn.mjTRN_JOINT)&(model.actuator_trnid[:,0]==j));assert len(a)==1
            self.actuator_ids.append(int(a[0]))
        self.actuator_ids=np.array(self.actuator_ids);self.limits=model.jnt_range[self.joint_ids].copy()
        assert np.allclose(model.actuator_ctrlrange[self.actuator_ids],self.limits)
        assert np.allclose(model.actuator_gear[self.actuator_ids,0],1)
        kp=model.actuator_gainprm[self.actuator_ids,0];bias=model.actuator_biasprm[self.actuator_ids]
        assert np.all(kp>0) and np.allclose(bias[:,1],-kp) and np.all(bias[:,2]<0),'Expected upstream affine position servo'
        self.site_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,self.cfg['tool_site']);assert self.site_id>=0
        self.base_body_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,self.cfg['base_body'])
    def positions(self):return self.data.qpos[self.qpos_indices].copy()
    def velocities(self):return self.data.qvel[self.dof_indices].copy()
    def validate(self,q):
        q=np.asarray(q,dtype=float)
        if q.shape!=(7,) or not np.isfinite(q).all():raise ValueError('Expected seven finite joint angles in radians')
        if np.any(q<self.limits[:,0]) or np.any(q>self.limits[:,1]):raise ValueError('Joint target outside actual model limits')
        return q.copy()
    def tool_pose(self):return self.data.site_xpos[self.site_id].copy(),self.data.site_xmat[self.site_id].reshape(3,3).copy()
    def inspection(self):
        m=self.model;arm=[]
        for n,j,a,p,v,lim in zip(self.names,self.joint_ids,self.actuator_ids,self.qpos_indices,self.dof_indices,self.limits):
            arm.append(dict(name=n,joint_id=int(j),qpos_address=int(p),dof_address=int(v),range_rad=lim.tolist(),actuator=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,int(a)),actuator_id=int(a),ctrlrange=m.actuator_ctrlrange[a].tolist(),forcerange=m.actuator_forcerange[a].tolist(),gain=m.actuator_gainprm[a,:3].tolist(),bias=m.actuator_biasprm[a,:3].tolist(),gear=m.actuator_gear[a].tolist(),control_meaning='desired joint position in radians; not joint torque'))
        allj=[]
        for j in range(m.njnt):allj.append(dict(name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j),id=j,type=int(m.jnt_type[j]),qpos_address=int(m.jnt_qposadr[j]),dof_address=int(m.jnt_dofadr[j]),range=m.jnt_range[j].tolist()))
        alla=[]
        for a in range(m.nu):alla.append(dict(id=a,name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,a),transmission_type=int(m.actuator_trntype[a]),transmission_ids=m.actuator_trnid[a].tolist(),ctrlrange=m.actuator_ctrlrange[a].tolist(),gain=m.actuator_gainprm[a,:3].tolist(),bias=m.actuator_biasprm[a,:3].tolist()))
        return dict(nq=m.nq,nv=m.nv,nu=m.nu,njnt=m.njnt,nbody=m.nbody,ngeom=m.ngeom,neq=m.neq,ntendon=m.ntendon,timestep=m.opt.timestep,integrator=int(m.opt.integrator),arm_joints=arm,all_joints=allj,all_actuators=alla,tool_site_id=self.site_id,tool_body=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.site_bodyid[self.site_id])),tool_local_position=m.site_pos[self.site_id].tolist(),tool_local_quaternion=m.site_quat[self.site_id].tolist(),tool_world_position=self.tool_pose()[0].tolist(),base_world_position=self.data.xpos[self.base_body_id].tolist(),gripper='Two slide joints coupled by equality; actuator8 drives split tendon, ctrl0..255 maps to average finger opening0..0.04m')
