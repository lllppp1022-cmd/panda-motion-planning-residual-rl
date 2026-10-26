"""Existing MJCF position servos, slew-limited commands, actual mj_step dynamics."""
import numpy as np
import mujoco
from src.common import config
class JointPositionController:
    def __init__(self,robot,max_velocity=None):
        self.robot=robot;self.command_q=robot.positions();self.max_velocity=config()['max_command_velocity_rad_s'] if max_velocity is None else float(max_velocity)
        if not np.isfinite(self.max_velocity) or self.max_velocity<=0:raise ValueError('Positive finite command velocity required')
        self.physics_steps=0
    def step(self,target):
        desired=self.robot.validate(target);m=self.robot.model;d=self.robot.data;dt=float(m.opt.timestep)
        self.command_q+=np.clip(desired-self.command_q,-self.max_velocity*dt,self.max_velocity*dt)
        d.ctrl[self.robot.actuator_ids]=self.command_q
        # No qpos/qvel assignment or reset exists in the controller.
        mujoco.mj_step(m,d);self.physics_steps+=1
        # Refresh post-integration poses/contacts without another integration step.
        mujoco.mj_forward(m,d)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():raise FloatingPointError('Nonfinite dynamic state')
        x,_=self.robot.tool_pose()
        row=dict(time=float(d.time),physics_step=self.physics_steps,tool_x=x[0],tool_y=x[1],tool_z=x[2])
        for i,(q,v,u,goal) in enumerate(zip(self.robot.positions(),self.robot.velocities(),self.command_q,desired),1):row.update({f'q{i}':q,f'v{i}':v,f'command{i}':u,f'target{i}':goal})
        return row
