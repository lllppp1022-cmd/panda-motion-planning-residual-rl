"""Reuse Phase3D-A RGB/state primitives, labelled deterministic BC/control comparison."""
import hashlib,threading
import numpy as np
import mujoco
import imageio.v2 as imageio
from src.training_observer.observer import annotate,numbers,state_vector,STATE
from src.common import write_json


class Recording:
    def __init__(self,e,dest,method,condition):
        self.e=e;self.dest=dest;self.method=method;self.condition=condition;self.owner=threading.get_ident();self.frames=[];self.states=[];self.qpos=[];self.qvel=[];self.ctrl=[];self.time=[];self.step=[]
        self.renderer=mujoco.Renderer(e.model,height=544,width=960);self.camera=mujoco.MjvCamera();self.camera.lookat[:]=[.2,0,.4];self.camera.distance=1.85;self.camera.azimuth=135;self.camera.elevation=-25
        self.writer=imageio.get_writer(str(dest/'bc_live.mp4'),fps=10,codec='libx264',macro_block_size=16);self.next_time=0.
    def capture(self,row,latest_action):
        assert threading.get_ident()==self.owner;e=self.e;c=e.controller;before=state_vector(e.model,e.data)
        self.states.append(before);self.qpos.append(e.data.qpos.copy());self.qvel.append(e.data.qvel.copy());self.ctrl.append(e.data.ctrl.copy());self.time.append(float(e.data.time));self.step.append(e.steps)
        if row['time']<self.next_time-1e-9 and not e.success:return
        self.next_time=float(row['time'])+.1;self.renderer.update_scene(e.data,camera=self.camera);rgb=self.renderer.render().copy()
        assert np.array_equal(before,state_vector(e.model,e.data))
        last=c.last;qr=[row[f'q_ref_{j}'] for j in range(1,8)];qa=e.robot.positions().tolist();ctrl=e.data.ctrl[e.robot.actuator_ids].copy()
        lines=[f'BASELINE COMPARISON | ACTUAL DYNAMIC SERVO EXECUTION | {self.method}',
            f'{e.bundle.evidence["scene"]} | {self.condition["name"]} | sim={e.data.time:.3f}s | {row["phase"]}',
            'q_ref: '+numbers(qr),'q_actual: '+numbers(qa),'own observable integral rad: '+numbers(c.memory.value),
            'executed target residual rad: '+numbers(last['residual_target_rad']),'actual applied residual rad: '+numbers(last['residual_rad']),
            'actual ctrl (position targets): '+numbers(ctrl),'latest20Hz interface request: '+numbers(latest_action),
            f'Actual TCP error={row["tcp_error_m"]*1000:.5f}mm | gap={row["minimum_clearance_m"]*1000:.5f}mm | hard threshold5mm',
            'Original +/-0.002rad residual |0.01rad/s residual slew |0.35rad/s total target slew | native force limits',
            'LIVE integrating MjData RGB; no kinematic path playback; no PPO updates',
            'SUCCESS' if e.success else e.failure_reason or 'CONTROLLING']
        image=annotate(rgb,lines);self.writer.append_data(image)
        self.frames.append(dict(frame=len(self.frames),time=float(e.data.time),physics_step=e.steps,q_actual=qa,q_ref=qr,ctrl=ctrl.tolist(),
            observable_integral=c.memory.value.tolist(),target_residual=last['residual_target_rad'].tolist(),applied_residual=last['residual_rad'].tolist(),
            tcp_error_m=row['tcp_error_m'],minimum_clearance_m=row['minimum_clearance_m'],raw_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),live_data_id=id(e.data)))
    def close(self):
        self.writer.close();self.renderer.close();write_json(self.dest/'bc_video_frames.json',self.frames)
        np.savez_compressed(self.dest/'integration_states.npz',states=np.asarray(self.states),qpos=np.asarray(self.qpos),qvel=np.asarray(self.qvel),ctrl=np.asarray(self.ctrl),
            time=np.asarray(self.time),physics_step=np.asarray(self.step),state_spec=int(STATE))
