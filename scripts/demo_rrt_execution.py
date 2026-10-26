"""Native DYNAMIC SERVO EXECUTION; video renders the integrating live MjData."""
from _bootstrap import destination,main_guard
import argparse,hashlib,json,queue,shutil,time
from contextlib import nullcontext
import imageio.v2 as imageio
import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from src.common import ROOT,write_json,sha
from src.control.validated_path import load_verified_path
from src.control.trajectory_executor import TrajectoryExecutor
from src.robot.teleop import KeyboardQueue
from src.visualization.windows_close import QueuedWindowClose


class DynamicRecording:
    def __init__(self,e,dest,fps=10):
        self.e=e;self.dest=dest;self.fps=fps;self.frames=[];self.path=dest/'phase2c_dynamic_execution.mp4'
        self.renderer=mujoco.Renderer(e.model,height=640,width=960)
        self.camera=mujoco.MjvCamera();self.camera.lookat[:]=[.2,0,.4];self.camera.distance=1.9;self.camera.azimuth=135;self.camera.elevation=-25
        self.writer=imageio.get_writer(str(self.path),fps=fps,codec='libx264',macro_block_size=16)
        self.first=e.robot.positions()
    def capture(self,e,row):
        if self.frames and row['time']-self.frames[-1]['time']<1/self.fps-1e-9 and not e.success:return
        self.renderer.update_scene(e.data,camera=self.camera);raw=self.renderer.render().copy()
        im=Image.new('RGB',(960,768),'#16212d');im.paste(Image.fromarray(raw),(0,128));pen=ImageDraw.Draw(im)
        try:font=ImageFont.truetype('arial.ttf',17)
        except OSError:font=ImageFont.load_default()
        actual=[row[f'tcp_actual_{i}'] for i in range(3)];ref=[row[f'tcp_ref_{i}'] for i in range(3)]
        status='SUCCESS / stable goal' if e.success else 'FAILED / '+e.failure_reason if e.failure_reason else row['phase']
        gap=row['minimum_clearance_m'];gaptext='unknown' if gap is None else f'{gap*1000:.3f} mm'
        lines=['DYNAMIC SERVO EXECUTION | ACTUAL LIVE MuJoCo ctrl + mj_step',
               f"{e.bundle.evidence['scene']} | t={row['time']:.3f}s | segment={row['segment']} | progress={row['progress']*100:.1f}% | {status}",
               f'TCP reference m: {np.round(ref,4)} | actual m: {np.round(actual,4)}',
               f"Actual minimum gap: {gaptext} | TCP tracking error: {row['tcp_error_m']*1000:.3f} mm",
               f"Physics step {row['physics_step']} | q_path / q_ref / ctrl / q_actual logged separately"]
        for i,line in enumerate(lines):pen.text((10,5+i*24),line,font=font,fill='white' if i!=3 else '#6de8bc')
        self.writer.append_data(np.asarray(im))
        self.frames.append(dict(frame=len(self.frames),time=row['time'],physics_step=row['physics_step'],
            q_actual=e.robot.positions().tolist(),q_ref=[row[f'q_ref_{i+1}'] for i in range(7)],
            ctrl=e.data.ctrl[e.robot.actuator_ids].tolist(),raw_rgb_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
            state=status,live_data_id=id(e.data),source='same integrating live MjData; no scratch pose playback'))
        if len(self.frames)==1 or e.success:im.save(self.dest/('first.png' if len(self.frames)==1 else 'final.png'))
    def close(self):
        self.writer.close();self.renderer.close();n=0;changed=set()
        with imageio.get_reader(str(self.path)) as r:
            for rgb in r:n+=1;changed.add(hashlib.sha256(rgb.tobytes()).hexdigest())
            shape=r.get_data(0).shape if n else None
        q=np.array([r['q_actual'] for r in self.frames])
        result=dict(source='DYNAMIC SERVO EXECUTION / same live MjData used by mj_step',frames=len(self.frames),decoded_frames=n,
            duration_s=n/self.fps,fps=self.fps,resolution=list(shape[:2][::-1]) if shape else None,
            raw_unique_frames=len({f['raw_rgb_sha256'] for f in self.frames}),decoded_unique_frames=len(changed),
            maximum_joint_displacement_rad=float(np.max(np.linalg.norm(q-self.first,axis=1))) if len(q) else 0.,
            all_frames_same_live_data=all(f['live_data_id']==id(self.e.data) for f in self.frames),
            frame_times_monotonic=all(b['time']>a['time'] for a,b in zip(self.frames,self.frames[1:])),
            sha256=sha(self.path),file_bytes=self.path.stat().st_size,final_success=self.e.success)
        assert n==len(self.frames)
        write_json(self.dest/'video_frame_log.json',self.frames);write_json(self.dest/'video_validation.json',result)
        return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--viewer',action='store_true');p.add_argument('--record',action='store_true')
    p.add_argument('--scene',choices=['A_clear','B_sphere','C_cylinder','D_two_obstacles'],default='B_sphere')
    p.add_argument('--speed',type=float,choices=[.10,.15],default=.10)
    a=p.parse_args();dest=destination('phase2c_dynamic_demo')
    e=TrajectoryExecutor(load_verified_path(a.scene,42),a.speed,dest)
    keys=KeyboardQueue();view=None;hook=None;video=None;reason='';result=None;closed=True;video_result=None;visual_errors=[]
    wall=time.perf_counter();last_sync=-1.
    try:
        if a.viewer:
            view=mujoco.viewer.launch_passive(e.model,e.data,key_callback=keys.callback,show_left_ui=False,show_right_ui=False)
            hook=QueuedWindowClose(keys,'MuJoCo : Panda Phase2A kinematic collision inspection')
            e.integration_lock=view.lock
            with view.lock():
                view.cam.lookat[:]=[.2,0,.4];view.cam.distance=1.9;view.cam.azimuth=135;view.cam.elevation=-25
        if a.record:video=DynamicRecording(e,dest)
        def stop():
            nonlocal reason
            if view and not view.is_running():reason='viewer_closed'
            while True:
                try:
                    if keys.commands.get_nowait()[0] in (256,ord('Q')):reason='viewer_closed'
                except queue.Empty:break
            return reason
        def display(executor,row):
            nonlocal last_sync
            if video:
                with view.lock() if view else nullcontext():video.capture(executor,row)
            if view and row['time']-last_sync>=.02-1e-8 and view.is_running():
                gap=row['minimum_clearance_m'];text=(f"DYNAMIC SERVO EXECUTION\n{a.scene} | t={row['time']:.3f}s | segment {row['segment']}\n"
                    f"{row['phase']} | progress {row['progress']*100:.1f}%\nActual min gap {gap*1000:.3f} mm\n"
                    f"TCP ref {[round(row[f'tcp_ref_{i}'],4) for i in range(3)]}\nTCP actual {[round(row[f'tcp_actual_{i}'],4) for i in range(3)]}\n"
                    f"{'SUCCESS' if executor.success else executor.failure_reason or 'ctrl + mj_step; no live qpos teleport'}\nEsc/Q or window X: stop and archive")
                view.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150,mujoco.mjtGridPos.mjGRID_TOPLEFT,text,''))
                view.sync();last_sync=row['time']
                delay=row['time']-(time.perf_counter()-wall)
                if delay>0:time.sleep(min(delay,.02))
        result=e.run(on_step=display,stop_requested=stop)
    finally:
        if view:view.close();closed=hook.wait_destroyed()
        if video:
            try:video_result=video.close()
            except Exception as exc:visual_errors.append(repr(exc))
        write_json(dest/'demo_complete.json',dict(execution=result,viewer_launched=a.viewer,viewer_closed=closed,
             viewer_stop_reason=reason,video=video_result,visual_errors=visual_errors))
    if video_result and not reason:
        shutil.copy2(video.path,ROOT/'assets/videos/phase2c_dynamic_execution.mp4')
    write_json(ROOT/('results/phase2c_closed_viewer_latest.json' if reason else 'results/phase2c_demo_latest.json'),dict(run=dest))
    print('OUTPUT',dest,'success',None if result is None else result['success'],'reason',reason,video_result,flush=True)


if __name__=='__main__':main_guard(main)
