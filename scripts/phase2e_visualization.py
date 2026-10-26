"""Record actual M0/M1/M2 integrations, never shadow prediction frames."""
import argparse,hashlib,json,queue,shutil,time
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
import mujoco
import mujoco.viewer
from PIL import Image,ImageDraw,ImageFont
from src.common import ROOT,sha,write_json
from src.control.phase2e_runtime import ControllerRun
from src.control.validated_path import load_verified_path
from scripts._bootstrap import destination
from src.robot.teleop import KeyboardQueue
from src.visualization.windows_close import QueuedWindowClose


class LiveControlVideo:
    def __init__(self,run):
        self.run=run;self.e=run.e;self.dest=run.dest;self.frames=[];self.path=self.dest/'live_control.mp4'
        self.writer=imageio.get_writer(str(self.path),fps=10,codec='libx264',macro_block_size=16)
        self.renderer=mujoco.Renderer(self.e.model,height=640,width=960);self.camera=mujoco.MjvCamera()
        self.camera.lookat[:]=[.2,0,.4];self.camera.distance=1.9;self.camera.azimuth=135;self.camera.elevation=-25
        self.last_event=False
    def capture(self,run,row):
        event=bool(run.e.first_violation) and not self.last_event
        if self.frames and row['time']-self.frames[-1]['time']<.1-1e-9 and not event and not run.e.success:return
        e=run.e;self.last_event=bool(e.first_violation)
        self.renderer.update_scene(e.data,camera=self.camera);raw=self.renderer.render().copy()
        im=Image.new('RGB',(960,816),'#142334');im.paste(Image.fromarray(raw),(0,176));draw=ImageDraw.Draw(im)
        try:font=ImageFont.truetype('arial.ttf',16)
        except OSError:font=ImageFont.load_default()
        warning=run.guard.first_warning_time if run.guard else None
        outcome='SUCCESS / goal held' if e.success else 'GOAL NOT REACHED / '+e.failure_reason if e.failure_reason else row['phase']
        q_error=max(abs(row[f'q_actual_{j}']-row[f'q_ref_{j}']) for j in range(1,8))
        lines=[f'DYNAMIC SERVO EXECUTION | {run.method} | original saved D / seed 42',
            f'Gravity compensation {"OFF" if run.method=="M0" else "ON"} | guard {run.guard_mode} | hard execution threshold 5 mm',
            f"sim t={row['time']:.3f}s | progress={row['progress']*100:.1f}% | segment={row['segment']} | {outcome}",
            f"Actual min gap={row['minimum_clearance_m']*1000:.4f} mm | TCP error={row['tcp_error_m']*1000:.3f} mm | max joint error={q_error:.5f} rad",
            f"TCP ref {[round(row[f'tcp_ref_{j}'],4) for j in range(3)]} / actual {[round(row[f'tcp_actual_{j}'],4) for j in range(3)]}",
            f'First warning: {warning if warning is not None else "none"} | first actual <5mm: {run.first_hard_violation} | first contact: {run.first_contact}',
            'Same integrating live MjData / original native torque limits / no qpos playback']
        for i,line in enumerate(lines):draw.text((8,4+24*i),line,font=font,fill='#ffb9a9' if e.failure_reason else 'white')
        self.writer.append_data(np.asarray(im))
        self.frames.append(dict(frame=len(self.frames),physics_step=row['physics_step'],time=row['time'],method=run.method,
            q_actual=e.robot.positions().tolist(),q_ref=[row[f'q_ref_{j}'] for j in range(1,8)],ctrl=e.data.ctrl[e.robot.actuator_ids].tolist(),
            raw_rgb_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),live_data_id=id(e.data),status=outcome))
        if event:im.save(self.dest/'protective_event.png')
        if e.success:im.save(self.dest/'success.png')
    def close(self):
        self.writer.close();self.renderer.close()
        with imageio.get_reader(str(self.path)) as reader:decoded=sum(1 for _ in reader)
        assert decoded==len(self.frames)
        q=np.array([f['q_actual'] for f in self.frames]);result=dict(frames=decoded,duration_s=decoded/10,
            unique_raw_frames=len({f['raw_rgb_sha256'] for f in self.frames}),all_same_live_data=all(f['live_data_id']==id(self.e.data) for f in self.frames),
            maximum_joint_displacement_rad=float(np.max(np.linalg.norm(q-q[0],axis=1))),sha256=sha(self.path))
        write_json(self.dest/'video_frames.json',self.frames);write_json(self.dest/'video_validation.json',result);return result


def episode(method,speed,dest,viewer):
    run=ControllerRun(load_verified_path('D_two_obstacles',42),method,speed,dest);e=run.e
    keys=KeyboardQueue();view=None;hook=None;closed=True;aborted='';last=-1.;wall=time.perf_counter();video=None;r=None;v=None
    try:
        if viewer:
            view=mujoco.viewer.launch_passive(e.model,e.data,key_callback=keys.callback,show_left_ui=False,show_right_ui=False)
            hook=QueuedWindowClose(keys,'MuJoCo : Panda Phase2A kinematic collision inspection');e.integration_lock=view.lock
            with view.lock():view.cam.lookat[:]=[.2,0,.4];view.cam.distance=1.9;view.cam.azimuth=135;view.cam.elevation=-25
        video=LiveControlVideo(run)
        def stop():
            nonlocal aborted
            if view and not view.is_running():aborted='viewer_closed'
            while True:
                try:
                    if keys.commands.get_nowait()[0] in (256,ord('Q')):aborted='viewer_closed'
                except queue.Empty:break
            return aborted
        def display(runner,row):
            nonlocal last
            if view:
                with view.lock():video.capture(runner,row)
            else:video.capture(runner,row)
            if view and view.is_running() and row['time']-last>=.02-1e-9:
                warning=runner.guard.first_warning_time if runner.guard else None
                text=(f"DYNAMIC SERVO EXECUTION | {method}\ncompensation {'OFF' if method=='M0' else 'ON'} / guard {runner.guard_mode}\n"
                    f"t={row['time']:.3f}s progress={row['progress']*100:.1f}%\nActual gap {row['minimum_clearance_m']*1000:.4f}mm / threshold 5mm\n"
                    f"First warning {warning}\n{'SUCCESS' if e.success else e.failure_reason or row['phase']}")
                view.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150,mujoco.mjtGridPos.mjGRID_TOPLEFT,text,''));view.sync();last=row['time']
                delay=row['time']-(time.perf_counter()-wall)
                if delay>0:time.sleep(min(delay,.02))
        r=run.run(display=display,external_stop=stop)
    finally:
        if view:view.close();closed=hook.wait_destroyed()
        if video:v=video.close()
        write_json(dest/'demo_complete.json',dict(execution=r,video=v,viewer_launched=viewer,viewer_closed=closed,aborted=aborted))
    return dict(method=method,execution=r,video=v,aborted=aborted)


def main(kind):
    parser=argparse.ArgumentParser(description='Actual live control comparison; no kinematic trajectory playback')
    parser.add_argument('--viewer',action='store_true');parser.add_argument('--speed',type=float,choices=[.10,.15],default=.15)
    args=parser.parse_args();dest=destination('phase2e_'+kind+'_video');methods=['M0','M1'] if kind=='controller' else ['M1','M2'];records=[]
    for method in methods:
        rec=episode(method,args.speed,dest/method,args.viewer);records.append(rec);print('VIDEO',kind,method,rec['execution']['outcome'],flush=True)
        if rec['aborted']:break
    name='phase2e_controller_comparison.mp4' if kind=='controller' else 'phase2e_safety_guard.mp4';count=0
    if len(records)==2 and not records[-1]['aborted']:
        with imageio.get_writer(str(dest/name),fps=10,codec='libx264',macro_block_size=16) as writer:
            for rec in records:
                with imageio.get_reader(str(dest/rec['method']/'live_control.mp4')) as reader:
                    for image in reader:writer.append_data(image);count+=1
        with imageio.get_reader(str(dest/name)) as reader:decoded=sum(1 for _ in reader)
        assert decoded==count==sum(x['video']['frames'] for x in records)
        shutil.copy2(dest/name,ROOT/'assets/videos'/name)
    write_json(dest/'complete.json',dict(episodes=records,frames=count,duration_s=count/10,independent_initializations=True,source='only actual live MjData integrations'))
    write_json(ROOT/f'results/phase2e_{kind}_video_latest.json',dict(run=dest));print('COMPLETE',dest,flush=True)
