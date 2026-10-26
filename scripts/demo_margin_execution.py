"""Two independent actual live dynamics episodes: old failure and margin path."""
from _bootstrap import destination,main_guard
import argparse,hashlib,json,queue,shutil,time
from contextlib import nullcontext
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
import mujoco
import mujoco.viewer
from PIL import Image,ImageDraw,ImageFont
from src.common import ROOT,write_json,sha
from src.control.validated_path import load_verified_path
from src.control.trajectory_executor import TrajectoryExecutor
from src.planning.margin_study import load_margin_path
from src.robot.teleop import KeyboardQueue
from src.visualization.windows_close import QueuedWindowClose


class MarginVideo:
    def __init__(self,e,dest,label,planning_margin):
        self.e=e;self.dest=dest;self.label=label;self.margin=planning_margin;self.frames=[];self.first_failure_frame=None
        self.path=dest/'actual_dynamic.mp4';self.writer=imageio.get_writer(str(self.path),fps=10,codec='libx264',macro_block_size=16)
        self.renderer=mujoco.Renderer(e.model,height=640,width=960);self.camera=mujoco.MjvCamera()
        self.camera.lookat[:]=[.2,0,.4];self.camera.distance=1.9;self.camera.azimuth=135;self.camera.elevation=-25
    def capture(self,e,row):
        first_failure=e.first_violation is not None and self.first_failure_frame is None
        if self.frames and row['time']-self.frames[-1]['time']<.1-1e-9 and not first_failure and not e.success:return
        self.renderer.update_scene(e.data,camera=self.camera);raw=self.renderer.render().copy()
        im=Image.new('RGB',(960,800),'#15212f');im.paste(Image.fromarray(raw),(0,160));pen=ImageDraw.Draw(im)
        try:font=ImageFont.truetype('arial.ttf',17)
        except OSError:font=ImageFont.load_default()
        state='SUCCESS / stable goal' if e.success else 'FAILED / '+e.failure_reason if e.failure_reason else row['phase']
        lines=['DYNAMIC SERVO EXECUTION | '+self.label,
            f'PLANNING threshold {self.margin*1000:.1f} mm | EXECUTION threshold 5.0 mm (unchanged)',
            f"t={row['time']:.3f}s | segment {row['segment']} | progress {row['progress']*100:.1f}% | {state}",
            f"Actual min gap {row['minimum_clearance_m']*1000:.4f} mm | TCP error {row['tcp_error_m']*1000:.3f} mm",
            f"TCP ref {[round(row[f'tcp_ref_{i}'],4) for i in range(3)]} / actual {[round(row[f'tcp_actual_{i}'],4) for i in range(3)]}",
            f"FIRST 5mm VIOLATION at t={e.first_violation['time']:.3f}s; reference stopped" if e.first_violation else 'Actual live MjData / original position servo / ctrl + mj_step']
        for i,line in enumerate(lines):pen.text((10,5+25*i),line,font=font,fill='#ff8d86' if e.first_violation and i>=2 else 'white')
        self.writer.append_data(np.asarray(im))
        if first_failure:self.first_failure_frame=len(self.frames);im.save(self.dest/'first_violation.png')
        if e.success:im.save(self.dest/'final_success.png')
        self.frames.append(dict(frame=len(self.frames),time=row['time'],physics_step=row['physics_step'],state=state,
            q_actual=e.robot.positions().tolist(),q_ref=[row[f'q_ref_{j}'] for j in range(1,8)],ctrl=e.controller.command_q.tolist(),
            raw_rgb_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),live_data_id=id(e.data),planning_clearance_m=self.margin,execution_clearance_m=.005))
    def close(self):
        self.writer.close();self.renderer.close()
        with imageio.get_reader(str(self.path)) as reader:n=sum(1 for _ in reader)
        assert n==len(self.frames)
        result=dict(frames=n,duration_s=n/10,raw_unique_frames=len({x['raw_rgb_sha256'] for x in self.frames}),
            first_violation_frame=self.first_failure_frame,all_same_live_data=all(x['live_data_id']==id(self.e.data) for x in self.frames),
            maximum_joint_displacement_rad=float(np.max(np.linalg.norm(np.array([f['q_actual'] for f in self.frames])-self.frames[0]['q_actual'],axis=1))),
            planning_clearance_m=self.margin,execution_clearance_m=.005,sha256=sha(self.path))
        write_json(self.dest/'video_frames.json',self.frames);write_json(self.dest/'video_validation.json',result);return result


def episode(bundle,dest,label,planning_margin,viewer):
    e=TrajectoryExecutor(bundle,.10,dest);keys=KeyboardQueue();view=None;hook=None;closed=True;aborted='';last=-1.;wall=time.perf_counter()
    recording=None;result=None;video=None
    try:
        if viewer:
            view=mujoco.viewer.launch_passive(e.model,e.data,key_callback=keys.callback,show_left_ui=False,show_right_ui=False)
            hook=QueuedWindowClose(keys,'MuJoCo : Panda Phase2A kinematic collision inspection');e.integration_lock=view.lock
            with view.lock():view.cam.lookat[:]=[.2,0,.4];view.cam.distance=1.9;view.cam.azimuth=135;view.cam.elevation=-25
        recording=MarginVideo(e,dest,label,planning_margin)
        def stop():
            nonlocal aborted
            if view and not view.is_running():aborted='viewer_closed'
            while True:
                try:
                    if keys.commands.get_nowait()[0] in (256,ord('Q')):aborted='viewer_closed'
                except queue.Empty:break
            return aborted
        def display(ex,row):
            nonlocal last
            with view.lock() if view else nullcontext():recording.capture(ex,row)
            if view and view.is_running() and row['time']-last>=.02-1e-9:
                view.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150,mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    f"DYNAMIC SERVO EXECUTION\n{label}\nPlanning {planning_margin*1000:.1f} mm / execution 5.0 mm\n"
                    f"t={row['time']:.3f}s progress {row['progress']*100:.1f}%\nActual gap {row['minimum_clearance_m']*1000:.4f} mm\n"
                    f"{'SUCCESS' if e.success else e.failure_reason or row['phase']}",''))
                view.sync();last=row['time'];delay=row['time']-(time.perf_counter()-wall)
                if delay>0:time.sleep(min(delay,.02))
        result=e.run(on_step=display,stop_requested=stop)
    finally:
        if view:view.close();closed=hook.wait_destroyed()
        if recording:video=recording.close()
        write_json(dest/'demo_complete.json',dict(execution=result,video=video,viewer_launched=viewer,viewer_closed=closed,aborted=aborted))
    return result,video,aborted


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--viewer',action='store_true')
    p.add_argument('--margin-mm',type=int,choices=[8,10,15],default=8);p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();dest=destination('phase2d_margin_video')
    plan=Path(json.loads((ROOT/'results/phase2d_planning_latest.json').read_text())['run'])
    bundles=[('A_original_failed_baseline',load_verified_path('D_two_obstacles',42),.005),
             ('B_replanned_margin',load_margin_path(plan/f'margin{a.margin_mm:02d}mm_seed{a.seed}.json'),a.margin_mm/1000)]
    records=[]
    for name,bundle,margin in bundles:
        r,v,aborted=episode(bundle,dest/name,name,margin,a.viewer)
        records.append(dict(episode=name,execution=r,video=v));print(name,r['success'],r['failure_reason'],flush=True)
        if aborted:break
    if len(records)==2 and not aborted:
        combined=dest/'phase2d_margin_comparison.mp4';count=0
        with imageio.get_writer(str(combined),fps=10,codec='libx264',macro_block_size=16) as writer:
            for rec in records:
                with imageio.get_reader(str(dest/rec['episode']/'actual_dynamic.mp4')) as reader:
                    for frame in reader:writer.append_data(frame);count+=1
        with imageio.get_reader(str(combined)) as reader:decoded=sum(1 for _ in reader)
        assert count==decoded==sum(r['video']['frames'] for r in records)
        shutil.copy2(combined,ROOT/'assets/videos/phase2d_margin_comparison.mp4')
        write_json(dest/'comparison_validation.json',dict(frames=decoded,duration_s=decoded/10,resolution=[960,800],
            independent_episodes=True,source='Two actual live dynamics recordings; time restarts at each explicitly labelled initialization',sha256=sha(combined)))
    write_json(dest/'complete.json',dict(episodes=records))
    write_json(ROOT/'results/phase2d_video_latest.json',dict(run=dest));print('COMPLETE',dest,flush=True)


if __name__=='__main__':main_guard(main)
