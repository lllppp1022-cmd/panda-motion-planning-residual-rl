"""Native passive viewer and direct-frame recording of actual integration results."""
import time,hashlib
from contextlib import nullcontext
import numpy as np
import mujoco
from PIL import Image,ImageDraw,ImageFont,ImageSequence
from src.common import write_json,write_csv,sha
class LiveViewer:
    def __init__(self,robot,enabled=False,speed=1.):
        self.robot=robot;self.enabled=enabled;self.speed=speed;self.handle=None;self.started=time.perf_counter();self.was_running=False
        if enabled:
            import mujoco.viewer
            self.handle=mujoco.viewer.launch_passive(robot.model,robot.data,show_left_ui=False,show_right_ui=False)
            with self.handle.lock():
                self.handle.cam.lookat[:]=[.25,0,.45];self.handle.cam.distance=1.8;self.handle.cam.azimuth=135;self.handle.cam.elevation=-25
            self.handle.sync();self.was_running=self.handle.is_running();assert self.was_running
    def lock(self):return self.handle.lock() if self.handle else nullcontext()
    def running(self):return self.handle is None or self.handle.is_running()
    def sync(self):
        if self.handle:
            self.handle.sync();delay=self.robot.data.time/self.speed-(time.perf_counter()-self.started)
            if delay>0:time.sleep(min(delay,.02))
    def close(self):
        if self.handle:
            self.handle.close()
            deadline=time.perf_counter()+3
            while self.handle.is_running() and time.perf_counter()<deadline:time.sleep(.01)
            return dict(launched=True,was_running=self.was_running,closed=not self.handle.is_running())
        return dict(launched=False,was_running=False,closed=True)
class Recorder:
    def __init__(self,robot,dest,name,enabled=False,fps=25):
        self.robot=robot;self.dest=dest;self.name=name;self.enabled=enabled;self.fps=fps;self.frames=[];self.records=[];self.mp4=None;self.error=None;self.renderer=None
        if enabled:
            self.renderer=mujoco.Renderer(robot.model,height=480,width=640);self.camera=mujoco.MjvCamera();self.camera.lookat[:]=[.25,0,.45];self.camera.distance=1.8;self.camera.azimuth=135;self.camera.elevation=-25
            try:
                import imageio.v2 as imageio
                self.mp4=imageio.get_writer(str(dest/f'{name}.mp4'),fps=fps,codec='libx264',macro_block_size=16)
            except Exception as e:self.error=repr(e)
            self.first_q=robot.positions();self.motion=0.
    def capture(self,label,force=False):
        if not self.enabled:return
        d=self.robot.data
        if self.records and not force and d.time-self.records[-1]['time']<1/self.fps-1e-8:return
        self.renderer.update_scene(d,camera=self.camera);frame=self.renderer.render().copy();im=Image.new('RGB',(640,528),'#182632');im.paste(Image.fromarray(frame),(0,48));draw=ImageDraw.Draw(im)
        try:font=ImageFont.truetype('DejaVuSans.ttf',14)
        except OSError:font=ImageFont.load_default()
        x,_=self.robot.tool_pose();draw.text((8,4),'Actual MuJoCo dynamics | '+label,fill='white',font=font);draw.text((8,25),f't={d.time:.3f}s | tool world m={x.round(4)}',fill='#65e6b6',font=font)
        self.frames.append(im.quantize(colors=128,method=Image.Quantize.FASTOCTREE))
        if self.mp4:self.mp4.append_data(np.asarray(im))
        self.motion=max(self.motion,float(np.linalg.norm(self.robot.positions()-self.first_q)))
        self.records.append(dict(frame=len(self.records),time=float(d.time),raw_rgb_sha256=hashlib.sha256(frame.tobytes()).hexdigest(),label=label,q=self.robot.positions().tolist(),tool_world=x.tolist()))
        if len(self.records)==1:im.save(self.dest/f'{self.name}_first.png')
        self.last=im
    def close(self):
        if not self.enabled:return None
        if self.mp4:self.mp4.close()
        if self.renderer:self.renderer.close()
        if not self.frames:return dict(error='No actual frames captured')
        path=self.dest/f'{self.name}.gif';self.frames[0].save(path,save_all=True,append_images=self.frames[1:],duration=round(1000/self.fps),loop=0,optimize=False);self.last.save(self.dest/f'{self.name}_last.png')
        count=0;duration=0;hashes=set()
        with Image.open(path) as im:
            for frame in ImageSequence.Iterator(im):hashes.add(hashlib.sha256(frame.convert('RGB').tobytes()).hexdigest());duration+=frame.info.get('duration',0);count+=1
        result=dict(source='Actual mj_step dynamics; rendered directly from the same live MjData',path=str(path),sha256=sha(path),raw_frames=len(self.frames),raw_unique_frames=len({r['raw_rgb_sha256'] for r in self.records}),encoded_frames=count,unique_decoded_frames=len(hashes),duration_ms=duration,joint_displacement_rad=self.motion,mp4_path=str(self.dest/f'{self.name}.mp4') if self.mp4 else None,mp4_error=self.error)
        if self.mp4:
            import imageio.v2 as imageio
            with imageio.get_reader(result['mp4_path']) as reader:result['mp4_decoded_frames']=sum(1 for _ in reader)
            assert result['mp4_decoded_frames']==len(self.frames)
        assert count>1 and len(hashes)>1 and self.motion>.02
        write_json(self.dest/f'{self.name}_video_validation.json',result);write_json(self.dest/f'{self.name}_frame_log.json',self.records)
        self.frames=[];return result
