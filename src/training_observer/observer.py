"""Physics-owner capture. No step/reset/predict and no random-number operations."""
from pathlib import Path
import hashlib,json,os,threading,time
import numpy as np
import mujoco
from PIL import Image,ImageDraw,ImageFont
from src.common import write_json
from src.experiments.phase3c import DiagnosticEnv

STATE=mujoco.mjtState.mjSTATE_INTEGRATION


def state_vector(model,data):
    out=np.empty(mujoco.mj_stateSize(model,STATE));mujoco.mj_getState(model,data,out,STATE);return out


def atomic_json(path,content):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(content,ensure_ascii=False),encoding='utf-8')
    for attempt in range(8):
        try:os.replace(tmp,path);return
        except PermissionError:
            if attempt==7:raise
            time.sleep(.002)


def annotate(rgb,lines):
    im=Image.new('RGB',(960,832),'#132133');im.paste(Image.fromarray(rgb).resize((960,544)),(0,288))
    pen=ImageDraw.Draw(im)
    try:font=ImageFont.truetype('consola.ttf',15)
    except OSError:font=ImageFont.load_default()
    for i,line in enumerate(lines[:15]):pen.text((8,5+18*i),str(line),font=font,fill='white')
    return np.array(im)


def numbers(values):return ' '.join(f'{v:+.4f}' for v in values)


class EpisodeCapture:
    def __init__(self,env,e,video):
        self.owner=threading.get_ident();self.env=env;self.e=e;self.video=video;self.states=[];self.qpos=[];self.qvel=[];self.ctrl=[];self.times=[];self.steps=[]
        self.frames=[];self.renderer=None;self.writer=None;self.next_time=0.
        if video:
            import imageio.v2 as imageio
            self.renderer=mujoco.Renderer(e.model,height=544,width=960)
            self.cam=mujoco.MjvCamera();self.cam.lookat[:]=[.2,0,.4];self.cam.distance=1.85;self.cam.azimuth=135;self.cam.elevation=-25
            self.writer=imageio.get_writer(str(env.folder/'actual_rgb.mp4'),fps=env.record_fps,codec='libx264',macro_block_size=16)

    def capture(self,row):
        assert threading.get_ident()==self.owner
        e=self.e;env=self.env;before=state_vector(e.model,e.data)
        self.states.append(before);self.qpos.append(e.data.qpos.copy());self.qvel.append(e.data.qvel.copy());self.ctrl.append(e.data.ctrl.copy());self.times.append(float(e.data.time));self.steps.append(e.steps)
        if not self.video or row['time']<self.next_time-1e-9:return
        self.next_time=float(row['time'])+1/env.record_fps
        self.renderer.update_scene(e.data,camera=self.cam);rgb=self.renderer.render().copy()
        assert np.array_equal(before,state_vector(e.model,e.data)),'Rendering modified integration state'
        intent=env.intent if env.active_decision else {};raw=intent.get('raw',[]);clipped=intent.get('clipped',[]);actual=e.controller.last['residual_rad'].copy();target=e.controller.last['residual_target_rad'].copy()
        meta=dict(frame=len(self.frames),episode=env.episode,physics_step=e.steps,time=float(e.data.time),global_decision=env.active_decision,
            policy_version=env.policy_version,source='ACTUAL STOCHASTIC TRAINING / live post-integration RGB',
            raw_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),qpos=e.data.qpos.tolist(),qvel=e.data.qvel.tolist(),ctrl=e.data.ctrl.tolist(),
            q_actual=e.robot.positions().tolist(),q_ref=[row[f'q_ref_{j}'] for j in range(1,8)],raw_action=raw,clipped_action=clipped,
            target_residual=target.tolist(),applied_residual=actual.tolist(),tcp_error_m=row['tcp_error_m'],minimum_clearance_m=row['minimum_clearance_m'])
        lines=['ACTUAL STOCHASTIC TRAINING | LIVE RGB / ctrl + mj_step',
            f'Episode {env.episode} | decision {env.active_decision} | policy update v{env.policy_version} | sim {e.data.time:.3f}s | {row["phase"]}',
            'Actor mean: '+numbers(intent.get('mean',[])),'Actor std: '+numbers(intent.get('std',[])),
            'raw: '+numbers(raw),'clipped: '+numbers(clipped),'target residual rad: '+numbers(target),'applied residual rad: '+numbers(actual),
            'q_ref: '+numbers(meta['q_ref']),'q_actual: '+numbers(meta['q_actual']),
            f'TCP error {row["tcp_error_m"]*1000:.4f}mm | actual gap {row["minimum_clearance_m"]*1000:.4f}mm | threshold 5mm']
        meta.update(last_completed_reward=env.last_reward,last_completed_reward_components=env.last_components)
        self.writer.append_data(rgb);self.frames.append(meta)
        lines+=['Last completed decision reward: '+('N/A' if env.last_reward is None else f'{env.last_reward:+.7f}'),
            'Current decision reward becomes available AFTER its physical interval.',
            'Status: '+('SUCCESS' if e.success else e.failure_reason or 'COLLECTING EXPERIENCE')]
        if env.render_training:
            live_folder=env.dest/'live_frames';live_folder.mkdir(exist_ok=True)
            path=live_folder/f'episode{env.episode:04d}_step{e.steps:06d}.png'
            Image.fromarray(annotate(rgb,lines)).save(path);meta['rgb_path']=str(path.resolve())
        atomic_json(env.dest/'live_status.json',dict(status='COLLECTING EXPERIENCE',**meta))

    def close(self):
        assert threading.get_ident()==self.owner
        if self.writer:self.writer.close()
        if self.renderer:self.renderer.close()
        np.savez_compressed(self.env.folder/'integration_states.npz',states=np.asarray(self.states),qpos=np.asarray(self.qpos),qvel=np.asarray(self.qvel),
            ctrl=np.asarray(self.ctrl),time=np.asarray(self.times),physics_step=np.asarray(self.steps),state_spec=int(STATE),
            mocap_pos=self.e.data.mocap_pos.copy(),mocap_quat=self.e.data.mocap_quat.copy())
        write_json(self.env.folder/'rgb_frames.json',self.frames)


class ObservedEnv(DiagnosticEnv):
    def __init__(self,*args,record=False,render_training=False,record_every=1,max_recorded=4,record_fps=10,**kwargs):
        self.record=record;self.render_training=render_training;self.record_every=record_every;self.max_recorded=max_recorded;self.record_fps=record_fps
        self.recorder=None;self.capture_failed=False;self.record_errors=[];self.policy_version=0;self.active_decision=0;self.intent={};self.last_reward=None;self.last_components={};self.capture_lock=threading.Lock()
        super().__init__(*args,**kwargs)

    def reset(self,**kwargs):
        self.active_decision=0;self.intent={};self.last_reward=None;self.last_components={}
        out=super().reset(**kwargs);return out

    def _on_step(self,e,row):
        if self.record and not self.capture_failed:
            try:
                if self.recorder is None:
                    selected=self.episode%self.record_every==0 and self.episode//self.record_every<self.max_recorded
                    self.recorder=EpisodeCapture(self,e,selected)
                with self.capture_lock:self.recorder.capture(row)
            except Exception as exc:
                self.record_errors.append(dict(episode=self.episode,step=e.steps,error=repr(exc),capture_disabled=True));self.capture_failed=True
        # Must notify only after copying/rendering the stable owner state.
        super()._on_step(e,row)

    def _worker(self,*args):
        try:super()._worker(*args)
        finally:
            if self.recorder:
                try:
                    with self.capture_lock:self.recorder.close()
                except Exception as exc:self.record_errors.append(dict(episode=self.episode,error=repr(exc)))
                self.recorder=None

    def step(self,action):
        self.active_decision=self.total_decisions+1
        out=super().step(action);self.last_reward=float(out[1]);self.last_components=out[4]['reward_components_selected']
        return out
