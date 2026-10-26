"""Actual M1/M3 uncertain-physics execution, never qpos trajectory playback."""
from _bootstrap import ROOT,destination,main_guard
import argparse,hashlib,queue,shutil,time
import imageio.v2 as imageio
import mujoco,mujoco.viewer
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from src.common import write_json,sha
from scripts.run_phase3a import prepared,read,check_lock,design,UncertainRun,bundle
from src.robot.teleop import KeyboardQueue
from src.visualization.windows_close import QueuedWindowClose

def episode(spec,method,perturbation,dest,viewer):
    runner=UncertainRun(bundle(spec),method,perturbation,3101,dest);e=runner.e;frames=[];last_sync=-1.;aborted=''
    writer=imageio.get_writer(str(dest/'live.mp4'),fps=10,codec='libx264',macro_block_size=16)
    renderer=mujoco.Renderer(e.model,height=640,width=960);camera=mujoco.MjvCamera()
    camera.lookat[:]=[.2,0,.4];camera.distance=1.9;camera.azimuth=135;camera.elevation=-25
    keys=KeyboardQueue();view=None;hook=None;wall=time.perf_counter();result=None;closed=True
    try:
        if viewer:
            view=mujoco.viewer.launch_passive(e.model,e.data,key_callback=keys.callback,show_left_ui=False,show_right_ui=False)
            hook=QueuedWindowClose(keys,'MuJoCo : Panda Phase2A kinematic collision inspection');e.integration_lock=view.lock
            with view.lock():view.cam.lookat[:]=camera.lookat;view.cam.distance=camera.distance;view.cam.azimuth=camera.azimuth;view.cam.elevation=camera.elevation
        def stop():
            nonlocal aborted
            if view and not view.is_running():aborted='viewer_closed'
            while True:
                try:
                    if keys.commands.get_nowait()[0] in (256,ord('Q')):aborted='viewer_closed'
                except queue.Empty:break
            return aborted
        def display(run,row):
            nonlocal last_sync
            if not frames or row['time']-frames[-1]['time']>=.1-1e-9 or e.success or (e.first_violation and frames[-1]['phase']!='protective_hold' and row['phase']=='protective_hold'):
                renderer.update_scene(e.data,camera=camera);rgb=renderer.render().copy()
                im=Image.new('RGB',(960,816),'#142334');im.paste(Image.fromarray(rgb),(0,176));draw=ImageDraw.Draw(im)
                try:font=ImageFont.truetype('arial.ttf',17)
                except OSError:font=ImageFont.load_default()
                state='GOAL HELD' if e.success else e.failure_reason or row['phase']
                lines=[f'DYNAMIC SERVO EXECUTION | {method} | validation_cylinder | {perturbation["name"]}',
                    'Nominal controller parameters / independently perturbed physics / native force limits unchanged',
                    f'sim={row["time"]:.3f}s | progress={row["progress"]*100:.1f}% | {state}',
                    f'Actual gap={row["minimum_clearance_m"]*1000:.4f}mm | hard threshold=5mm | TCP error={row["tcp_error_m"]*1000:.3f}mm',
                    f'Kp factor={perturbation.get("kp_scale",1)} | mass/inertia={perturbation.get("mass_inertia_scale",1)} | seed=3101',
                    f'Actual forbidden contacts now={row["prohibited_contact_count"]} | physical timestep=0.002s',
                    'Actual integrating live MjData / no qpos playback / supervisory noise and delay only']
                for i,line in enumerate(lines):draw.text((8,5+24*i),line,font=font,fill='#ffb9a9' if e.failure_reason else 'white')
                writer.append_data(np.asarray(im));frames.append(dict(frame=len(frames),time=row['time'],physics_step=row['physics_step'],phase=row['phase'],
                    q_actual=e.robot.positions(),q_ref=[row[f'q_ref_{j}'] for j in range(1,8)],ctrl=e.data.ctrl[e.robot.actuator_ids].copy(),
                    raw_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),live_data_id=id(e.data)))
            if view and view.is_running() and row['time']-last_sync>=.02-1e-9:
                view.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150,mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    f'DYNAMIC SERVO EXECUTION | {method} / {perturbation["name"]}\nt={row["time"]:.3f}s | actual gap={row["minimum_clearance_m"]*1000:.3f}mm\n{e.failure_reason or row["phase"]}',''))
                view.sync();last_sync=row['time'];pause=row['time']-(time.perf_counter()-wall)
                if pause>0:time.sleep(min(pause,.02))
        result=runner.run(display=display,stop_requested=stop)
    finally:
        if view:view.close();closed=hook.wait_destroyed()
        writer.close();renderer.close();write_json(dest/'video_frames.json',frames)
    with imageio.get_reader(str(dest/'live.mp4')) as reader:decoded=sum(1 for _ in reader)
    assert decoded==len(frames)
    import csv
    with (dest/'tracking.csv').open(encoding='utf-8-sig') as f:tracking={int(x['physics_step']):x for x in csv.DictReader(f)}
    for frame in frames:
        row=tracking[frame['physics_step']]
        for key in ('q_actual','q_ref','ctrl'):assert np.array_equal(frame[key],np.array([float(row[f'{key}_{j}']) for j in range(1,8)]))
        assert row['time']==str(frame['time']) and frame['live_data_id']==id(e.data)
    validation=dict(frames=decoded,duration_s=decoded/10,unique_rgb_frames=len({x['raw_rgb_sha256'] for x in frames}),
        joint_displacement_rad=float(np.max(np.linalg.norm(np.array([x['q_actual'] for x in frames])-frames[0]['q_actual'],axis=1))),
        all_frames_match_actual_CSV=True,viewer_launched=viewer,viewer_closed=closed,aborted=aborted,sha256=sha(dest/'live.mp4'))
    write_json(dest/'video_validation.json',validation);return dict(result=result,validation=validation,folder=dest)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--viewer',action='store_true');a=p.parse_args();check_lock()
    dest=destination('phase3a_video');source=next(x for x in read(prepared()/'plans.json') if x['split']=='validation' and x['success'])
    spec=dict(kind='new',source_json=source['source_json']);records=[]
    # Fixed illustrative condition, not chosen for apparent success; all outcomes retained.
    jobs=[('M1',design()['perturbations'][0]),('M1',design()['perturbations'][-1]),('M3',design()['perturbations'][-1])]
    for method,perturbation in jobs:
        r=episode(spec,method,perturbation,dest/f"{method}_{perturbation['name']}",a.viewer);records.append(r)
        print('VIDEO',method,perturbation['name'],r['result']['status'],flush=True)
        if r['validation']['aborted']:break
    count=0;path=dest/'phase3a_robustness_comparison.mp4'
    if len(records)==3 and not any(x['validation']['aborted'] for x in records):
        with imageio.get_writer(str(path),fps=10,codec='libx264',macro_block_size=16) as writer:
            for r in records:
                with imageio.get_reader(str(r['folder']/'live.mp4')) as reader:
                    for frame in reader:writer.append_data(frame);count+=1
        with imageio.get_reader(str(path)) as reader:assert sum(1 for _ in reader)==count
        shutil.copy2(path,ROOT/'assets/videos'/path.name)
    write_json(dest/'complete.json',dict(episodes=records,frames=count,independent_initializations=True,only_live_integrations=True))
    write_json(ROOT/'results/phase3a_video_latest.json',dict(run=dest));print('COMPLETE',dest,flush=True)

if __name__=='__main__':main_guard(main)
