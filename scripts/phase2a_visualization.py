"""Render owned collision-query states; explicitly not servo/dynamic execution."""
import hashlib
import json
import queue
import shutil
import textwrap
import time

import imageio.v2 as imageio
import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.common import ROOT,write_json,sha
from src.robot.teleop import KeyboardQueue
from src.visualization.windows_close import QueuedWindowClose
from scripts.phase2a_cases import restore_scene


def latest_cases():
    pointer=ROOT/'results/phase2a_latest.json'
    if not pointer.exists():raise RuntimeError('Run scripts/run_phase2a_validation.py first')
    run=__import__('pathlib').Path(json.loads(pointer.read_text(encoding='utf-8'))['run'])
    return run,json.loads((run/'collision_cases_full.json').read_text(encoding='utf-8'))


def render_demo(checker,frames,dest,name,*,viewer=False,record=False,fps=10):
    m,d=checker.model,checker.data
    keys=KeyboardQueue();view=None;hook=None;renderer=None;writer=None
    frame_log=[];interrupted=False;closed=True
    robot_visual=[g for g in range(m.ngeom) if checker.name(g).startswith('panda_')]
    camera=mujoco.MjvCamera();camera.lookat[:]=[.2,0,.4];camera.distance=1.9;camera.azimuth=135;camera.elevation=-25
    try:
        if viewer:
            view=mujoco.viewer.launch_passive(m,d,key_callback=keys.callback,show_left_ui=False,show_right_ui=False)
            hook=QueuedWindowClose(keys,'MuJoCo : Panda Phase2A kinematic collision inspection')
            with view.lock():
                view.cam.lookat[:]=camera.lookat;view.cam.distance=camera.distance
                view.cam.azimuth=camera.azimuth;view.cam.elevation=camera.elevation
        if record:
            renderer=mujoco.Renderer(m,height=640,width=960)
            writer=imageio.get_writer(str(dest/f'{name}.mp4'),fps=fps,codec='libx264',macro_block_size=16)
        for index,item in enumerate(frames):
            if view and not view.is_running():interrupted=True;break
            stop=False
            while True:
                try:
                    key=keys.commands.get_nowait()[0]
                    if key in (256,ord('Q')):stop=True
                except queue.Empty:break
            if stop:interrupted=True;break
            lock=view.lock() if view else __import__('contextlib').nullcontext()
            with lock:
                restore_scene(checker,item['scene'])
                r=checker.state_valid(item['q'])
                reference=False
                if not r.joint_limit_valid or r.rejection_reason.startswith('query_error'):
                    # Do not show the preceding pose as if the invalid input was evaluated.
                    checker.state_valid(checker.home);reference=True
                q_draw=checker.robot.positions()
                color=[.82,.12,.10,1.] if not r.valid else [.95,.65,.08,1.] if r.minimum_clearance<.006 else [.15,.68,.35,1.]
                m.geom_rgba[robot_visual]=color
                m.geom_matid[robot_visual]=-1  # appearance only, on the owned model
                draw_state=dict(label=item['label'],requested_q=np.asarray(item['q']).tolist(),
                                displayed_q=q_draw.tolist(),reference_posture_only=reference,
                                valid=r.valid,minimum_clearance=r.minimum_clearance,
                                minimum_is_lower_bound=r.minimum_is_lower_bound,
                                offending_geom_pairs=r.offending_geom_pairs,
                                rejection_reason=r.rejection_reason,query_data_time=float(d.time),
                                source='Kinematic state/interpolation check, NOT dynamic servo execution')
                if index==0 or item['label']!=frames[index-1]['label']:
                    print(item['label'],r.valid,r.rejection_reason,
                          [(p['geom1'],p['geom2'],p['distance_m']) for p in r.offending_geom_pairs[:3]],flush=True)
                if record:
                    renderer.update_scene(d,camera=camera);raw=renderer.render().copy()
                    im=Image.new('RGB',(960,752),'#16212d');im.paste(Image.fromarray(raw),(0,112));pen=ImageDraw.Draw(im)
                    try:font=ImageFont.truetype('arial.ttf',17)
                    except OSError:font=ImageFont.load_default()
                    pen.text((10,4),'KINEMATIC COLLISION CHECK - NOT servo/dynamic execution',font=font,fill='white')
                    badge='INVALID' if not r.valid else 'NEAR MARGIN' if r.minimum_clearance<.006 else 'VALID'
                    minimum='not evaluated' if r.minimum_clearance is None else ('>= ' if r.minimum_is_lower_bound else '')+f'{r.minimum_clearance*1000:.4f} mm'
                    pen.text((10,27),f"{item['label']} | {badge} | min gap {minimum}",font=font,fill=tuple(int(x*255) for x in color[:3]))
                    message='HOME REFERENCE ONLY: rejected before FK' if reference else r.rejection_reason or 'All audited pairs satisfy the configured clearance'
                    pen.text((10,50),message,font=font,fill='white')
                    pair=r.offending_geom_pairs[0] if r.offending_geom_pairs else None
                    pairtext=f"{pair['geom1']} <-> {pair['geom2']}" if pair else 'Green=valid | Yellow=near margin | Red=invalid | simulation time stays 0'
                    for line_i,line in enumerate(textwrap.wrap(pairtext,width=95)[:2]):
                        pen.text((10,73+line_i*18),line,font=font,fill='white')
                    writer.append_data(np.asarray(im))
                    draw_state['raw_rgb_sha256']=hashlib.sha256(raw.tobytes()).hexdigest()
                    if index%10==0:im.save(dest/f'{name}_frame_{index:04d}.png')
                frame_log.append(dict(frame=index,**draw_state))
            if view:
                message=(f"KINEMATIC CHECK / NOT dynamic execution\n{item['label']}\n"
                         f"{'VALID' if r.valid else 'INVALID'} | {r.rejection_reason or 'clear'}\n")
                if reference:message+='HOME REFERENCE ONLY: candidate rejected before FK\n'
                if r.offending_geom_pairs:
                    first=r.offending_geom_pairs[0]
                    message+=first['geom1']+'\n<-> '+first['geom2']
                view.set_texts((mujoco.mjtFontScale.mjFONTSCALE_150,mujoco.mjtGridPos.mjGRID_TOPLEFT,message,''))
                view.sync()
                time.sleep(1/fps)
    finally:
        if view:
            view.close();closed=hook.wait_destroyed()
        if writer:writer.close()
        if renderer:renderer.close()
    result=dict(source='Independent query MjData; mj_forward only; no mj_step; not a planned executable trajectory',
                frames=len(frame_log),expected_frames=len(frames),interrupted=interrupted,viewer_closed=closed,
                native_viewer_launched=viewer,fps=fps,physics_steps=0,
                max_displayed_joint_motion_rad=float(np.max(np.linalg.norm(np.array([r['displayed_q'] for r in frame_log])-frame_log[0]['displayed_q'],axis=1))) if frame_log else 0.)
    if record and frame_log:
        path=dest/f'{name}.mp4'
        with imageio.get_reader(str(path)) as reader:
            decoded=sum(1 for _ in reader)
            shape=reader.get_data(0).shape
        result.update(mp4_sha256=sha(path),decoded_frames=decoded,duration_s=decoded/fps,
                      resolution=list(shape[:2][::-1]),file_bytes=path.stat().st_size,
                      raw_unique_frames=len({r['raw_rgb_sha256'] for r in frame_log}))
        if decoded!=len(frame_log):raise AssertionError('MP4 frame count mismatch')
        if not interrupted:
            shutil.copy2(path,ROOT/'assets/videos'/path.name)
    write_json(dest/f'{name}_frames.json',frame_log)
    write_json(dest/'complete.json',result)
    return result
