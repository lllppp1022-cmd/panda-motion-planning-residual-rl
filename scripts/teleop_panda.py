"""Native Windows keyboard teleoperation; no scripted/replayed motion by default."""
from _bootstrap import ROOT, destination, main_guard
import argparse
import csv
import json
import queue
import time
import threading
import faulthandler
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from src.common import write_json, sha
from src.robot.panda_interface import PandaInterface
from src.robot.teleop import HELP, KeyboardQueue, TeleopController, load_options, load_practice_model
from src.visualization.viewer import Recorder
from src.visualization.windows_close import QueuedWindowClose


class StreamCSV:
    def __init__(self, path):
        self.file = path.open('w', encoding='utf-8-sig', newline='')
        self.writer = None

    def append(self, row):
        row = {k: json.dumps(v, ensure_ascii=False, allow_nan=False) if isinstance(v, (list, dict)) else v for k, v in row.items()}
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()


def status_text(s):
    return (f"{s['mode']} | joint {s['selected_joint']} | sim {s['time']:.3f}s\n"
            f"q rad: {np.round(s['q'], 3)}\n"
            f"tool m: {np.round(s['actual_xyz_m'], 4)}\n"
            f"goal m: {np.round(s['target_xyz_m'], 4)} | error {s['error_mm']:.2f} mm\n"
            f"IK {s['ik_status']} | limit {s['limit'] or 'no'}\n"
            f"external {s['external_contact_count']} | self {s['self_contact_count']} | {s['alert'] or 'normal'}\n"
            '1..7 J/K | M:mode | W/S A/D R/F | Space:stop H:home Esc:exit')


def main():
    faulthandler.enable()
    p = argparse.ArgumentParser(description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', type=Path)
    p.add_argument('--record', action='store_true', help='Record actual integrated states, bounded by record_seconds')
    p.add_argument('--max-seconds', type=float, help='Optional wall-clock timeout for GUI smoke tests; no generated commands')
    a = p.parse_args()
    if a.max_seconds is not None and (not np.isfinite(a.max_seconds) or a.max_seconds <= 0):
        p.error('--max-seconds must be positive and finite')
    options = load_options(a.config)
    dest = destination('teleop_panda')
    m, d = load_practice_model()
    robot = PandaInterface(m, d)
    control = TeleopController(robot, options)
    keys = KeyboardQueue()
    commands = StreamCSV(dest / 'interaction_commands.csv')
    states = StreamCSV(dest / 'physics_states.csv')
    events = StreamCSV(dest / 'events.csv')
    view = video = None
    close_hook = None
    viewer_closed = False
    video_result = None
    video_error = None
    count = 0
    status = 'initializing'
    error = None
    print(HELP, 'Output:', dest, flush=True)
    write_json(dest / 'run_config.json', dict(options=options, model_sha256=sha(ROOT / 'models/derived/teleop_practice_scene.xml'),
               source='native MuJoCo key_callback queue; actual ctrl + mj_step', owner_thread=threading.get_ident(),
               argv=vars(a), mujoco_version=mujoco.__version__, default_scene='no additional obstacles'))
    started = time.perf_counter()
    try:
        view = mujoco.viewer.launch_passive(m, d, key_callback=keys.callback, show_left_ui=False, show_right_ui=False)
        close_hook = QueuedWindowClose(keys, 'MuJoCo : Panda keyboard practice Phase 1.5')
        with view.lock():
            view.cam.lookat[:] = [.25, 0, .45]
            view.cam.distance = 1.8
            view.cam.azimuth = 135
            view.cam.elevation = -25
            # Number keys are also the viewer's geom-group shortcuts; W/R/F and
            # other letters affect rendering. Restore these visual-only options
            # so joint selection never makes the robot disappear.
            display = {name: getattr(view.opt, name).copy() for name in
                       ('geomgroup', 'sitegroup', 'jointgroup', 'tendongroup', 'actuatorgroup', 'flags')}
            render_flags = view.user_scn.flags.copy()
        if a.record:
            try:
                video = Recorder(robot, dest, 'teleop_actual_interaction', True, fps=options['record_fps'])
            except Exception as exc:
                video_error = repr(exc)
        next_render = next_status = next_log = 0.
        previous_alert = ''
        wall_origin = time.perf_counter()
        events.append(dict(event='startup', **control.state()))
        status = 'viewer_closed'
        while view.is_running():
            wall = time.perf_counter()
            if a.max_seconds is not None and wall - wall_origin >= a.max_seconds:
                status = 'wall_timeout'
                break
            with view.lock():
                batch = []
                for _ in range(128):
                    try:
                        batch.append(keys.commands.get_nowait())
                    except queue.Empty:
                        break
                # Stop/exit take priority over accumulated movement commands.
                priority = next((event for event in batch if event[0] in (32, 256, ord('Q'))), None)
                if priority is not None:
                    while True:
                        try:
                            batch.append(keys.commands.get_nowait())
                        except queue.Empty:
                            break
                    batch.remove(priority)
                    batch.insert(0, priority)
                for code, ns, tid in batch:
                    if priority is not None and (code, ns, tid) != priority:
                        row = dict(key=str(code), command_type='discarded_pending', accepted=False,
                                   reason='stop_or_exit_priority', received_ns=ns, callback_thread=tid,
                                   processing_thread=threading.get_ident(), before_q=control.robot.positions().tolist(),
                                   before_goal_q=control.goal.tolist(), before_actual_xyz_m=robot.tool_pose()[0].tolist(),
                                   before_target_xyz_m=control.state()['target_xyz_m'], **control.state())
                    else:
                        row = control.command(code, ns, tid)
                    commands.append(row)
                    count += 1
                if control.exit_requested:
                    status = 'viewer_closed' if close_hook.requests else 'keyboard_exit'
                    break
                step = control.step()
                for name, value in display.items():
                    getattr(view.opt, name)[:] = value
                marker = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'target_marker')
                d.mocap_pos[m.body_mocapid[marker]] = control.state()['target_xyz_m']
                if d.time >= next_log:
                    states.append(dict(**step, **{k: v for k, v in control.state().items() if k != 'time'}))
                    next_log = float(d.time) + 1 / options['state_log_hz'] - 1e-8
                if control.alert != previous_alert:
                    events.append(dict(event='motion_or_contact_transition', **control.state()))
                    previous_alert = control.alert
                if video is not None:
                    try:
                        if d.time <= options['record_seconds']:
                            video.capture(f'{control.mode} joint{control.selected+1} | keyboard intent')
                        else:
                            video_result = video.close()
                            video = None
                    except Exception as exc:
                        video_error = repr(exc)
                        try:
                            if video.mp4: video.mp4.close()
                            if video.renderer: video.renderer.close()
                        except Exception:
                            pass
                        video = None
                if wall >= next_status:
                    s = control.state()
                    print(status_text(s).replace('\n', ' | '), flush=True)
                    write_json(dest / 'live_status.json', s)
                    commands.flush(); states.flush(); events.flush()
                    next_status = wall + 1 / options['status_hz']
            # Terminal status is deliberate: avoid another render-thread text
            # handshake during native window teardown. Physics/render locks stay
            # short and the public passive-viewer loop owns synchronization.
            if not view.is_running():
                break
            if batch:
                # The viewer's W/R/F shortcuts also toggle scene render flags,
                # not only mjvOption. Force a documented user_scn flag change,
                # sync it, then restore defaults and sync again. This overrides
                # the built-in shortcut without patching MuJoCo/private APIs.
                with view.lock():
                    view.user_scn.flags[:] = 1 - render_flags
                before_q, before_time = d.qpos.copy(), float(d.time)
                view.sync()
                if not np.array_equal(before_q, d.qpos) or float(d.time) != before_time:
                    raise RuntimeError('Viewer UI edited physics during shortcut synchronization')
                with view.lock():
                    view.user_scn.flags[:] = render_flags
                next_render = 0
            if wall >= next_render:
                # sync is outside lock. Detect unauthorized GUI state edits; never restore qpos.
                before_q = d.qpos.copy()
                before_time = float(d.time)
                view.sync()
                if not np.array_equal(before_q, d.qpos) or float(d.time) != before_time:
                    raise RuntimeError('Viewer UI edited physics state; exiting rather than restoring qpos')
                next_render = wall + 1 / options['viewer_hz']
            # Never catch up with a burst of steps after slow rendering/IK.
            delay = float(m.opt.timestep) - (time.perf_counter() - wall)
            if delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        status = 'terminal_interrupt'
    except Exception as exc:
        status, error = 'error', repr(exc)
        events.append(dict(event=error, **control.state()))
    finally:
        if view is not None:
            view.close()
            viewer_closed = close_hook.wait_destroyed() if close_hook and close_hook.hwnd else not view.is_running()
        if video is not None:
            try:
                video_result = video.close()
            except Exception as exc:
                video_error = repr(exc)
        events.append(dict(event='shutdown:' + status, **control.state()))
        for stream in (commands, states, events):
            stream.close()
        summary = dict(exit_status=status, error=error, commands=count, physics_steps=control.servo.physics_steps,
                       simulation_seconds=float(d.time), wall_seconds=time.perf_counter() - started,
                       viewer_closed=viewer_closed, native_close_requests=close_hook.requests if close_hook else 0,
                       video=video_result, video_error=video_error,
                       final=control.state(), qpos_control_writes=0,
                       metrics=control.metrics,
                       scope='Local practice only; no orientation/path/collision safety guarantee')
        write_json(dest / 'complete.json', summary)
        print('FINISHED', dest, status, flush=True)
    if error:
        raise RuntimeError(error)


if __name__ == '__main__':
    main_guard(main)
