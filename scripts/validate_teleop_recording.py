"""Audit saved real interaction; this script never steps or controls a robot."""
from _bootstrap import ROOT, main_guard
import argparse
import csv
import json
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.common import write_json, write_csv, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    complete = json.loads((run / 'complete.json').read_text(encoding='utf-8'))
    assert complete['error'] is None and complete['viewer_closed']
    with (run / 'interaction_commands.csv').open(encoding='utf-8-sig', newline='') as f:
        commands = list(csv.DictReader(f))
    with (run / 'physics_states.csv').open(encoding='utf-8-sig', newline='') as f:
        states = list(csv.DictReader(f))
    times = np.array([float(r['time']) for r in states])
    q = np.array([json.loads(r['q']) for r in states])
    xyz = np.array([json.loads(r['actual_xyz_m']) for r in states])
    goal_xyz = np.array([json.loads(r['target_xyz_m']) for r in states])
    sample = lambda t: int(np.argmin(abs(times - t)))
    measurements = []
    for c in commands:
        assert json.loads(c['before_q']) == json.loads(c['q']), 'Command teleported live qpos'
        if c['accepted'] != 'True' or c['command_type'] not in ('joint_jog', 'xyz_jog'):
            continue
        start, end = sample(float(c['time'])), sample(float(c['time']) + .65)
        row = dict(key=c['key'], command_type=c['command_type'], time=float(c['time']),
                   endpoint_time=float(times[end]), actual_delta_q_rad=(q[end] - np.array(json.loads(c['before_q']))).tolist(),
                   actual_delta_xyz_m=(xyz[end] - np.array(json.loads(c['before_actual_xyz_m']))).tolist())
        if c['command_type'] == 'joint_jog':
            joint = int(c['selected_joint']) - 1
            row['selected_joint'] = joint + 1
            row['correct_direction'] = row['actual_delta_q_rad'][joint] * (1 if c['key'] == 'K' else -1) > .005
        else:
            axis, sign = {'W': (0,1), 'S': (0,-1), 'A': (1,1), 'D': (1,-1), 'R': (2,1), 'F': (2,-1)}[c['key']]
            row['axis'] = 'XYZ'[axis]
            row['signed_axis_movement_mm'] = row['actual_delta_xyz_m'][axis] * sign * 1000
            row['correct_direction'] = row['signed_axis_movement_mm'] > 1
        measurements.append(row)
    assert all(r['correct_direction'] for r in measurements), measurements
    assert {r['selected_joint'] for r in measurements if r['command_type'] == 'joint_jog'} == set(range(1,8))
    assert {r['key'] for r in measurements if r['command_type'] == 'xyz_jog'} == set('WSADRF')
    assert all(c['callback_thread'] != c['processing_thread'] for c in commands)
    frames = json.loads((run / 'teleop_actual_interaction_frame_log.json').read_text(encoding='utf-8'))
    max_frame_state_error = 0.
    for frame in frames:
        i = sample(frame['time'])
        assert abs(times[i] - frame['time']) < 1e-8
        max_frame_state_error = max(max_frame_state_error, float(np.max(abs(q[i] - frame['q']))))
    assert max_frame_state_error < 1e-12
    assert complete['metrics']['max_ctrl_step_rad'] <= .2 * .002 + 1e-12
    assert complete['metrics']['alert_steps'] == 0
    assert abs(complete['physics_steps'] * .002 - complete['simulation_seconds']) < 1e-7
    write_csv(run / 'command_motion_validation.csv', measurements)
    manifest = {p.relative_to(ROOT).as_posix(): sha(p) for p in [
        ROOT / 'scripts/teleop_panda.py', ROOT / 'src/robot/teleop.py',
        ROOT / 'src/visualization/windows_close.py', ROOT / 'configs/teleop.yaml',
        run / 'interaction_commands.csv', run / 'physics_states.csv', run / 'complete.json']}
    result = dict(passed=True, run=str(run), commands=len(commands), joint_directions_verified=8,
                  cartesian_directions_verified=6, measurements=measurements,
                  physics_steps=complete['physics_steps'], simulation_seconds=complete['simulation_seconds'],
                  metrics=complete['metrics'], callback_processing_threads_distinct=True,
                  commands_do_not_write_qpos=True, frame_state_max_q_error_rad=max_frame_state_error,
                  video=complete['video'], source_and_data_sha256_audited_after_run=manifest,
                  external_contact_max=max(int(r['external_contact_count']) for r in states),
                  self_contact_max=max(int(r['self_contact_count']) for r in states))
    write_json(run / 'teleop_recording_validation.json', result)
    write_json(ROOT / 'results/teleop_recording_validation.json', result)
    fig, axes = plt.subplots(4, 2, figsize=(12, 10))
    for i, ax in enumerate(axes.flat):
        if i == 7:
            ax.axis('off'); continue
        ax.plot(times, q[:,i], label='actual')
        ax.plot(times, [float(r[f'target{i+1}']) for r in states], '--', label='target')
        ax.plot(times, [float(r[f'command{i+1}']) for r in states], ':', label='slew-limited ctrl')
        ax.set(xlabel='Simulation seconds', ylabel='rad', title=f'Joint {i+1}')
        ax.legend(fontsize=7); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(run / 'teleop_joint_tracking.png', dpi=140); plt.close(fig)
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for i in range(3):
        axes[i].plot(times, xyz[:,i], label='actual tool')
        axes[i].plot(times, goal_xyz[:,i], '--', label='position goal')
        axes[i].set(ylabel='World ' + 'XYZ'[i] + ' (m)'); axes[i].legend(); axes[i].grid(alpha=.2)
    axes[3].plot(times, np.linalg.norm(xyz - goal_xyz, axis=1)*1000)
    axes[3].set(xlabel='Simulation seconds', ylabel='Position error (mm)'); axes[3].grid(alpha=.2)
    fig.tight_layout(); fig.savefig(run / 'teleop_xyz_tracking.png', dpi=140); plt.close(fig)
    out = ROOT / 'assets/videos'
    for ext in ('gif', 'mp4'):
        source = run / f'teleop_actual_interaction.{ext}'
        target = out / f'panda_keyboard_teleop_20261021.{ext}'
        if target.exists() and sha(target) != sha(source):
            raise FileExistsError('Refuse replacing a previous published demonstration')
        shutil.copy2(source, target)
    print(json.dumps({k:v for k,v in result.items() if k not in ('measurements', 'video', 'source_and_data_sha256_audited_after_run')}, indent=2))


if __name__ == '__main__':
    main_guard(main)
