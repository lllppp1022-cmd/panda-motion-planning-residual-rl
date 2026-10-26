"""Queued keyboard intent, position-only IK, and bounded real-servo targets.

This module never writes the live qpos/qvel. It is not a collision-safe planner.
"""
import queue
import threading
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml

from src.common import ROOT
from src.robot.panda_interface import PandaInterface
from src.robot.joint_controller import JointPositionController
from src.robot.inverse_kinematics import DampedLeastSquaresIK


HELP = """Panda Phase 1.5 | focus this MuJoCo window; use English keyboard input
M: JOINT <-> XYZ (enter XYZ starts at the actual tool position)
1..7: select joint | J/K: -/+ joint angle (JOINT mode)
W/S: world X +/- | A/D: world Y +/- | R/F: world Z +/- (XYZ mode)
Space: soft stop (servo holds current q; not an emergency brake)
H: slew-limited home goal | Esc or window X: exit | Q: exit fallback
Small home-centred practice envelope; no collision-safe path guarantee.
Viewer visual shortcuts are restored to the practice display; passive physics keeps running.
Do not use joint sliders, Backspace/reset, or Ctrl-drag perturbations.
"""


def load_options(path=None):
    o = yaml.safe_load((ROOT / 'configs/teleop.yaml' if path is None else path).read_text(encoding='utf-8'))
    for k, v in o.items():
        if not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0:
            raise ValueError(f'{k} must be positive and finite')
    if not o['minimum_joint_step_rad'] <= o['joint_step_rad'] <= .04:
        raise ValueError('joint_step_rad must be between configured minimum and .04')
    if not o['minimum_cartesian_step_m'] <= o['cartesian_step_m'] <= .01:
        raise ValueError('cartesian_step_m must be between configured minimum and .01')
    if o['max_command_velocity_rad_s'] > .35 or o['practice_radius_rad'] > .35:
        raise ValueError('Practice velocity/radius exceeds Phase 1.5 ceiling')
    if max(o['maximum_goal_lead_rad'], o['maximum_ik_joint_delta_rad']) > .12:
        raise ValueError('Motion lead ceiling is .12 rad')
    return o


def load_practice_model():
    """Separate scene, no added obstacles; retain floor and original robot physics."""
    source = ROOT / 'models/derived/phase1_scene.xml'
    dest = source.with_name('teleop_practice_scene.xml')
    tree = ET.parse(source)
    world = tree.getroot().find('worldbody')
    for b in list(world.findall('body')):
        if b.get('name') in ('sphere_obstacle', 'cylinder_obstacle'):
            world.remove(b)
    tree.getroot().set('model', 'Panda keyboard practice Phase 1.5')
    text = ET.tostring(tree.getroot(), encoding='unicode')
    if not dest.exists() or dest.read_text(encoding='utf-8') != text:
        dest.write_text(text, encoding='utf-8')
    m = mujoco.MjModel.from_xml_path(str(dest))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, 'home'))
    mujoco.mj_forward(m, d)
    return m, d


class KeyboardQueue:
    """The viewer callback only submits an immutable event to a thread-safe queue."""
    def __init__(self):
        self.commands = queue.SimpleQueue()

    def callback(self, keycode):
        self.commands.put((int(keycode), time.perf_counter_ns(), threading.get_ident()))


def contact_status(robot):
    m, d = robot.model, robot.data
    robot_bodies = {i for i in range(m.nbody)
                    if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or '').startswith(('link', 'hand', 'left_finger', 'right_finger'))}
    external, self_contacts = [], []
    for c in d.contact:
        a, b = int(c.geom1), int(c.geom2)
        ba, bb = int(m.geom_bodyid[a]), int(m.geom_bodyid[b])
        row = dict(geom1=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, a),
                   geom2=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, b), distance_m=float(c.dist))
        if ba in robot_bodies and bb in robot_bodies:
            self_contacts.append(row)
        elif (ba in robot_bodies) != (bb in robot_bodies):
            external.append(row)
    return dict(external_contacts=external, external_contact_count=len(external),
                self_contacts=self_contacts, self_contact_count=len(self_contacts))


class TeleopController:
    def __init__(self, robot, options=None):
        self.robot = robot
        self.options = load_options() if options is None else options.copy()
        self.servo = JointPositionController(robot, self.options['max_command_velocity_rad_s'])
        self.ik = DampedLeastSquaresIK(robot.model)
        self.home = robot.positions()
        self.goal = self.home.copy()
        self.mode = 'JOINT'
        self.selected = 0
        self.xyz_goal = None
        self.last_ik = 'not_requested'
        self.limit = ''
        self.exit_requested = False
        self.alert = ''
        self.interlocked = False
        self.owner_thread = threading.get_ident()
        self.metrics = dict(max_actual_joint_step_rad=0., max_actual_velocity_rad_s=0.,
                            max_ctrl_step_rad=0., alert_steps=0)
        self.envelope = np.column_stack((np.maximum(robot.limits[:, 0], self.home - self.options['practice_radius_rad']),
                                         np.minimum(robot.limits[:, 1], self.home + self.options['practice_radius_rad'])))

    def state(self):
        x, _ = self.robot.tool_pose()
        target = self.ik.kin.forward(self.goal)[0] if self.xyz_goal is None else self.xyz_goal.copy()
        return dict(time=float(self.robot.data.time), mode=self.mode, selected_joint=self.selected + 1,
                    q=self.robot.positions().tolist(), goal_q=self.goal.tolist(),
                    servo_command_q=self.servo.command_q.tolist(), actual_xyz_m=x.tolist(),
                    target_xyz_m=target.tolist(), error_mm=float(np.linalg.norm(target - x) * 1000),
                    ik_status=self.last_ik, limit=self.limit, alert=self.alert,
                    interlocked=self.interlocked, **contact_status(self.robot))

    def _accept(self, q, home_return=False):
        q = self.robot.validate(q)
        if np.any(q < self.envelope[:, 0]) or np.any(q > self.envelope[:, 1]):
            return False, 'practice_envelope'
        lead = self.options['practice_radius_rad'] + .02 if home_return else self.options['maximum_goal_lead_rad']
        if np.max(np.abs(q - self.robot.positions())) > lead:
            return False, 'goal_lead_limit_wait_for_servo'
        return True, ''

    def command(self, keycode, received_ns=0, callback_thread=0):
        assert threading.get_ident() == self.owner_thread
        before = self.state()
        key = 'SPACE' if keycode == 32 else 'ESC' if keycode == 256 else chr(keycode).upper() if 0 <= keycode < 128 else str(keycode)
        kind, accepted, reason = 'unknown', False, 'unmapped_key'
        self.limit = ''
        try:
            if key in ('ESC', 'Q'):
                kind, accepted, reason = 'exit', True, ''
                self.exit_requested = True
            elif key == 'SPACE':
                self.goal = self.robot.validate(self.robot.positions())
                self.xyz_goal = None
                self.interlocked = False
                self.alert = ''
                kind, accepted, reason = 'soft_stop', True, ''
            elif key == 'M':
                self.mode = 'XYZ' if self.mode == 'JOINT' else 'JOINT'
                # Discard any pending distant target on mode entry; do not teleport.
                self.goal = self.robot.validate(self.robot.positions())
                self.xyz_goal = self.robot.tool_pose()[0].copy() if self.mode == 'XYZ' else None
                kind, accepted, reason = 'mode', True, ''
            elif key in '1234567' and len(key) == 1:
                self.selected = int(key) - 1
                kind, accepted, reason = 'select_joint', True, ''
            elif self.interlocked:
                kind, reason = 'blocked', 'contact_or_motion_interlock_press_space_after_inspection'
            elif key == 'H':
                kind = 'home'
                accepted, reason = self._accept(self.home, home_return=True)
                if accepted:
                    self.goal = self.home.copy()
                    self.xyz_goal = None
            elif key in ('J', 'K'):
                kind = 'joint_jog'
                reason = 'wrong_mode'
                if self.mode == 'JOINT':
                    q = self.goal.copy()
                    proposed = q[self.selected] + (1 if key == 'K' else -1) * self.options['joint_step_rad']
                    q[self.selected] = np.clip(proposed, *self.robot.limits[self.selected])
                    if q[self.selected] != proposed:
                        self.limit = 'mjcf_joint_limit'
                    clipped = np.clip(q, self.envelope[:, 0], self.envelope[:, 1])
                    if not np.array_equal(q, clipped):
                        self.limit = (self.limit + ';practice_envelope').strip(';')
                    q = clipped
                    accepted, reason = self._accept(q)
                    if np.max(np.abs(q - self.goal)) < self.options['minimum_joint_step_rad']:
                        accepted, reason = False, 'below_minimum_or_at_limit'
                    if accepted:
                        self.goal = q
                        self.xyz_goal = None
            elif key in ('W', 'S', 'A', 'D', 'R', 'F'):
                kind, reason = 'xyz_jog', 'wrong_mode'
                if self.mode == 'XYZ':
                    origin = self.robot.tool_pose()[0].copy() if self.xyz_goal is None else self.xyz_goal.copy()
                    axis, sign = {'W': (0, 1), 'S': (0, -1), 'A': (1, 1), 'D': (1, -1), 'R': (2, 1), 'F': (2, -1)}[key]
                    candidate = origin.copy()
                    candidate[axis] += sign * self.options['cartesian_step_m']
                    result = self.ik.solve(candidate, self.robot.positions())
                    self.last_ik = result.status
                    reason = 'ik_' + result.status
                    if result.success:
                        accepted, reason = self._accept(result.q)
                        if np.max(np.abs(result.q - self.robot.positions())) > self.options['maximum_ik_joint_delta_rad']:
                            accepted, reason = False, 'ik_joint_delta_limit'
                        if accepted:
                            self.goal = result.q.copy()
                            self.xyz_goal = candidate
            if reason in ('practice_envelope', 'ik_joint_delta_limit', 'goal_lead_limit_wait_for_servo'):
                self.limit = reason
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            accepted, reason = False, 'invalid_or_nonfinite:' + str(exc)
        return dict(key=key, command_type=kind, accepted=accepted, reason=reason,
                    received_ns=received_ns, callback_thread=callback_thread,
                    processing_thread=threading.get_ident(), before_q=before['q'],
                    before_goal_q=before['goal_q'], before_actual_xyz_m=before['actual_xyz_m'],
                    before_target_xyz_m=before['target_xyz_m'], **self.state())

    def step(self):
        prior = self.robot.positions()
        prior_ctrl = self.servo.command_q.copy()
        row = self.servo.step(self.goal)
        c = contact_status(self.robot)
        alerts = []
        self.metrics['max_actual_joint_step_rad'] = max(self.metrics['max_actual_joint_step_rad'], float(np.max(np.abs(self.robot.positions() - prior))))
        self.metrics['max_actual_velocity_rad_s'] = max(self.metrics['max_actual_velocity_rad_s'], float(np.max(np.abs(self.robot.velocities()))))
        self.metrics['max_ctrl_step_rad'] = max(self.metrics['max_ctrl_step_rad'], float(np.max(np.abs(self.servo.command_q - prior_ctrl))))
        if c['external_contact_count'] or c['self_contact_count']:
            alerts.append('actual_contact')
        if np.max(np.abs(self.robot.positions() - prior)) > self.options['unexpected_step_rad']:
            alerts.append('unexpected_joint_step')
        if np.max(np.abs(self.robot.velocities())) > self.options['unexpected_velocity_rad_s']:
            alerts.append('unexpected_velocity')
        if np.max(np.abs(self.robot.positions() - self.servo.command_q)) > self.options['unexpected_tracking_error_rad']:
            alerts.append('unexpected_tracking_error')
        if alerts:
            self.metrics['alert_steps'] += 1
            self.goal = self.robot.validate(self.robot.positions())
            self.xyz_goal = None
            self.interlocked = True
            self.alert = ';'.join(alerts)
        return row
