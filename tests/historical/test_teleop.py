"""Phase 1.5: real dynamics and rejected intent must not mutate physics state."""
import inspect
import queue
import threading
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from src.robot.teleop import KeyboardQueue, TeleopController, load_options, load_practice_model
from src.robot.panda_interface import PandaInterface


@pytest.fixture
def teleop(robot):
    return TeleopController(robot)


def settle(t, seconds=.7):
    for _ in range(round(seconds / t.robot.model.opt.timestep)):
        t.step()


@pytest.mark.parametrize('joint', range(7))
def test_each_joint_uses_servo_not_state_assignment(teleop, joint):
    t = teleop
    settle(t)
    initial = t.robot.positions()
    t.command(ord(str(joint + 1)))
    q, time = t.robot.data.qpos.copy(), float(t.robot.data.time)
    result = t.command(ord('K'))
    assert result['accepted']
    np.testing.assert_array_equal(t.robot.data.qpos, q)
    assert t.robot.data.time == time
    before = t.servo.command_q.copy()
    t.step()
    assert np.max(np.abs(t.servo.command_q - before)) <= .2 * .002 + 1e-12
    settle(t)
    assert t.robot.positions()[joint] - initial[joint] > .015


@pytest.mark.parametrize('key,axis,sign', [('W',0,1),('S',0,-1),('A',1,1),('D',1,-1),('R',2,1),('F',2,-1)])
def test_xyz_real_motion_all_six_directions(teleop, key, axis, sign):
    t = teleop
    settle(t)
    original = t.robot.tool_pose()[0]
    t.command(ord('M'))
    np.testing.assert_array_equal(t.xyz_goal, original)
    # First match the current actual point to remove the previous home servo's gravity offset.
    result = t.ik.solve(original, t.robot.positions())
    t.goal = result.q.copy()
    settle(t)
    baseline = t.robot.tool_pose()[0]
    before_target = t.xyz_goal.copy()
    result = t.command(ord(key))
    assert result['accepted'], result
    assert t.xyz_goal[axis] - before_target[axis] == pytest.approx(.005 * sign)
    settle(t)
    assert (t.robot.tool_pose()[0][axis] - baseline[axis]) * sign > .001


def test_callback_only_queue_different_thread(teleop):
    keys = KeyboardQueue()
    q = teleop.robot.data.qpos.copy()
    producer = threading.Thread(target=keys.callback, args=(ord('K'),))
    producer.start(); producer.join()
    np.testing.assert_array_equal(teleop.robot.data.qpos, q)
    event = keys.commands.get_nowait()
    row = teleop.command(*event)
    assert row['callback_thread'] != row['processing_thread']
    assert row['accepted']


def test_practice_limits_and_actual_mjcf_limits(teleop):
    t = teleop
    for _ in range(35):
        t.command(ord('K')); settle(t, .15)
    assert t.goal[0] == pytest.approx(t.envelope[0, 1])
    assert t.command(ord('K'))['accepted'] is False
    assert np.all(t.goal >= t.robot.limits[:, 0]) and np.all(t.goal <= t.robot.limits[:, 1])
    # H must also work from the outer practice envelope; it remains slew-limited.
    assert t.command(ord('H'))['accepted']
    settle(t, 2.5)
    assert abs(t.robot.positions()[0] - t.home[0]) < .01
    # Unit fixture exercises physical limit clipping without moving any live joint there.
    t.envelope = t.robot.limits.copy()
    t.goal[0] = t.robot.limits[0, 1]
    row = t.command(ord('K'))
    assert not row['accepted'] and 'mjcf_joint_limit' in row['limit']


def test_ik_failure_nonfinite_reject_without_changing_goal(teleop, monkeypatch):
    t = teleop
    t.command(ord('M'))
    q, xyz, live = t.goal.copy(), t.xyz_goal.copy(), t.robot.data.qpos.copy()
    failure = t.ik.solve([10, 10, 10], t.robot.positions())
    monkeypatch.setattr(t.ik, 'solve', lambda *a: failure)
    row = t.command(ord('W'))
    assert not row['accepted'] and t.last_ik == 'unreachable_bound'
    np.testing.assert_array_equal(t.goal, q)
    np.testing.assert_array_equal(t.xyz_goal, xyz)
    np.testing.assert_array_equal(t.robot.data.qpos, live)
    monkeypatch.setattr(t.ik, 'solve', lambda *a: replace(failure, success=True, q=np.full(7, np.nan)))
    assert not t.command(ord('W'))['accepted']
    np.testing.assert_array_equal(t.goal, q)


def test_soft_stop_home_and_modes(teleop):
    t = teleop
    t.command(ord('K')); settle(t, .03)
    q, command = t.robot.positions(), t.servo.command_q.copy()
    t.command(32)
    np.testing.assert_array_equal(t.goal, q)
    np.testing.assert_array_equal(t.servo.command_q, command)
    settle(t)
    row = t.command(ord('H'))
    assert row['accepted']
    np.testing.assert_array_equal(t.goal, t.home)
    assert not t.command(ord('W'))['accepted']
    t.command(256)
    assert t.exit_requested


def test_reject_large_ik_change_and_goal_backlog(teleop, monkeypatch):
    t = teleop
    for _ in range(15):
        t.command(ord('K'))
    assert abs(t.goal[0] - t.robot.positions()[0]) <= t.options['maximum_goal_lead_rad']
    t.command(ord('M'))
    result = t.ik.solve(t.robot.tool_pose()[0], t.robot.positions())
    monkeypatch.setattr(t.ik, 'solve', lambda *a: replace(result, q=t.home + .25))
    assert not t.command(ord('W'))['accepted']


def test_no_obstacles_separate_scene_and_config_bounds():
    m, d = load_practice_model()
    assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'obstacle_sphere') == -1
    assert m.nq == 9 and m.nu == 8
    o = load_options()
    assert o['joint_step_rad'] == .02 and o['cartesian_step_m'] == .005
    assert 'qpos[' not in inspect.getsource(TeleopController)


def test_contact_interlock_uses_actual_contacts(robot):
    from src.scene.obstacles import ObstacleScene
    t = TeleopController(robot)
    scene = ObstacleScene(robot)
    gid = next(g for g in scene.robot_ids if 'link4' in scene.name(g))
    scene.move('obstacle_sphere', robot.data.geom_xpos[gid].copy())
    t.step()
    assert t.interlocked and t.state()['external_contact_count'] > 0
    assert not t.command(ord('K'))['accepted']
