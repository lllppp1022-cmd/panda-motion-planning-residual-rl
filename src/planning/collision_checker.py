"""Discrete geometric checks on an owned model/data, never on live dynamics.

Distances use native CCD convex collision geoms. A capped distance is a lower
bound, not an exact separation. One checker per planning worker; no mj_step.
"""
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from types import MappingProxyType
import copy
import hashlib
import threading
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml

from src.common import ROOT
from src.robot.panda_interface import PandaInterface


def prepare_scene():
    source = ROOT / 'models/derived/phase1_scene.xml'
    path = source.with_name('phase2a_scene.xml')
    tree = ET.parse(source)
    tree.getroot().set('model', 'Panda Phase2A kinematic collision inspection')
    body = ET.SubElement(tree.getroot().find('worldbody'), 'body',
                         name='box_obstacle', mocap='true', pos='1.2 -0.8 0.5')
    ET.SubElement(body, 'geom', name='obstacle_box', type='box', size='0.06 0.08 0.08',
                  rgba='0.65 0.25 0.85 0.8', contype='1', conaffinity='1')
    content = ET.tostring(tree.getroot(), encoding='unicode')
    if not path.exists() or path.read_text(encoding='utf-8') != content:
        path.write_text(content, encoding='utf-8')
    return path


@dataclass
class StateResult:
    valid: bool = False
    joint_limit_valid: bool = False
    external_collision: bool = False
    self_collision: bool = False
    clearance_violation: bool = False
    minimum_clearance: float | None = None
    minimum_is_lower_bound: bool = False
    offending_geom_pairs: list = field(default_factory=list)
    rejection_reason: str = ''
    pair_results: list = field(default_factory=list)
    native_contacts: list = field(default_factory=list)
    distance_calls: int = 0
    elapsed_s: float = 0.
    forward_s: float = 0.
    distance_s: float = 0.

    def to_dict(self):
        return asdict(self)

    def __bool__(self):
        return self.valid


@dataclass
class EdgeResult:
    valid: bool = False
    start_valid: bool | None = None
    end_valid: bool | None = None
    minimum_clearance: float | None = None
    minimum_is_lower_bound: bool = False
    first_invalid_index: int | None = None
    first_invalid_alpha: float | None = None
    first_invalid_q: list | None = None
    offending_geom_pairs: list = field(default_factory=list)
    rejection_reason: str = ''
    state_calls: int = 0
    distance_calls: int = 0
    segments: int = 0
    maximum_joint_step_rad: float = 0.
    elapsed_s: float = 0.
    samples: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    def __bool__(self):
        return self.valid


class CollisionChecker:
    """Own a cloned MjModel and new MjData; caller's model/data remain untouched.

    All named robot group-3 geoms are included, including fingers/pads. The only
    floor exclusion is the fixed base mount. Non-adjacent self pairs remain.
    """
    def __init__(self, model=None, *, options=None):
        start = time.perf_counter()
        cfg = yaml.safe_load((ROOT / 'configs/planning_collision.yaml').read_text(encoding='utf-8'))
        if options:
            unknown = set(options) - set(cfg)
            if unknown:
                raise ValueError(f'Unknown collision options: {unknown}')
            cfg.update(options)
        for name in ('minimum_clearance_m', 'distance_query_max_m', 'numerical_tolerance_m',
                     'edge_step_rad', 'verification_step_rad', 'maximum_edge_segments', 'gripper_opening_m'):
            if not np.isfinite(cfg[name]):
                raise ValueError('Nonfinite collision configuration')
        if cfg['minimum_clearance_m'] < 0 or cfg['numerical_tolerance_m'] < 0:
            raise ValueError('Clearance/tolerance must be nonnegative')
        if cfg['distance_query_max_m'] <= cfg['minimum_clearance_m'] + cfg['numerical_tolerance_m']:
            raise ValueError('distmax must exceed clearance plus tolerance')
        if cfg['edge_step_rad'] <= 0 or cfg['verification_step_rad'] <= 0:
            raise ValueError('Positive edge resolution required')
        if int(cfg['maximum_edge_segments']) != cfg['maximum_edge_segments'] or cfg['maximum_edge_segments'] < 1:
            raise ValueError('Positive integer edge segment budget required')
        if not 0 <= cfg['gripper_opening_m'] <= .04:
            raise ValueError('Finger opening must be in [0,.04] m')
        self.options = MappingProxyType(copy.deepcopy(cfg))
        self.model = copy.copy(model) if model is not None else mujoco.MjModel.from_xml_path(str(prepare_scene()))
        self.data = mujoco.MjData(self.model)
        self.robot = PandaInterface(self.model, self.data)
        self.owner_thread = threading.get_ident()
        m = self.model
        if m.npair:
            raise ValueError('Explicit contact pairs require a new pair-filter audit')
        self.home_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, 'home')
        if self.home_id < 0:
            raise ValueError('Model requires an explicit home keyframe')
        mujoco.mj_resetDataKeyframe(m, self.data, self.home_id)
        self.home = self.robot.positions()
        self.mocap_pos = self.data.mocap_pos.copy()
        self.mocap_quat = self.data.mocap_quat.copy()
        finger_ids=np.array([mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in ('finger_joint1','finger_joint2')])
        if np.any(finger_ids<0) or np.any(m.jnt_type[finger_ids] != mujoco.mjtJoint.mjJNT_SLIDE):
            raise ValueError('Expected both named Panda slide finger joints')
        if not np.allclose(m.jnt_range[finger_ids],[[0,.04],[0,.04]]):
            raise ValueError('Unaudited finger joint ranges')
        self.fingers=m.jnt_qposadr[finger_ids].copy()
        self.planning_bounds = None
        if cfg['planning_joint_bounds_rad'] is not None:
            bounds = np.asarray(cfg['planning_joint_bounds_rad'], dtype=float)
            if bounds.shape != (7,2) or not np.isfinite(bounds).all() or np.any(bounds[:,0] > bounds[:,1]):
                raise ValueError('Planning bounds must be finite ordered (7,2)')
            if np.any(bounds[:,0] < self.robot.limits[:,0]) or np.any(bounds[:,1] > self.robot.limits[:,1]):
                raise ValueError('Additional planning bounds cannot expand MJCF limits')
            self.planning_bounds = bounds
        self.robot_geoms = [i for i in range(m.ngeom) if self.name(i).startswith('panda_') and m.geom_group[i] == 3]
        if len(self.robot_geoms) != 23:
            raise ValueError('Unaudited robot collision set; expected 23 Panda geoms')
        self.environment_geoms = [i for i in range(m.ngeom) if i not in self.robot_geoms and
                                  (self.name(i)=='floor' or self.name(i).startswith('obstacle_') or
                                   m.geom_contype[i] or m.geom_conaffinity[i])]
        if any(self.name(i) != 'floor' and not self.name(i).startswith('obstacle_') for i in self.environment_geoms):
            raise ValueError('Environment collision geometry needs an explicit audit')
        if 'floor' not in {self.name(i) for i in self.environment_geoms}:
            raise ValueError('Floor collision geometry is required')
        if any(m.geom_type[i] not in (mujoco.mjtGeom.mjGEOM_PLANE,mujoco.mjtGeom.mjGEOM_SPHERE,
                                     mujoco.mjtGeom.mjGEOM_CYLINDER,mujoco.mjtGeom.mjGEOM_BOX)
               for i in self.environment_geoms):
            raise ValueError('Unaudited environment shape')
        if any(m.body_weldid[m.geom_bodyid[i]] != 0 and m.body_mocapid[m.geom_bodyid[i]] < 0
               for i in self.environment_geoms):
            raise ValueError('Planning environment must be fixed or mocap, not dynamically articulated')
        self.pairs, self.excluded_pairs = self._build_pairs()
        self._pair_set = {tuple(sorted((r['id1'], r['id2']))) for r in self.pairs}
        self._model_arrays = ('geom_pos','geom_quat','geom_size','body_pos','body_quat','jnt_range',
                              'geom_contype','geom_conaffinity','geom_type','geom_bodyid','body_weldid','exclude_signature',
                              'body_mass','body_inertia','dof_armature','geom_rbound','geom_aabb','jnt_type','jnt_qposadr','jnt_dofadr')
        self._model_signature = self._signature()
        self.load_elapsed_s = time.perf_counter() - start

    def _signature(self):
        digest = hashlib.sha256()
        for name in self._model_arrays:
            digest.update(getattr(self.model,name).tobytes())
        return digest.digest()

    def name(self, gid):
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or f'geom#{gid}'

    def body_name(self, bid):
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(bid)) or 'world'

    def _build_pairs(self):
        m = self.model
        pairs, excluded = [], []
        native_exclusions = {(int(s) >> 16, int(s) & 65535) for s in m.exclude_signature}
        candidates = [(a,b,'self') for a,b in combinations(self.robot_geoms,2)]
        candidates += [(a,b,'external') for a in self.robot_geoms for b in self.environment_geoms]
        for a,b,kind in candidates:
            ba,bb = int(m.geom_bodyid[a]),int(m.geom_bodyid[b])
            wa,wb = int(m.body_weldid[ba]),int(m.body_weldid[bb])
            pa = int(m.body_weldid[m.body_parentid[wa]]) if wa else -1
            pb = int(m.body_weldid[m.body_parentid[wb]]) if wb else -1
            reason = ''
            if kind == 'self':
                if tuple(sorted((ba,bb))) in native_exclusions:
                    reason = 'official_contact_exclusion'
                elif ba == bb:
                    reason = 'same_body_compound_geometry'
                elif wa == wb:
                    reason = 'fixed_weld_same_rigid_body'
                elif (pa == wb and wb != 0) or (pb == wa and wa != 0):
                    reason = 'adjacent_welded_bodies_at_joint'
            elif self.body_name(ba) == 'link0' and self.name(b) == 'floor':
                reason = 'fixed_base_floor_mount_only'
            compatible = bool((m.geom_contype[a] & m.geom_conaffinity[b]) or
                              (m.geom_contype[b] & m.geom_conaffinity[a]))
            if not compatible:
                # Do not silently make the robot safe by disabling its masks.
                raise ValueError(f'Collision masks disable an audited collision pair {self.name(a)}, {self.name(b)}')
            row = dict(id1=a,id2=b,geom1=self.name(a),geom2=self.name(b),category=kind,
                       body1=self.body_name(ba),body2=self.body_name(bb))
            if reason:
                excluded.append(dict(**row, exclusion_reason=reason))
            else:
                pairs.append(row)
        return pairs, excluded

    def set_obstacle_pose(self, name, position, quaternion=(1,0,0,0)):
        """Static scene edit on this checker's independent scene only."""
        if threading.get_ident() != self.owner_thread:
            raise RuntimeError('Use a separate checker per worker thread')
        gid = mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,name)
        if gid not in self.environment_geoms or name == 'floor':
            raise ValueError('Expected a named mocap obstacle in this scene')
        p, quat = np.asarray(position,dtype=float),np.asarray(quaternion,dtype=float)
        if p.shape != (3,) or quat.shape != (4,) or not np.isfinite(p).all() or not np.isfinite(quat).all() or np.linalg.norm(quat) < 1e-12:
            raise ValueError('Finite position and nonzero finite wxyz quaternion required')
        mid = int(self.model.body_mocapid[self.model.geom_bodyid[gid]])
        if mid < 0:
            raise ValueError('Obstacle is not mocap')
        self.mocap_pos[mid] = p
        self.mocap_quat[mid] = quat / np.linalg.norm(quat)

    def _input(self, q):
        try:
            q = np.asarray(q,dtype=float)
        except (ValueError,TypeError):
            return None,'invalid_dimension_or_type'
        if q.shape != (7,):
            return None,'invalid_dimension'
        if not np.isfinite(q).all():
            return None,'nonfinite_input'
        if np.any(q < self.robot.limits[:,0]) or np.any(q > self.robot.limits[:,1]):
            return q,'joint_limit'
        if self.planning_bounds is not None and (np.any(q < self.planning_bounds[:,0]) or np.any(q > self.planning_bounds[:,1])):
            return q,'planning_bounds'
        return q.copy(),''

    def state_valid(self, q):
        start = time.perf_counter()
        result = StateResult()
        q, reason = self._input(q)
        if reason:
            result.joint_limit_valid = reason == 'planning_bounds'
            result.rejection_reason = reason
            result.elapsed_s = time.perf_counter() - start
            return result
        result.joint_limit_valid = True
        try:
            m,d = self.model,self.data
            if threading.get_ident() != self.owner_thread:
                raise ValueError('checker_thread_mismatch')
            if any(not np.isfinite(getattr(m,n)).all() for n in self._model_arrays):
                raise ValueError('nonfinite_model')
            if self._signature() != self._model_signature:
                raise ValueError('model_mutated_rebuild_checker')
            if not np.isfinite(m.opt.gravity).all() or not np.isfinite(m.opt.timestep) or m.opt.timestep <= 0:
                raise ValueError('invalid_model_options')
            if not np.isfinite(self.mocap_pos).all() or not np.isfinite(self.mocap_quat).all():
                raise ValueError('nonfinite_scene')
            if not np.allclose(np.linalg.norm(self.mocap_quat,axis=1),1,atol=1e-10):
                raise ValueError('invalid_scene_quaternion')
            flag = getattr(mujoco.mjtDisableBit,'mjDSBL_NATIVECCD',None)
            if flag is not None and int(m.opt.disableflags) & int(flag):
                raise ValueError('legacy_ccd_not_supported')
            t = time.perf_counter()
            mujoco.mj_resetDataKeyframe(m,d,self.home_id)
            d.qpos[self.robot.qpos_indices] = q
            d.qpos[self.fingers] = self.options['gripper_opening_m']
            d.qvel[:] = 0
            d.qacc_warmstart[:] = 0
            d.mocap_pos[:] = self.mocap_pos
            d.mocap_quat[:] = self.mocap_quat
            mujoco.mj_forward(m,d)
            result.forward_s = time.perf_counter() - t
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.qacc).all() or not np.isfinite(d.geom_xpos).all() or not np.isfinite(d.geom_xmat).all():
                raise ValueError('nonfinite_forward_state')
            distmax = self.options['distance_query_max_m']
            tolerance = self.options['numerical_tolerance_m']
            threshold = self.options['minimum_clearance_m'] + tolerance
            t = time.perf_counter()
            for pair in self.pairs:
                distance = float(mujoco.mj_geomDistance(m,d,pair['id1'],pair['id2'],distmax,None))
                result.distance_calls += 1
                if not np.isfinite(distance):
                    raise ValueError('nonfinite_geometry_distance')
                row = dict(**pair,distance_m=distance,is_lower_bound=distance >= distmax)
                result.pair_results.append(row)
                if distance <= threshold:
                    result.offending_geom_pairs.append(row)
                    result.clearance_violation = True
                    if distance <= tolerance:
                        if pair['category'] == 'self': result.self_collision = True
                        else: result.external_collision = True
            result.distance_s = time.perf_counter() - t
            minimum = min(result.pair_results,key=lambda r:r['distance_m'])
            result.minimum_clearance = minimum['distance_m']
            result.minimum_is_lower_bound = minimum['is_lower_bound']
            for contact in d.contact:
                key = tuple(sorted((int(contact.geom1),int(contact.geom2))))
                result.native_contacts.append(dict(geom1=self.name(contact.geom1),geom2=self.name(contact.geom2),
                                                  distance_m=float(contact.dist), checked_pair=key in self._pair_set))
            result.valid = not result.clearance_violation
            result.rejection_reason = ('self_collision' if result.self_collision else
                                       'external_collision' if result.external_collision else
                                       'minimum_clearance' if result.clearance_violation else '')
        except (ValueError,RuntimeError,mujoco.FatalError) as exc:
            result.valid = False
            result.rejection_reason = 'query_error:' + str(exc)
        result.elapsed_s = time.perf_counter() - start
        return result

    def edge_valid(self, q_start, q_end, *, step_rad=None):
        start = time.perf_counter()
        result = EdgeResult()
        step = self.options['edge_step_rad'] if step_rad is None else step_rad
        if not np.isscalar(step) or not np.isfinite(step) or step <= 0:
            result.rejection_reason = 'invalid_edge_resolution'
            result.elapsed_s = time.perf_counter() - start
            return result
        a, reason_a = self._input(q_start)
        b, reason_b = self._input(q_end)
        if a is None or b is None:
            result.rejection_reason = 'start:' + reason_a if a is None else 'end:' + reason_b
            result.elapsed_s = time.perf_counter() - start
            return result
        n = max(1,int(np.ceil(np.max(np.abs(b-a)) / step)))
        if n > self.options['maximum_edge_segments']:
            result.rejection_reason = 'sample_budget_exceeded'
            result.elapsed_s = time.perf_counter() - start
            return result
        result.segments = n
        result.maximum_joint_step_rad = float(np.max(np.abs(b-a))/n)
        for i in range(n+1):
            alpha = i/n
            q = a.copy() if i == 0 else b.copy() if i == n else a+(b-a)*alpha
            state = self.state_valid(q)
            result.state_calls += 1
            result.distance_calls += state.distance_calls
            result.samples.append(dict(index=i,alpha=alpha,q=q.tolist(),valid=state.valid,
                                       minimum_clearance=state.minimum_clearance,
                                       minimum_is_lower_bound=state.minimum_is_lower_bound,
                                       rejection_reason=state.rejection_reason,offending_geom_pairs=state.offending_geom_pairs))
            if i == 0: result.start_valid = state.valid
            if i == n: result.end_valid = state.valid
            if not state.valid and result.first_invalid_index is None:
                result.first_invalid_index = i
                result.first_invalid_alpha = alpha
                result.first_invalid_q = q.tolist()
                result.offending_geom_pairs = state.offending_geom_pairs
                result.rejection_reason = ('start:' if i == 0 else 'end:' if i == n else 'interior:') + state.rejection_reason
            if state.minimum_clearance is not None and (result.minimum_clearance is None or state.minimum_clearance < result.minimum_clearance):
                result.minimum_clearance = state.minimum_clearance
                result.minimum_is_lower_bound = state.minimum_is_lower_bound
        result.valid = result.first_invalid_index is None
        result.elapsed_s = time.perf_counter() - start
        return result


def state_valid(q, scene):
    """Planner-facing convenience API; scene is an owned CollisionChecker."""
    return scene.state_valid(q)


def edge_valid(q_start, q_end, scene, *, step_rad=None):
    return scene.edge_valid(q_start,q_end,step_rad=step_rad)
