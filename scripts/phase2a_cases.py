"""Deterministic real-geometry fixtures; construction is saved, never a validity oracle."""
from src.common import ROOT
import json
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from src.common import write_json
from src.planning.collision_checker import CollisionChecker, prepare_scene


def configured_checker(*, shapes=None, options=None, extra_obstacles=0):
    if shapes is None and not extra_obstacles:
        return CollisionChecker(options=options)
    root = ET.parse(prepare_scene())
    for name,size in (shapes or {}).items():
        g = root.getroot().find(f".//geom[@name='{name}']")
        if g is None: raise ValueError(name)
        g.set('size',' '.join(map(str,size)))
    world = root.getroot().find('worldbody')
    for i in range(extra_obstacles):
        b = ET.SubElement(world,'body',name=f'extra_obstacle_{i}',mocap='true',pos=f'{2+i*.15} 1 1')
        ET.SubElement(b,'geom',name=f'obstacle_extra_{i}',type=('sphere','cylinder','box')[i%3],
                      size=('0.03','0.03 0.07','0.04 0.03 0.05')[i%3],rgba='.6 .4 .2 1')
    path = ROOT / 'models/derived' / f'phase2a_fixture_{time.time_ns()}.xml'
    root.write(path,encoding='utf-8')
    return CollisionChecker(mujoco.MjModel.from_xml_path(str(path)),options=options)


def park(checker):
    for i,g in enumerate(checker.environment_geoms):
        name = checker.name(g)
        if name != 'floor': checker.set_obstacle_pose(name,[2+i*.2,1,1])


def snapshot_scene(c):
    return {c.name(g):dict(position=c.mocap_pos[int(c.model.body_mocapid[c.model.geom_bodyid[g]])].tolist(),
                           quaternion=c.mocap_quat[int(c.model.body_mocapid[c.model.geom_bodyid[g]])].tolist(),
                           size=c.model.geom_size[g].tolist(),type=int(c.model.geom_type[g]))
            for g in c.environment_geoms if c.name(g) != 'floor'}


def restore_scene(c, scene):
    for name,pose in scene.items(): c.set_obstacle_pose(name,pose['position'],pose['quaternion'])


def external_distance(c,r):
    return min(p['distance_m'] for p in r.pair_results if p['category']=='external' and p['geom2'] != 'floor')


def build_cases(dest):
    c = CollisionChecker()
    park(c)
    home = c.home.copy()
    cases = []
    def add(name,q,note):
        r=c.state_valid(q)
        cases.append(dict(name=name,q=np.asarray(q).tolist(),scene=snapshot_scene(c),
                          construction=note,result=r.to_dict()))
        return r
    add('A_clear_home',home,'Original home; obstacles parked away; floor and all self pairs remain checked')
    c.state_valid(home)
    link4 = next(g for g in c.robot_geoms if 'link4' in c.name(g))
    center = c.data.geom_xpos[link4].copy()
    for name,obstacle in [('B_sphere_collision','obstacle_sphere'),('C_cylinder_collision','obstacle_cylinder'),('B2_box_collision','obstacle_box')]:
        park(c);c.set_obstacle_pose(obstacle,center)
        r=add(name,home,'Place the compiled obstacle at the actual world centre of link4 collision geom')
        if not r.external_collision: raise AssertionError((name,r.to_dict()))
    # Boundary by measured surface distance, not centre distance or fabricated expected clearance.
    park(c)
    direction=np.array([1.,0.,0.])
    lo,hi=0.,1.
    target=.0055
    for _ in range(45):
        mid=(lo+hi)/2;c.set_obstacle_pose('obstacle_sphere',center+direction*mid)
        if external_distance(c,c.state_valid(home))<target:lo=mid
        else:hi=mid
    c.set_obstacle_pose('obstacle_sphere',center+direction*((lo+hi)/2))
    r=add('D_clearance_boundary',home,'45-step bisection of measured robot-sphere surface clearance to 5.5 mm')
    if not r.valid: raise AssertionError(r.to_dict())
    park(c);bad=home.copy();bad[3]=c.robot.limits[3,1]+.01
    add('F_joint_limit',bad,'joint4 exceeds real MJCF upper bound by .01 rad')
    # Real self collision search over hard limits; no modifications to robot geoms/masks.
    rng=np.random.default_rng(20261021)
    search=[];found=None;floor=None
    for i in range(1500):
        q=rng.uniform(c.robot.limits[:,0]+.001,c.robot.limits[:,1]-.001)
        r=c.state_valid(q)
        search.append(dict(index=i,q=q.tolist(),valid=r.valid,self_collision=r.self_collision,
                           external_collision=r.external_collision,minimum_clearance=r.minimum_clearance))
        if found is None and r.self_collision:
            found=q.copy();add('G_nonadjacent_self_collision',q,'Seed20261021 uniform true hard-limit search, first measured self collision')
        if floor is None and any(p['geom2']=='floor' and p['distance_m']<0 for p in r.offending_geom_pairs):
            floor=q.copy();add('H_floor_collision',q,'Same real configuration search, first measured moving-link floor penetration')
        if found is not None and floor is not None:break
    write_json(dest/'self_collision_search.json',search)
    if found is None: raise RuntimeError('No self collision found; report as missing rather than fabricate')
    if floor is None: raise RuntimeError('No floor collision found')
    park(c);c.state_valid(home)
    adjacent=[]
    for p in c.excluded_pairs:
        if p['exclusion_reason']=='adjacent_welded_bodies_at_joint':
            d=float(mujoco.mj_geomDistance(c.model,c.data,p['id1'],p['id2'],.2,None))
            adjacent.append(dict(**p,distance_m=d))
    adjacent.sort(key=lambda r:r['distance_m'])
    write_json(dest/'allowed_adjacent_pairs_measured.json',adjacent)
    add('G_allowed_adjacent_home',home,'Excluded adjacent joint geometry can overlap at home; checked nonadjacent pairs still clear')
    # Broad interior collision: finger collision geometry traverses a static sphere.
    park(c);c.state_valid(home)
    finger = next(g for g in c.robot_geoms if 'left_finger_fingertip_pad_collision_1' in c.name(g))
    c.set_obstacle_pose('obstacle_sphere',c.data.geom_xpos[finger].copy())
    a,b=home.copy(),home.copy();a[0]=-.5;b[0]=.5
    edge_scene=snapshot_scene(c)
    endpoints=[c.state_valid(a),c.state_valid(b)]
    if not all(r.valid for r in endpoints):raise AssertionError('Interior fixture endpoints must both be valid')
    if c.state_valid(home).valid:raise AssertionError('Interior fixture did not collide')
    cases.append(dict(name='E_interior_collision',q=home.tolist(),q_start=a.tolist(),q_end=b.tolist(),
                      scene=edge_scene,construction='Static sphere placed at real fingertip position along a joint1 sweep',
                      result=c.state_valid(home).to_dict()))
    write_json(dest/'collision_cases_full.json',cases)
    write_json(dest/'pair_audit.json',dict(included=c.pairs,excluded=c.excluded_pairs,
                  robot_geoms=[dict(id=g,name=c.name(g),body=c.body_name(c.model.geom_bodyid[g]),
                                    type=int(c.model.geom_type[g]),contype=int(c.model.geom_contype[g]),
                                    conaffinity=int(c.model.geom_conaffinity[g])) for g in c.robot_geoms],
                  official_exclude_signature=c.model.exclude_signature.tolist()))
    return cases
