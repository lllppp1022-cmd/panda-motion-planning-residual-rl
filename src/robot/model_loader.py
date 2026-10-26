"""Derive a separate MJCF; upstream XML/assets remain untouched."""
from pathlib import Path
import xml.etree.ElementTree as ET
import subprocess
import mujoco
from src.common import ROOT,config,sha,write_json
def prepare_model():
    c=config();up=ROOT/c['upstream_directory'];dest=ROOT/'models/derived';dest.mkdir(parents=True,exist_ok=True)
    required=['panda.xml','scene.xml','LICENSE','README.md'];assert all((up/n).exists() for n in required)
    tree=ET.parse(up/'panda.xml');r=tree.getroot();r.set('model','panda_phase1_derived')
    r.find('compiler').set('meshdir','../mujoco_menagerie/franka_emika_panda/assets')
    r.find('option').set('timestep',str(c['timestep_s']))
    hand=r.find(".//body[@name='hand']");assert hand is not None
    ET.SubElement(hand,'site',dict(name=c['tool_site'],pos=' '.join(map(str,c['tool_site_position_in_hand_m'])),size='0.012',rgba='0.1 0.9 0.2 1',type='sphere'))
    for body in r.find('worldbody').iter('body'):
        for i,g in enumerate(body.findall('geom')):
            if not g.get('name'):g.set('name',f"panda_{body.get('name')}_{g.get('class','visual')}_{i}")
    robot=dest/'panda_phase1.xml';tree.write(robot,encoding='utf-8',xml_declaration=True)
    scene='''<mujoco model="Panda Phase1 scene">
 <include file="panda_phase1.xml"/>
 <visual><global offwidth="960" offheight="720"/><headlight diffuse="0.7 0.7 0.7" ambient="0.4 0.4 0.4"/></visual>
 <asset><texture name="floor_tex" type="2d" builtin="checker" rgb1="0.22 0.28 0.34" rgb2="0.16 0.21 0.26" width="256" height="256"/><material name="floor_mat" texture="floor_tex" texrepeat="3 3" reflectance="0.1"/></asset>
 <worldbody>
  <light pos="0 -1 2.5" dir="0 0 -1" directional="true"/>
  <geom name="floor" type="plane" size="2 2 0.05" material="floor_mat"/>
  <body name="target_marker" mocap="true" pos="0.45 0 0.55"><geom name="target_visual" type="sphere" size="0.018" rgba="0.1 0.8 0.2 0.55" contype="0" conaffinity="0" group="1"/></body>
  <body name="sphere_obstacle" mocap="true" pos="1.1 0.8 0.5"><geom name="obstacle_sphere" type="sphere" size="0.06" rgba="0.9 0.15 0.1 0.8" contype="1" conaffinity="1"/></body>
  <body name="cylinder_obstacle" mocap="true" pos="-0.6 0.7 0.35"><geom name="obstacle_cylinder" type="cylinder" size="0.065 0.18" rgba="0.95 0.65 0.05 0.8" contype="1" conaffinity="1"/></body>
 </worldbody>
</mujoco>
'''
    (dest/'phase1_scene.xml').write_text(scene,encoding='utf-8')
    upstream_repo=up.parent;commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=upstream_repo,text=True).strip()
    assets={p.relative_to(up).as_posix():sha(p) for p in up.rglob('*') if p.is_file()}
    write_json(ROOT/'results/model_provenance.json',dict(url='https://github.com/google-deepmind/mujoco_menagerie',commit=commit,license='Apache-2.0',original_files_sha256=assets,derived_files={p.name:sha(p) for p in dest.glob('*.xml')},changes=['Separate derived XML only; upstream not rewritten','Add hand-fixed tool_site at [0,0,.1034]m with identity local orientation','Name robot geoms for explicit collision reporting','Retain original actuator gains, gear, force limits, contact exclusions and implicitfast integrator','Separate scene floor, noncolliding target marker and two mocap obstacles; timestep.002']))
    return dest/'phase1_scene.xml'
def load_model():
    path=ROOT/config()['scene']
    if not path.exists():prepare_model()
    model=mujoco.MjModel.from_xml_path(str(path));data=mujoco.MjData(model)
    key=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_KEY,'home');assert key>=0
    # One-time initial condition, never used as a motion controller.
    mujoco.mj_resetDataKeyframe(model,data,key);mujoco.mj_forward(model,data)
    return model,data
