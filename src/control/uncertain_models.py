"""Explicit nominal/physics separation. Never edits historical MJCF files."""
import copy,hashlib,json
import mujoco
import numpy as np
from src.robot.panda_interface import PandaInterface

PARAMETERS=('body_mass','body_inertia','body_ipos','body_iquat','dof_damping','dof_armature',
            'actuator_gainprm','actuator_biasprm','actuator_gear','actuator_forcerange',
            'actuator_ctrlrange','jnt_range','geom_size','geom_pos','geom_quat','geom_contype','geom_conaffinity')

def snapshot(model):
    result={n:getattr(model,n).tolist() for n in PARAMETERS}
    result.update(gravity=model.opt.gravity.tolist(),timestep=float(model.opt.timestep),integrator=int(model.opt.integrator))
    return result

def signature(model):
    return hashlib.sha256(json.dumps(snapshot(model),sort_keys=True,allow_nan=False).encode()).hexdigest()

def model_pair(nominal,perturbation):
    allowed={'name','kp_scale','joint_damping_scale','mass_inertia_scale','q_noise_std_rad','observation_delay_steps','action_delay_steps'}
    if set(perturbation)-allowed:raise ValueError('Unknown perturbation')
    n=copy.copy(nominal);p=copy.copy(nominal);d=mujoco.MjData(p);r=PandaInterface(p,d)
    for key in ('kp_scale','joint_damping_scale','mass_inertia_scale'):
        value=perturbation.get(key,1.)
        if not np.isfinite(value) or not .5<=value<=1.5:raise ValueError('Out-of-audit physical parameter factor')
    for key in ('observation_delay_steps','action_delay_steps'):
        value=perturbation.get(key,0)
        if not isinstance(value,int) or not 0<=value<=25:raise ValueError('Invalid bounded delay')
    noise=perturbation.get('q_noise_std_rad',0.)
    if not np.isfinite(noise) or not 0<=noise<=.005:raise ValueError('Invalid supervisory noise')
    scale=perturbation.get('kp_scale',1.)
    p.actuator_gainprm[r.actuator_ids,0]*=scale
    p.actuator_biasprm[r.actuator_ids,1]*=scale
    p.dof_damping[r.dof_indices]*=perturbation.get('joint_damping_scale',1.)
    # Scale every massive moving robot body coherently, retaining inertia triangles.
    bodies=np.flatnonzero((p.body_weldid!=0)&(p.body_mocapid<0)&(p.body_mass>0))
    scale=perturbation.get('mass_inertia_scale',1.)
    p.body_mass[bodies]*=scale;p.body_inertia[bodies]*=scale
    mujoco.mj_setConst(p,d)
    for key in PARAMETERS:
        if not np.isfinite(getattr(p,key)).all():raise ValueError('Invalid model arrays')
    if np.any(p.body_mass<0) or np.any(p.body_inertia<0):raise ValueError('Invalid inertial model')
    if np.any(2*np.max(p.body_inertia,axis=1)>np.sum(p.body_inertia,axis=1)+1e-10):raise ValueError('Inertia triangle inequality')
    for key in ('jnt_range','actuator_forcerange','actuator_ctrlrange','actuator_gear','geom_size','geom_pos','geom_quat','geom_contype','geom_conaffinity'):
        if not np.array_equal(getattr(n,key),getattr(p,key)):raise ValueError('Perturbation illegally changes geometry/limits')
    return n,p,dict(nominal_signature=signature(n),physics_signature=signature(p),
                    perturbation=perturbation,moving_body_ids=bodies.tolist(),nominal=snapshot(n),physics=snapshot(p))
