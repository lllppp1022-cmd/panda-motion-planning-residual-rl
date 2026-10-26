"""Fail-closed loading of frozen Phase2B CSV paths and historical model evidence."""
from dataclasses import dataclass
import csv,json,hashlib
from pathlib import Path
import numpy as np
import mujoco
from src.common import ROOT,sha
from src.planning.collision_checker import CollisionChecker
from scripts.phase2a_cases import restore_scene


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


@dataclass
class ValidatedPath:
    path:np.ndarray
    checker:CollisionChecker
    evidence:dict


def load_verified_path(scene_name,seed=42,run=None):
    run=Path(run or read(ROOT/'results/phase2b_batch_latest.json')['run'])
    # Historical hashes: derived XML was frozen before Phase2B; all upstream
    # mesh/source hashes were recorded at Phase1, not invented in this stage.
    evidence={}
    historical=read(ROOT/'results/phase2b_prior_snapshot.json')
    historical.update(read(run/'source_hashes.json'))
    required=['models/derived/panda_phase1.xml','models/derived/phase2a_scene.xml',
              'src/planning/collision_checker.py','configs/planning_collision.yaml']
    if any(p not in historical for p in required):raise ValueError('Missing historical model/filter hash')
    for name,h in historical.items():
        if sha(ROOT/name)!=h:raise ValueError('Historical source mismatch: '+name)
    provenance=read(ROOT/'results/model_provenance.json')
    if not provenance.get('original_files_sha256'):raise ValueError('Missing upstream asset hashes')
    upstream=ROOT/'models/mujoco_menagerie/franka_emika_panda'
    for name,h in provenance['original_files_sha256'].items():
        if sha(upstream/name)!=h:raise ValueError('Historical upstream mismatch: '+name)
    source=run/f'{scene_name}_seed{seed}.json';record=read(source);result=record['result']
    if not result['success'] or not result['final_validation_passed']:raise ValueError('Saved path was not validated successfully')
    def rows(p):
        with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    summary=next((r for r in rows(run/'phase2b_planning_results.csv') if r['scene']==scene_name and int(r['seed'])==seed),None)
    audits=[r for r in rows(run/'phase2b_path_validation.csv') if r['scene']==scene_name and int(r['seed'])==seed]
    if not summary or summary['success']!='True' or summary['final_validation_passed']!='True':raise ValueError('Planning CSV is not successful')
    if not any(r['path_kind']=='shortcut' and r['passed']=='True' and float(r['step_rad'])<=.005 for r in audits):
        raise ValueError('Missing saved fine path audit')
    path_file=run/f'{scene_name}_seed{seed}_shortcut_path.csv'
    data=rows(path_file);q=np.array([[float(r[f'q{i}']) for i in range(1,8)] for r in data])
    if q.ndim!=2 or q.shape[1]!=7 or not len(q) or not np.isfinite(q).all():raise ValueError('Invalid path CSV shape/values')
    if not np.array_equal(q,np.asarray(result['path'])):raise ValueError('Path CSV/JSON mismatch')
    scene=record['scene_configuration'];c=CollisionChecker();restore_scene(c,scene['scene'])
    for name,s in scene['scene'].items():
        g=mujoco.mj_name2id(c.model,mujoco.mjtObj.mjOBJ_GEOM,name)
        if g<0 or not np.array_equal(c.model.geom_size[g],s['size']) or int(c.model.geom_type[g])!=s['type']:
            raise ValueError('Scene geometry mismatch: '+name)
    if c.robot.names!=['joint'+str(i) for i in range(1,8)]:raise ValueError('Joint order mismatch')
    if not np.array_equal(q[0],scene['q_start']) or not np.array_equal(q[-1],scene['q_goal']):raise ValueError('Endpoint mismatch')
    nodes=[c.state_valid(x).to_dict() for x in q]
    edges=[c.edge_valid(a,b,step_rad=.005).to_dict() for a,b in zip(q,q[1:])]
    if not all(s['valid'] for s in nodes+edges):raise ValueError('Current independent .005 rad audit failed')
    q.setflags(write=False)
    evidence=dict(scene=scene_name,seed=seed,source_json=str(source),source_json_sha256=sha(source),
        path_csv=str(path_file),path_csv_sha256=sha(path_file),scene_configuration=scene,joint_names=c.robot.names,
        historical_hashes=historical,upstream_commit=provenance['commit'],upstream_hashes_verified=len(provenance['original_files_sha256']),
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'src/control/validated_path.py',ROOT/'configs/trajectory_execution.yaml']},
        current_nodes=nodes,current_edges=edges,planning_minimum_clearance_m=result['minimum_sampled_clearance'],verified=True,
        minimum_is_lower_bound=result['minimum_is_lower_bound'])
    return ValidatedPath(q,c,evidence)
