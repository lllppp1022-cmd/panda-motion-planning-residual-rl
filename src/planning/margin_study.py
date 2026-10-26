"""Independent planning clearance, fixed execution clearance, unchanged old APIs."""
from dataclasses import dataclass
import csv,json
from pathlib import Path
import numpy as np
from src.common import ROOT,sha
from src.planning.collision_checker import CollisionChecker
from src.control.validated_path import ValidatedPath,load_verified_path
from scripts.phase2a_cases import restore_scene,snapshot_scene


@dataclass(frozen=True)
class MarginRequirements:
    planning_clearance_m:float=.005
    execution_clearance_m:float=.005

    def __post_init__(self):
        if not np.isfinite(self.planning_clearance_m) or self.planning_clearance_m<.005 or self.planning_clearance_m>=.1:
            raise ValueError('Planning margin must be finite, >=5mm, below distmax')
        if self.execution_clearance_m!=.005:raise ValueError('Execution threshold is frozen at 5mm')


def checker_for(base,margin):
    c=CollisionChecker(base.model,options={**dict(base.options),'minimum_clearance_m':margin})
    c.mocap_pos[:]=base.mocap_pos;c.mocap_quat[:]=base.mocap_quat
    return c


def load_margin_path(source):
    source=Path(source);record=json.loads(source.read_text(encoding='utf-8'));r=record['result']
    requirements=MarginRequirements(record['planning_clearance_m'],record['execution_clearance_m'])
    if not r['success'] or not r['final_validation_passed'] or not r['raw_final_validation_passed']:
        raise ValueError('Rejected or incompletely audited planning path')
    historical=load_verified_path('D_two_obstacles',42)
    if record['scene_configuration']!=historical.evidence['scene_configuration']:raise ValueError('Frozen D scene changed')
    for p,h in record['frozen_model_filter_hashes'].items():
        if sha(ROOT/p)!=h:raise ValueError('Model/filter mismatch')
    path_file=source.with_name(source.stem+'_shortcut_path.csv')
    if sha(path_file)!=record['shortcut_path_sha256']:raise ValueError('Path CSV hash mismatch')
    with path_file.open(encoding='utf-8-sig',newline='') as f:
        q=np.asarray([[float(row[f'q{i}']) for i in range(1,8)] for row in csv.DictReader(f)])
    if q.ndim!=2 or q.shape[1]!=7 or not len(q) or not np.isfinite(q).all() or not np.array_equal(q,r['path']):
        raise ValueError('Invalid or changed path CSV')
    scene=record['scene_configuration']
    if not np.array_equal(q[0],scene['q_start']) or not np.array_equal(q[-1],scene['q_goal']):raise ValueError('Wrong endpoints')
    planning=checker_for(historical.checker,requirements.planning_clearance_m)
    nodes=[planning.state_valid(x).to_dict() for x in q]
    edges=[planning.edge_valid(a,b,step_rad=.005).to_dict() for a,b in zip(q,q[1:])]
    if not all(s['valid'] for s in nodes+edges):raise ValueError('Current planning-margin fine audit failed')
    # A separate instance with the ORIGINAL threshold is handed to the ORIGINAL executor.
    execution=checker_for(historical.checker,requirements.execution_clearance_m)
    if not np.array_equal(planning.mocap_pos,execution.mocap_pos):raise ValueError('Mismatched scene')
    q.setflags(write=False)
    e=dict(scene='D_two_obstacles',seed=record['planning_seed'],planning_seed=record['planning_seed'],historical_scene_seed=42,
       planning_clearance_m=requirements.planning_clearance_m,execution_clearance_m=requirements.execution_clearance_m,
       source_json=str(source),source_json_sha256=sha(source),path_csv=str(path_file),path_csv_sha256=sha(path_file),
       scene_configuration=scene,verified=True,planning_minimum_clearance_m=r['minimum_sampled_clearance'],
       minimum_is_lower_bound=r['minimum_is_lower_bound'],current_nodes=nodes,current_edges=edges,
       frozen_model_filter_hashes=record['frozen_model_filter_hashes'],joint_names=historical.evidence['joint_names'])
    return ValidatedPath(q,execution,e)
