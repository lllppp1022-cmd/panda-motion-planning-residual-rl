from _bootstrap import destination,main_guard
import json
from src.common import ROOT,sha,write_json
from src.control.validated_path import load_verified_path


def main():
    dest=destination('phase2c_freeze');records={}
    for scene in ['A_clear','B_sphere','C_cylinder','D_two_obstacles']:
        verified=load_verified_path(scene,42)
        write_json(dest/(scene+'_current_fine_audit.json'),verified.evidence)
        records[scene]=dict(path_csv=verified.evidence['path_csv'],path_sha256=verified.evidence['path_csv_sha256'],
                           path_nodes=len(verified.path),source=verified.evidence['source_json'],minimum_clearance_m=verified.evidence['planning_minimum_clearance_m'])
    files=['results/phase2b_planning_results.csv','results/phase2b_path_validation.csv','results/phase2b_performance.json',
        'results/phase2b_acceptance.json','assets/videos/phase2b_rrt_planning_demo.mp4','configs/trajectory_execution.yaml',
        'src/robot/teleop.py','src/robot/joint_controller.py','models/derived/panda_phase1.xml','models/derived/phase2a_scene.xml',
        'src/planning/collision_checker.py','src/planning/rrt_connect.py']
    write_json(dest/'frozen_inputs.json',dict(scenes=records,hashes={p:sha(ROOT/p) for p in files},
        no_replanning=True,thresholds_frozen_before_first_execution=True))
    write_json(ROOT/'results/phase2c_freeze_latest.json',dict(run=dest));print(dest,flush=True)


if __name__=='__main__':main_guard(main)
