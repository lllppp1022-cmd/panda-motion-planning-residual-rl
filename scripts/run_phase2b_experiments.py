"""Budgeted real RRT pilot/batch; every selected seed and every outcome is saved."""
from _bootstrap import ROOT,destination,main_guard
import argparse
from collections import Counter
import json
import platform
import shutil
import time

import numpy as np
import yaml

from src.common import write_json,write_csv,sha
from src.planning.rrt_connect import RRTConnectPlanner,PlannerConfig
from scripts.phase2b_common import load_scenes,scene_checker


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['pilot','batch'],required=True)
    a=p.parse_args();dest=destination('phase2b_'+a.stage);payload=load_scenes()
    seeds=payload['pilot_seeds'] if a.stage=='pilot' else payload['planning_seeds']
    config=PlannerConfig.from_file()
    shutil.copy2(ROOT/'configs/phase2b_scenes.yaml',dest/'scenes_snapshot.yaml')
    shutil.copy2(ROOT/'configs/rrt_connect.yaml',dest/'planner_snapshot.yaml')
    write_json(dest/'source_hashes.json',{p:sha(ROOT/p) for p in ['src/planning/rrt_connect.py','src/planning/collision_checker.py',
               'configs/planning_collision.yaml','scripts/run_phase2b_experiments.py']})
    results=[];validations=[];aggregate=[]
    for scene in payload['scenes']:
        c=scene_checker(scene)
        baseline_start=time.perf_counter();direct=c.edge_valid(scene['q_start'],scene['q_goal'],step_rad=.005)
        direct_time=time.perf_counter()-baseline_start
        write_json(dest/f"{scene['name']}_direct_fine.json",direct.to_dict())
        scene_results=[]
        for seed in seeds:
            result=RRTConnectPlanner(c,config,seed).plan(scene['q_start'],scene['q_goal'])
            record=dict(stage=a.stage,scene=scene['name'],seed=seed,scene_configuration=scene,
                        direct_fine_valid=direct.valid,direct_time_s=direct_time,result=result.to_dict())
            write_json(dest/f"{scene['name']}_seed{seed}.json",record)
            for name,path in [('raw',result.raw_path),('shortcut',result.smoothed_path)]:
                write_csv(dest/f"{scene['name']}_seed{seed}_{name}_path.csv",
                          [dict(node=i,**{f'q{j+1}':float(v) for j,v in enumerate(q)}) for i,q in enumerate(path)])
            row=dict(scene=scene['name'],seed=seed,success=result.success,failure_reason=result.failure_reason,
                     planning_time_s=result.planning_time,search_time_s=result.search_time,smoothing_time_s=result.smoothing_time,
                     fine_validation_time_s=result.validation_time,iterations=result.iterations,
                     nodes_start_tree=result.nodes_start_tree,nodes_goal_tree=result.nodes_goal_tree,
                     state_valid_calls=result.state_valid_calls,edge_valid_calls=result.edge_valid_calls,
                     geom_distance_calls=result.geom_distance_calls,collision_check_time_s=result.collision_check_time,
                     raw_nodes=result.raw_node_count,shortcut_nodes=result.smoothed_node_count,
                     raw_length_rad=result.raw_path_length,final_length_rad=result.path_length,
                     length_reduction_fraction=(1-result.path_length/result.raw_path_length) if result.success and result.raw_path_length else None,
                     minimum_sampled_clearance_m=result.minimum_sampled_clearance,
                     raw_fine_passed=result.raw_final_validation_passed,final_validation_passed=result.final_validation_passed,
                     direct_fine_valid=direct.valid,direct_time_s=direct_time,
                     record_path=str(dest/f"{scene['name']}_seed{seed}.json"))
            results.append(row);scene_results.append(row)
            for kind,check in [('raw',result.raw_validation),('shortcut',result.smoothed_validation)]:
                if not check:continue
                validations.append(dict(scene=scene['name'],seed=seed,path_kind=kind,step_rad=check['step_rad'],passed=check['passed'],
                                        samples=check['samples'],minimum_sampled_clearance_m=check['minimum_sampled_clearance'],
                                        minimum_is_lower_bound=check['minimum_is_lower_bound'],first_invalid=json.dumps(check['first_invalid'])))
            print(a.stage,scene['name'],seed,result.success,result.failure_reason,round(result.planning_time,3),
                  'nodes',result.nodes_start_tree,result.nodes_goal_tree,flush=True)
            write_csv(dest/'phase2b_planning_results.csv',results)
            write_csv(dest/'phase2b_path_validation.csv',validations)
        success=[r for r in scene_results if r['success']]
        aggregate.append(dict(scene=scene['name'],trials=len(seeds),successes=len(success),success_rate=len(success)/len(seeds),
                              direct_fine_valid=direct.valid,direct_fine_time_s=direct_time,
                              failure_reasons=dict(Counter(r['failure_reason'] for r in scene_results if not r['success'])),
                              confirmed_fine_detour=bool(success and not direct.valid),
                              solvability='confirmed_by_independently_fine_checked_path' if success else 'invalid_endpoint' if scene['name'].startswith('F_') else 'not_confirmed_by_this_run',
                              successful_planning_time_median_s=float(np.median([r['planning_time_s'] for r in success])) if success else None,
                              successful_planning_time_p95_s=float(np.percentile([r['planning_time_s'] for r in success],95)) if success else None,
                              raw_length_median_rad=float(np.median([r['raw_length_rad'] for r in success])) if success else None,
                              shortcut_length_median_rad=float(np.median([r['final_length_rad'] for r in success])) if success else None,
                              average_state_calls=float(np.mean([r['state_valid_calls'] for r in scene_results])),
                              average_edge_calls=float(np.mean([r['edge_valid_calls'] for r in scene_results])),
                              fine_rejections=sum(r['failure_reason'] in ('raw_fine_validation_failed','smoothed_fine_validation_failed') for r in scene_results),
                              fine_passed_paths=sum(r['final_validation_passed'] for r in scene_results)))
    summary=dict(stage=a.stage,seeds=seeds,config=__import__('dataclasses').asdict(config),scene_summaries=aggregate,
                 trials=len(results),successes=sum(r['success'] for r in results),platform=platform.platform(),python=platform.python_version(),
                 unit_convention='Nearest optionally range-normalized; extension/path length Euclidean radians; collision grid max-single-joint radians',
                 continuous_safety_guarantee=False,rrt_implementation='src/planning/rrt_connect.py; no external planner')
    write_json(dest/'phase2b_performance.json',summary)
    write_json(dest/'complete.json',dict(completed=True,trials=len(results),stage=a.stage))
    write_json(ROOT/f'results/phase2b_{a.stage}_latest.json',dict(run=str(dest)))
    if a.stage=='batch':
        for name in ['phase2b_planning_results.csv','phase2b_path_validation.csv','phase2b_performance.json']:
            shutil.copy2(dest/name,ROOT/'results'/name)
    print('COMPLETE',dest,flush=True)


if __name__=='__main__':main_guard(main)
