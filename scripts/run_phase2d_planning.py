from _bootstrap import destination,main_guard
import dataclasses,json
import yaml
from src.common import ROOT,write_json,write_csv,sha
from src.control.validated_path import load_verified_path
from src.planning.rrt_connect import RRTConnectPlanner,PlannerConfig
from src.planning.margin_study import MarginRequirements,checker_for


def main():
    dest=destination('phase2d_planning');base=load_verified_path('D_two_obstacles',42)
    cfg=yaml.safe_load((ROOT/'configs/phase2d_margin_study.yaml').read_text());config=PlannerConfig.from_file();rows=[]
    write_json(dest/'frozen_config.json',dict(study=cfg,rrt_config=dataclasses.asdict(config),study_sha256=sha(ROOT/'configs/phase2d_margin_study.yaml')))
    paths=['models/derived/panda_phase1.xml','models/derived/phase2a_scene.xml','configs/planning_collision.yaml',
           'src/planning/collision_checker.py','src/planning/rrt_connect.py','src/robot/joint_controller.py','configs/trajectory_execution.yaml']
    for margin in cfg['planning_clearances_m']:
        req=MarginRequirements(margin,cfg['execution_clearance_m']);c=checker_for(base.checker,margin)
        for seed in cfg['planning_seeds']:
            r=RRTConnectPlanner(c,config,seed).plan(base.path[0],base.path[-1]);stem=f'margin{margin*1000:02.0f}mm_seed{seed}'
            file=dest/(stem+'_shortcut_path.csv')
            write_csv(file,[dict(node=i,**{f'q{j+1}':v for j,v in enumerate(q)}) for i,q in enumerate(r.path)])
            write_csv(dest/(stem+'_raw_path.csv'),[dict(node=i,**{f'q{j+1}':v for j,v in enumerate(q)}) for i,q in enumerate(r.raw_path)])
            source=dest/(stem+'.json')
            write_json(source,dict(planning_clearance_m=margin,execution_clearance_m=req.execution_clearance_m,planning_seed=seed,
                scene_configuration=base.evidence['scene_configuration'],frozen_model_filter_hashes={p:sha(ROOT/p) for p in paths},
                shortcut_path_sha256=sha(file),result=r.to_dict()))
            rows.append(dict(planning_clearance_m=margin,execution_clearance_m=req.execution_clearance_m,planning_seed=seed,
                success=r.success,failure_reason=r.failure_reason,raw_fine_passed=r.raw_final_validation_passed,final_validation_passed=r.final_validation_passed,
                planning_time_s=r.planning_time,path_length_rad=r.path_length,raw_length_rad=r.raw_path_length,
                raw_nodes=r.raw_node_count,final_nodes=r.smoothed_node_count,minimum_sampled_clearance_m=r.minimum_sampled_clearance,
                minimum_is_lower_bound=r.minimum_is_lower_bound,iterations=r.iterations,nodes_start_tree=r.nodes_start_tree,nodes_goal_tree=r.nodes_goal_tree,
                state_calls=r.state_valid_calls,edge_calls=r.edge_valid_calls,source_json=str(source)))
            write_csv(dest/'phase2d_planning_margin_results.csv',rows)
            print('PLAN',margin,seed,r.success,r.failure_reason,round(r.planning_time,3),'min',r.minimum_sampled_clearance,flush=True)
    write_json(dest/'complete.json',dict(results=rows))
    __import__('shutil').copy2(dest/'phase2d_planning_margin_results.csv',ROOT/'results/phase2d_planning_margin_results.csv')
    write_json(ROOT/'results/phase2d_planning_latest.json',dict(run=dest));print('COMPLETE',dest,flush=True)


if __name__=='__main__':main_guard(main)
