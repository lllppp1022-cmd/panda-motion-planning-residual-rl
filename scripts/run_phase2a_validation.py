"""Real cases, full discrete edge checks, sensitivity study and performance measurement."""
from _bootstrap import ROOT, destination, main_guard
from collections import Counter
import csv
import json
import platform
import time

import mujoco
import numpy as np

from src.common import write_json, write_csv, sha
from src.planning.collision_checker import CollisionChecker
from scripts.phase2a_cases import build_cases, configured_checker, park, restore_scene
from scripts.phase2a_resolution import build_grazing_case


def main():
    dest=destination('phase2a_validation')
    print('OUTPUT',dest,flush=True)
    cases=build_cases(dest)
    rows=[]
    for case in cases:
        r=case['result']
        rows.append(dict(case=case['name'],q=json.dumps(case['q']),valid=r['valid'],
                         joint_limit_valid=r['joint_limit_valid'],external_collision=r['external_collision'],
                         self_collision=r['self_collision'],minimum_clearance_m=r['minimum_clearance'],
                         minimum_is_lower_bound=r['minimum_is_lower_bound'],reason=r['rejection_reason'],
                         offending_geom_pairs=json.dumps(r['offending_geom_pairs']),
                         native_contacts=json.dumps(r['native_contacts']),elapsed_s=r['elapsed_s']))
    edge_rows=[];edge_full=[]
    def edge(name,c,a,b,step):
        r=c.edge_valid(a,b,step_rad=step)
        edge_full.append(dict(case=name,step_rad=step,options=dict(c.options),scene=__import__('scripts.phase2a_cases',fromlist=['snapshot_scene']).snapshot_scene(c),**r.to_dict()))
        edge_rows.append(dict(case=name,step_rad=step,valid=r.valid,start_valid=r.start_valid,end_valid=r.end_valid,
                              minimum_clearance_m=r.minimum_clearance,minimum_is_lower_bound=r.minimum_is_lower_bound,
                              first_invalid_index=r.first_invalid_index,first_invalid_alpha=r.first_invalid_alpha,
                              first_invalid_q=json.dumps(r.first_invalid_q),reason=r.rejection_reason,
                              state_calls=r.state_calls,distance_calls=r.distance_calls,
                              maximum_joint_step_rad=r.maximum_joint_step_rad,elapsed_s=r.elapsed_s,
                              offending_geom_pairs=json.dumps(r.offending_geom_pairs)))
        return r
    c=CollisionChecker();park(c);home=c.home.copy();target=home.copy();target[0]+=.2
    for step in (.2,.05,.02,.01,.005):
        assert edge('clear_edge',c,home,target,step).valid
    inside=next(k for k in cases if k['name']=='E_interior_collision')
    restore_scene(c,inside['scene'])
    for step in (.2,.05,.02,.01,.005):
        r=edge('endpoints_clear_interior_collision',c,inside['q_start'],inside['q_end'],step)
        assert not r.valid and r.start_valid and r.end_valid and 0<r.first_invalid_alpha<1
    edge('invalid_start',c,inside['q'],inside['q_end'],.02)
    edge('invalid_end',c,inside['q_start'],inside['q'],.02)
    grazing,gc=build_grazing_case()
    write_json(dest/'grazing_fixture.json',grazing)
    if gc is not None:
        for step in (.05,.02,.01,.005,.0025,.00125):
            edge('grazing_zero_margin',gc,grazing['q_start'],grazing['q_end'],step)
        guarded=configured_checker(shapes=grazing['shapes'])
        restore_scene(guarded,grazing['scene'])
        for step in (.02,.005):
            edge('same_grazing_default_5mm_margin',guarded,grazing['q_start'],grazing['q_end'],step)
    # Repeat the boundary under a stricter explicitly distinct clearance configuration.
    near=next(k for k in cases if k['name']=='D_clearance_boundary')
    strict=CollisionChecker(options={'minimum_clearance_m':.006})
    restore_scene(strict,near['scene']);r=strict.state_valid(near['q'])
    assert not r.valid and not r.external_collision and r.clearance_violation
    rows.append(dict(case='D2_6mm_threshold_rejects_5_5mm_gap',q=json.dumps(near['q']),valid=r.valid,
                     joint_limit_valid=r.joint_limit_valid,external_collision=r.external_collision,self_collision=r.self_collision,
                     minimum_clearance_m=r.minimum_clearance,minimum_is_lower_bound=r.minimum_is_lower_bound,
                     reason=r.rejection_reason,offending_geom_pairs=json.dumps(r.offending_geom_pairs),
                     native_contacts=json.dumps(r.native_contacts),elapsed_s=r.elapsed_s))
    # Fixed sampled configurations, same robot and native CCD, full pairs in every query.
    rng=np.random.default_rng(20261021)
    inputs=rng.uniform(c.robot.limits[:,0]+.001,c.robot.limits[:,1]-.001,size=(1000,7))
    write_csv(dest/'performance_inputs.csv',[dict(index=i,q=json.dumps(q.tolist())) for i,q in enumerate(inputs)])
    performance=[];raw=[]
    for extra in (0,3,9):
        loaded=time.perf_counter();bench=configured_checker(extra_obstacles=extra);load=time.perf_counter()-loaded
        park(bench)
        for q in inputs[:20]:bench.state_valid(q)
        samples=[];forward=[];distance=[];valid=0
        t=time.perf_counter()
        for i,q in enumerate(inputs):
            before=time.perf_counter();r=bench.state_valid(q);elapsed=time.perf_counter()-before
            if r.rejection_reason.startswith('query_error'):raise RuntimeError(r.rejection_reason)
            samples.append(elapsed);forward.append(r.forward_s);distance.append(r.distance_s);valid+=int(r.valid)
            raw.append(dict(extra_obstacles=extra,index=i,elapsed_s=elapsed,forward_s=r.forward_s,distance_s=r.distance_s,
                            distance_calls=r.distance_calls,valid=r.valid,minimum_clearance=r.minimum_clearance))
        total=time.perf_counter()-t
        performance.append(dict(robot_geoms=len(bench.robot_geoms),environment_geoms=len(bench.environment_geoms),
                                total_collision_geoms=len(bench.robot_geoms)+len(bench.environment_geoms),
                                audited_pairs=len(bench.pairs),pair_categories=dict(Counter(p['category'] for p in bench.pairs)),
                                queries=1000,warmup_queries=20,model_load_s=load,query_total_s=total,
                                median_ms=float(np.median(samples)*1000),p95_ms=float(np.percentile(samples,95)*1000),
                                p99_ms=float(np.percentile(samples,99)*1000),max_ms=float(np.max(samples)*1000),
                                forward_total_s=float(sum(forward)),distance_total_s=float(sum(distance)),valid_queries=valid))
        print('BENCH',performance[-1],flush=True)
    perf=dict(seed=20261021,python=platform.python_version(),platform=platform.platform(),mujoco=mujoco.__version__,
              timing='perf_counter wall-clock; no renderer; no cache shortcut; all audited pairs checked even for invalid states',
              cpu=platform.processor(),inputs_sha256=sha(dest/'performance_inputs.csv'),benchmarks=performance,
              model_scene_sha256=sha(ROOT/'models/derived/phase2a_scene.xml'),
              robot_xml_sha256=sha(ROOT/'models/derived/panda_phase1.xml'))
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            perf['cpu']=winreg.QueryValueEx(key,'ProcessorNameString')[0].strip()
    except (ImportError,OSError):
        perf['cpu_information_unavailable']=True
    perf['ccd_options']=dict(ccd_tolerance=float(c.model.opt.ccd_tolerance),ccd_iterations=int(c.model.opt.ccd_iterations),
                             disableflags=int(c.model.opt.disableflags))
    write_csv(dest/'phase2a_collision_cases.csv',rows)
    write_csv(dest/'phase2a_edge_validation.csv',edge_rows)
    write_csv(dest/'performance_raw.csv',raw)
    write_json(dest/'phase2a_performance.json',perf)
    write_json(dest/'edge_validation_full.json',edge_full)
    import shutil
    for name in ('phase2a_collision_cases.csv','phase2a_edge_validation.csv','phase2a_performance.json'):
        shutil.copy2(dest/name,ROOT/'results'/name)
    write_json(dest/'complete.json',dict(completed=True,cases=len(rows),edges=len(edge_rows),grazing_found=gc is not None))
    write_json(ROOT/'results/phase2a_latest.json',dict(run=str(dest)))
    print('COMPLETE',dest,flush=True)


if __name__=='__main__':main_guard(main)
