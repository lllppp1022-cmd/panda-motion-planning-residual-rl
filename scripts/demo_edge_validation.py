from _bootstrap import destination,main_guard
import argparse
import numpy as np
from src.common import write_json
from src.planning.collision_checker import CollisionChecker
from scripts.phase2a_cases import restore_scene
from scripts.phase2a_visualization import latest_cases,render_demo


def main():
    p=argparse.ArgumentParser(description='KINEMATIC linear edge inspection; not an executable servo trajectory')
    p.add_argument('--viewer',action='store_true');p.add_argument('--record',action='store_true')
    p.add_argument('--step-rad',type=float,default=.02)
    a=p.parse_args();dest=destination('phase2a_edge_demo')
    _,cases=latest_cases();case=next(c for c in cases if c['name']=='E_interior_collision')
    checker=CollisionChecker();restore_scene(checker,case['scene'])
    edge=checker.edge_valid(case['q_start'],case['q_end'],step_rad=a.step_rad)
    if not edge.samples:raise ValueError(edge.rejection_reason)
    first_contact=next((s['index'] for s in edge.samples if any(p['distance_m']<=checker.options['numerical_tolerance_m'] for p in s['offending_geom_pairs'])),None)
    frames=[]
    for s in edge.samples:
        label=f"alpha={s['alpha']:.3f} sample {s['index']}/{edge.segments}"
        if s['index']==edge.first_invalid_index:label+=' FIRST INVALID GAP'
        if s['index']==first_contact:label+=' FIRST CONTACT'
        frames.extend([dict(q=s['q'],scene=case['scene'],label=label)]*(10 if s['index'] in (edge.first_invalid_index,first_contact) else 1))
    write_json(dest/'edge_result.json',edge.to_dict())
    result=render_demo(checker,frames,dest,'phase2a_edge_demo',viewer=a.viewer,record=a.record)
    write_json(dest/'first_invalid_and_contact.json',dict(first_invalid_index=edge.first_invalid_index,
                first_invalid_q=edge.first_invalid_q,first_contact_index=first_contact,
                first_contact_q=edge.samples[first_contact]['q'] if first_contact is not None else None,
                source='Kinematic samples only; no dynamics'))
    print('OUTPUT',dest,result,flush=True)


if __name__=='__main__':main_guard(main)
