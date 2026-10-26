from _bootstrap import destination,main_guard
import argparse
from src.planning.collision_checker import CollisionChecker
from scripts.phase2a_visualization import latest_cases,render_demo


def main():
    p=argparse.ArgumentParser(description='KINEMATIC collision configurations, not dynamic execution')
    p.add_argument('--viewer',action='store_true');p.add_argument('--record',action='store_true')
    a=p.parse_args();dest=destination('phase2a_collision_demo')
    _,cases=latest_cases();frames=[]
    for case in cases:
        frames.extend([dict(q=case['q'],scene=case['scene'],label=case['name'])]*15)
    result=render_demo(CollisionChecker(),frames,dest,'phase2a_collision_demo',viewer=a.viewer,record=a.record)
    print('OUTPUT',dest,result,flush=True)
    if result['interrupted']:print('Interrupted: partial visualization, not a full acceptance run')


if __name__=='__main__':main_guard(main)
