"""Construct a grazing collision to measure discrete edge sampling limitations."""
import numpy as np
from scripts.phase2a_cases import configured_checker, park, snapshot_scene, external_distance


def build_grazing_case():
    # Zero margin is intentionally a separate sensitivity case, not a change to
    # the default 5 mm planner configuration. Tiny grazing collisions can pass
    # coarse samples even with the same original robot geometry.
    c=configured_checker(shapes={'obstacle_sphere':[.003]},options={'minimum_clearance_m':0.})
    park(c);mid=c.home.copy();mid[0]=.005
    c.state_valid(mid)
    # Search deterministic surface placements; none changes robot meshes or q limits.
    attempts=[]
    fingers=[g for g in c.robot_geoms if 'finger' in c.name(g)]
    for g in fingers:
        c.state_valid(mid)
        centre=c.data.geom_xpos[g].copy()
        radial=centre.copy();radial[2]=0;radial/=np.linalg.norm(radial)
        for direction in (radial,np.array([0.,0.,1.]),np.array([0.,0.,-1.])):
            lo,hi=0.,.3
            for _ in range(45):
                off=(lo+hi)/2;c.set_obstacle_pose('obstacle_sphere',centre+direction*off)
                d=external_distance(c,c.state_valid(mid))
                if d < -.000010:lo=off
                else:hi=off
            c.set_obstacle_pose('obstacle_sphere',centre+direction*((lo+hi)/2))
            a,b=c.home.copy(),c.home.copy();a[0]=-.1;b[0]=.1
            coarse=c.edge_valid(a,b,step_rad=.02)
            fine=c.edge_valid(a,b,step_rad=.005)
            attempts.append(dict(anchor=c.name(g),direction=direction.tolist(),
                                 coarse_valid=coarse.valid,fine_valid=fine.valid,
                                 coarse_min=coarse.minimum_clearance,fine_min=fine.minimum_clearance))
            if coarse.valid and not fine.valid and coarse.start_valid and coarse.end_valid:
                return dict(name='resolution_grazing_zero_margin',q_start=a.tolist(),q_end=b.tolist(),
                            scene=snapshot_scene(c),minimum_clearance_m=0.,
                            shapes={'obstacle_sphere':[.003]},attempts=attempts,
                            construction='Measured tangency bisection to 10 micrometres penetration at joint1=.005rad; no robot geometry edits'),c
    return dict(name='resolution_grazing_not_found',attempts=attempts),None
