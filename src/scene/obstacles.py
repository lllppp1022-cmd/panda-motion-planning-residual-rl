"""Named collision geoms, signed surface distances and filtered external contacts."""
import mujoco
import numpy as np
class ObstacleScene:
    def __init__(self,robot):
        self.robot=robot;self.model=robot.model;self.data=robot.data
        self.obstacle_ids=[mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,n) for n in ['obstacle_sphere','obstacle_cylinder']]
        self.robot_ids=[i for i in range(self.model.ngeom) if (mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,i) or '').startswith('panda_') and self.model.geom_group[i]==3 and (self.model.geom_contype[i] or self.model.geom_conaffinity[i])]
        assert len(self.robot_ids)>7 and all(i>=0 for i in self.obstacle_ids)
    def name(self,g):return mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,int(g))
    def move(self,name,position):
        gid=mujoco.mj_name2id(self.model,mujoco.mjtObj.mjOBJ_GEOM,name);assert gid in self.obstacle_ids
        mid=int(self.model.body_mocapid[self.model.geom_bodyid[gid]]);assert mid>=0
        position=np.asarray(position,dtype=float)
        if position.shape!=(3,) or not np.isfinite(position).all():raise ValueError('Expected finite world xyz')
        self.data.mocap_pos[mid]=position;mujoco.mj_forward(self.model,self.data)
    def geometry(self):
        return [dict(name=self.name(g),id=g,type=int(self.model.geom_type[g]),position_world=self.data.geom_xpos[g].tolist(),size=self.model.geom_size[g].tolist(),rotation_world=self.data.geom_xmat[g].reshape(3,3).tolist(),contype=int(self.model.geom_contype[g]),conaffinity=int(self.model.geom_conaffinity[g])) for g in self.obstacle_ids]
    def query(self,distmax=2.):
        if not np.isfinite(distmax) or distmax<=0:raise ValueError('distmax must be positive')
        mujoco.mj_forward(self.model,self.data);pairs=[]
        for a in self.robot_ids:
            for b in self.obstacle_ids:
                enabled=bool((self.model.geom_contype[a]&self.model.geom_conaffinity[b]) or (self.model.geom_contype[b]&self.model.geom_conaffinity[a]))
                if not enabled:continue
                segment=np.zeros(6);distance=float(mujoco.mj_geomDistance(self.model,self.data,a,b,distmax,segment))
                pairs.append(dict(robot_geom=self.name(a),robot_geom_id=a,obstacle_geom=self.name(b),obstacle_geom_id=b,signed_distance_m=distance,distmax=distmax,capped=distance>=distmax,nearest_surface_segment=segment.tolist()))
        contacts=[]
        for k in range(self.data.ncon):
            c=self.data.contact[k];a,b=int(c.geom1),int(c.geom2)
            if (a in self.robot_ids and b in self.obstacle_ids) or (b in self.robot_ids and a in self.obstacle_ids):contacts.append(dict(geom1=self.name(a),geom2=self.name(b),distance_m=float(c.dist),position_world=c.pos.tolist()))
        closest=min(pairs,key=lambda r:r['signed_distance_m'])
        return dict(minimum=closest,external_contact_count=len(contacts),external_contacts=contacts,pairs=pairs,robot_collision_geoms=[self.name(g) for g in self.robot_ids],obstacles=self.geometry(),scope='Only named robot collision geoms vs two external obstacles. Not a planner, continuous collision guarantee or self-collision certificate.')
