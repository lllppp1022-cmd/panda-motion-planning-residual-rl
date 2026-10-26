"""Owned implementation of bidirectional RRT-Connect and checked shortcuts.

No external planner, IK, dynamics, or live-state writes. Search uses the full
Phase2A checker. Raw and shortcut paths must pass a fresh-model fine audit.
"""
from dataclasses import asdict,dataclass,field
from pathlib import Path
import time

import numpy as np
import yaml

from src.common import ROOT
from src.planning.collision_checker import CollisionChecker


@dataclass(frozen=True)
class PlannerConfig:
    max_extension_rad:float=.35
    normalized_nearest:bool=True
    goal_bias:float=.10
    max_iterations:int=800
    max_connect_steps:int=128
    max_planning_time_s:float=12.
    max_search_time_s:float=6.
    search_step_rad:float=.02
    final_step_rad:float=.005
    shortcut_attempts:int=40
    shortcut_time_s:float=2.

    @classmethod
    def from_file(cls,path=None):
        path=ROOT/'configs/rrt_connect.yaml' if path is None else Path(path)
        return cls(**yaml.safe_load(path.read_text(encoding='utf-8')))

    def __post_init__(self):
        for name in ('max_extension_rad','max_planning_time_s','max_search_time_s','search_step_rad','final_step_rad','shortcut_time_s'):
            v=getattr(self,name)
            if not np.isfinite(v) or v<=0:raise ValueError(f'Positive finite {name} required')
        for name in ('max_iterations','max_connect_steps','shortcut_attempts'):
            v=getattr(self,name)
            if not isinstance(v,(int,np.integer)) or isinstance(v,bool) or v<0:raise ValueError(f'Nonnegative integer {name} required')
        if self.max_connect_steps<1:raise ValueError('CONNECT needs a finite positive step budget')
        if not np.isfinite(self.goal_bias) or not 0<=self.goal_bias<=1:raise ValueError('goal_bias must be in [0,1]')
        if self.search_step_rad>.02 or self.final_step_rad>.005 or self.final_step_rad>self.search_step_rad:
            raise ValueError('Cannot relax Phase2A .02/.005 rad resolutions')


@dataclass
class Node:
    q:np.ndarray
    parent:int|None


@dataclass
class Tree:
    root_role:str
    nodes:list=field(default_factory=list)

    def trace(self,index):
        out=[];seen=set()
        while index is not None:
            if index in seen or not 0<=index<len(self.nodes):raise ValueError('Broken/cyclic parent chain')
            seen.add(index);node=self.nodes[index];out.append(node.q.copy());index=node.parent
        return list(reversed(out))


def clean_path(path):
    result=[]
    for q in path:
        q=np.asarray(q,dtype=float).copy()
        if not result or not np.array_equal(q,result[-1]):result.append(q)
    return result


def path_length(path):
    return float(sum(np.linalg.norm(np.asarray(b)-np.asarray(a)) for a,b in zip(path,path[1:])))


def reconstruct_path(a,ia,b,ib):
    if {a.root_role,b.root_role}!={'start','goal'}:raise ValueError('Two differently rooted trees required')
    pa,pb=a.trace(ia),b.trace(ib)
    if not np.array_equal(pa[-1],pb[-1]):raise ValueError('Trees did not connect at exactly the same verified configuration')
    left,right=(pa,pb) if a.root_role=='start' else (pb,pa)
    return clean_path(left+list(reversed(right))[1:])


@dataclass
class PlanResult:
    success:bool=False
    path:list=field(default_factory=list)
    raw_path:list=field(default_factory=list)
    smoothed_path:list=field(default_factory=list)
    failure_reason:str=''
    planning_time:float=0.
    search_time:float=0.
    smoothing_time:float=0.
    validation_time:float=0.
    iterations:int=0
    nodes_start_tree:int=0
    nodes_goal_tree:int=0
    state_valid_calls:int=0
    explicit_state_valid_calls:int=0
    edge_valid_calls:int=0
    geom_distance_calls:int=0
    collision_check_time:float=0.
    minimum_sampled_clearance:float|None=None
    minimum_is_lower_bound:bool=False
    path_length:float=0.
    raw_path_length:float=0.
    raw_node_count:int=0
    smoothed_node_count:int=0
    final_validation_passed:bool=False
    raw_final_validation_passed:bool=False
    seed:int=0
    config:dict=field(default_factory=dict)
    phases:dict=field(default_factory=dict)
    start_validation:dict=field(default_factory=dict)
    goal_validation:dict=field(default_factory=dict)
    raw_validation:dict=field(default_factory=dict)
    smoothed_validation:dict=field(default_factory=dict)
    search_fine_discrepancy:dict=field(default_factory=dict)
    tree_events:list=field(default_factory=list)
    shortcut_events:list=field(default_factory=list)
    trees:dict=field(default_factory=dict)

    def to_dict(self):return asdict(self)

    def __bool__(self):return self.success


class BudgetExceeded(Exception):pass


class RRTConnectPlanner:
    def __init__(self,collision_checker,config=None,seed=42):
        if not isinstance(collision_checker,CollisionChecker):raise TypeError('Expected Phase2A CollisionChecker')
        if collision_checker.options['minimum_clearance_m']<.005:raise ValueError('Planner requires at least the Phase2A 5 mm clearance')
        self.checker=collision_checker
        self.config=PlannerConfig.from_file() if config is None else PlannerConfig(**config) if isinstance(config,dict) else config
        if not isinstance(self.config,PlannerConfig):raise TypeError('Expected PlannerConfig or config dict')
        if not isinstance(seed,(int,np.integer)) or seed<0:raise ValueError('Nonnegative integer seed required')
        self.seed=int(seed)
        self.bounds=self.checker.robot.limits.copy()
        if self.checker.planning_bounds is not None:self.bounds=self.checker.planning_bounds.copy()
        self.ranges=self.bounds[:,1]-self.bounds[:,0]
        if np.any(self.ranges<=0):raise ValueError('Sampling bounds need positive joint ranges')

    def _budget(self,search=False):
        now=time.perf_counter()
        if now>=self.deadline:raise BudgetExceeded('time_budget')
        if search and now>=self.search_deadline:raise BudgetExceeded('search_time_budget')
        if not np.array_equal(self.checker.mocap_pos,self.scene_pos) or not np.array_equal(self.checker.mocap_quat,self.scene_quat):
            raise BudgetExceeded('scene_changed_during_plan')

    def _account(self,phase,states,edges,distances,elapsed,explicit=0):
        r=self.result;r.state_valid_calls+=states;r.explicit_state_valid_calls+=explicit
        r.edge_valid_calls+=edges;r.geom_distance_calls+=distances;r.collision_check_time+=elapsed
        p=r.phases.setdefault(phase,dict(state_calls=0,explicit_state_calls=0,edge_calls=0,distance_calls=0,collision_time_s=0.))
        p['state_calls']+=states;p['explicit_state_calls']+=explicit;p['edge_calls']+=edges
        p['distance_calls']+=distances;p['collision_time_s']+=elapsed

    def _state(self,q,phase,checker=None):
        self._budget(phase=='search')
        c=self.checker if checker is None else checker
        t=time.perf_counter();r=c.state_valid(q)
        self._account(phase,1,0,r.distance_calls,time.perf_counter()-t,explicit=1)
        self._budget(phase=='search')
        return r

    def _edge(self,a,b,step,phase,checker=None):
        self._budget(phase=='search')
        c=self.checker if checker is None else checker
        t=time.perf_counter();r=c.edge_valid(a,b,step_rad=step)
        self._account(phase,r.state_calls,1,r.distance_calls,time.perf_counter()-t)
        self._budget(phase=='search')
        return r

    def _nearest(self,tree,q):
        qs=np.array([n.q for n in tree.nodes])
        diff=(qs-q)/self.ranges if self.config.normalized_nearest else qs-q
        return int(np.argmin(np.sum(diff*diff,axis=1)))

    def _extend(self,tree,target,operation):
        self._budget(search=True)
        near=self._nearest(tree,target);a=tree.nodes[near].q
        delta=target-a;length=float(np.linalg.norm(delta))
        if length<=1e-12:
            # Exact target equality is required for a verified zero-length join.
            if np.array_equal(a,target):return 'REACHED',near
            q=target.copy()
        else:q=target.copy() if length<=self.config.max_extension_rad else a+delta*(self.config.max_extension_rad/length)
        node=self._state(q,'search')
        event=dict(iteration=self.result.iterations,operation=operation,tree=tree.root_role,parent=near,q=q.tolist(),
                   state_valid=node.valid,status='TRAPPED',reason=node.rejection_reason)
        if not node.valid:
            self.result.tree_events.append(event);return 'TRAPPED',near
        edge=self._edge(a,q,self.config.search_step_rad,'search')
        event.update(edge_valid=edge.valid,reason=edge.rejection_reason)
        if not edge.valid:
            self.result.tree_events.append(event);return 'TRAPPED',near
        tree.nodes.append(Node(q.copy(),near));index=len(tree.nodes)-1
        status='REACHED' if np.array_equal(q,target) else 'ADVANCED'
        event.update(status=status,node=index);self.result.tree_events.append(event)
        return status,index

    def _connect(self,tree,target):
        status,index='TRAPPED',0
        for _ in range(self.config.max_connect_steps):
            status,index=self._extend(tree,target,'CONNECT')
            if status!='ADVANCED':return status,index
        return 'CONNECT_BUDGET',index

    def _validate_path(self,path,start,goal,step,phase,checker):
        info=dict(passed=False,step_rad=step,samples=0,minimum_sampled_clearance=None,minimum_is_lower_bound=False,
                  first_invalid=None,edge_results=[],node_results=[])
        if not path or not np.array_equal(path[0],start) or not np.array_equal(path[-1],goal):
            info['first_invalid']=dict(reason='endpoint_or_direction_mismatch');return info
        def minimum(value,bound):
            if value is not None and (info['minimum_sampled_clearance'] is None or value<info['minimum_sampled_clearance']):
                info['minimum_sampled_clearance']=value;info['minimum_is_lower_bound']=bound
        for i,q in enumerate(path):
            state=self._state(q,phase,checker);info['samples']+=1
            info['node_results'].append(dict(index=i,valid=state.valid,rejection_reason=state.rejection_reason))
            minimum(state.minimum_clearance,state.minimum_is_lower_bound)
            if not state.valid and info['first_invalid'] is None:
                info['first_invalid']=dict(node_index=i,q=np.asarray(q).tolist(),reason=state.rejection_reason,
                                           offending_geom_pairs=state.offending_geom_pairs)
        for i,(a,b) in enumerate(zip(path,path[1:])):
            edge=self._edge(a,b,step,phase,checker);info['samples']+=edge.state_calls
            minimum(edge.minimum_clearance,edge.minimum_is_lower_bound)
            info['edge_results'].append(dict(edge_index=i,**edge.to_dict()))
            if not edge.valid and info['first_invalid'] is None:
                info['first_invalid']=dict(edge_index=i,q=edge.first_invalid_q,alpha=edge.first_invalid_alpha,
                                           reason=edge.rejection_reason,offending_geom_pairs=edge.offending_geom_pairs)
        info['passed']=info['first_invalid'] is None
        return info

    def _shortcut(self,path):
        path=clean_path(path);began=time.perf_counter()
        for attempt in range(self.config.shortcut_attempts):
            self._budget()
            if len(path)<3 or time.perf_counter()-began>=self.config.shortcut_time_s:break
            i,j=sorted(self.shortcut_rng.choice(len(path),size=2,replace=False).tolist())
            if j-i<2:continue
            edge=self._edge(path[i],path[j],self.config.search_step_rad,'shortcut')
            candidate=path[:i+1]+path[j:]
            accepted=edge.valid and path_length(candidate)<path_length(path)-1e-12
            self.result.shortcut_events.append(dict(attempt=attempt,i=i,j=j,edge_valid=edge.valid,accepted=accepted,
                                                    length_before=path_length(path),length_candidate=path_length(candidate),
                                                    reason=edge.rejection_reason))
            if accepted:path=candidate
        self.result.smoothing_time=time.perf_counter()-began
        return path

    def plan(self,q_start,q_goal):
        started=time.perf_counter();self.deadline=started+self.config.max_planning_time_s
        self.search_deadline=min(self.deadline,started+self.config.max_search_time_s)
        self.scene_pos=self.checker.mocap_pos.copy();self.scene_quat=self.checker.mocap_quat.copy()
        self.rng=np.random.default_rng(self.seed);self.shortcut_rng=np.random.default_rng(np.random.SeedSequence([self.seed,982451653]))
        self.result=PlanResult(seed=self.seed,config=asdict(self.config));r=self.result
        start_tree=goal_tree=None
        try:
            state=self._state(q_start,'endpoints');r.start_validation=state.to_dict()
            if not state.valid:r.failure_reason='invalid_start:'+state.rejection_reason;return r
            state=self._state(q_goal,'endpoints');r.goal_validation=state.to_dict()
            if not state.valid:r.failure_reason='invalid_goal:'+state.rejection_reason;return r
            start=np.asarray(q_start,dtype=float).copy();goal=np.asarray(q_goal,dtype=float).copy()
            start_tree=Tree('start',[Node(start,None)]);goal_tree=Tree('goal',[Node(goal,None)])
            if np.array_equal(start,goal):raw=[start.copy()]
            else:
                a,b=start_tree,goal_tree;raw=None
                for iteration in range(self.config.max_iterations):
                    self._budget(search=True);r.iterations=iteration+1
                    target=b.nodes[0].q.copy() if self.rng.random()<self.config.goal_bias else self.rng.uniform(self.bounds[:,0],self.bounds[:,1])
                    status,ia=self._extend(a,target,'EXTEND')
                    if status!='TRAPPED':
                        status_b,ib=self._connect(b,a.nodes[ia].q)
                        if status_b=='REACHED':raw=reconstruct_path(a,ia,b,ib);break
                    a,b=b,a
                if raw is None:r.failure_reason='iteration_budget';return r
            r.search_time=time.perf_counter()-started
            r.raw_path=[q.tolist() for q in raw];r.raw_node_count=len(raw);r.raw_path_length=path_length(raw)
            coarse=self._validate_path(raw,start,goal,self.config.search_step_rad,'raw_search_validation',self.checker)
            if not coarse['passed']:
                r.search_fine_discrepancy=dict(coarse=coarse);r.failure_reason='reconstructed_path_invalid';return r
            validator=CollisionChecker(self.checker.model,options=dict(self.checker.options))
            validator.mocap_pos[:]=self.scene_pos;validator.mocap_quat[:]=self.scene_quat
            t=time.perf_counter();r.raw_validation=self._validate_path(raw,start,goal,self.config.final_step_rad,'raw_fine_validation',validator)
            r.validation_time+=time.perf_counter()-t;r.raw_final_validation_passed=r.raw_validation['passed']
            if not r.raw_final_validation_passed:
                r.search_fine_discrepancy=dict(search_passed=True,fine_passed=False,first_invalid=r.raw_validation['first_invalid'])
                r.failure_reason='raw_fine_validation_failed';return r
            smooth=self._shortcut(raw);r.smoothed_path=[q.tolist() for q in smooth];r.smoothed_node_count=len(smooth)
            t=time.perf_counter();r.smoothed_validation=self._validate_path(smooth,start,goal,self.config.final_step_rad,'smoothed_fine_validation',validator)
            r.validation_time+=time.perf_counter()-t
            r.final_validation_passed=r.smoothed_validation['passed']
            if not r.final_validation_passed:
                r.search_fine_discrepancy=dict(shortcut_search_passed=True,fine_passed=False,first_invalid=r.smoothed_validation['first_invalid'])
                r.failure_reason='smoothed_fine_validation_failed';return r
            self._budget();r.success=True;r.path=r.smoothed_path.copy();r.path_length=path_length(smooth)
            r.minimum_sampled_clearance=r.smoothed_validation['minimum_sampled_clearance']
            r.minimum_is_lower_bound=r.smoothed_validation['minimum_is_lower_bound']
        except BudgetExceeded as exc:r.failure_reason=str(exc)
        except (ValueError,RuntimeError) as exc:r.failure_reason='planner_error:'+str(exc)
        finally:
            r.planning_time=time.perf_counter()-started
            if not r.search_time:r.search_time=r.planning_time
            for tree in (start_tree,goal_tree):
                if tree is not None:
                    r.trees[tree.root_role]=[dict(q=n.q.tolist(),parent=n.parent) for n in tree.nodes]
            r.nodes_start_tree=len(start_tree.nodes) if start_tree else 0
            r.nodes_goal_tree=len(goal_tree.nodes) if goal_tree else 0
        return r
