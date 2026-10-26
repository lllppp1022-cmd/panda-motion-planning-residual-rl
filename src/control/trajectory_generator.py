"""Piecewise linear joint path with rest-to-rest quintic time scaling."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TrajectorySample:
    t:float
    q_ref:np.ndarray
    dq_ref:np.ndarray
    ddq_ref:np.ndarray
    segment:int
    progress:float


class QuinticTrajectory:
    def __init__(self,path,vmax=.1,amax=.3,minimum_duration=.1):
        q=np.asarray(path,dtype=float)
        if q.ndim!=2 or q.shape[1]!=7 or not len(q) or not np.isfinite(q).all():
            raise ValueError('Expected nonempty finite N x 7 joint path')
        def limits(x):
            a=np.broadcast_to(np.asarray(x,dtype=float),(7,)).copy()
            if not np.isfinite(a).all() or np.any(a<=0):raise ValueError('Positive finite 7-joint limits required')
            return a
        self.vmax,self.amax=limits(vmax),limits(amax)
        if not np.isfinite(minimum_duration) or minimum_duration<=0:raise ValueError('Positive minimum duration required')
        self.path=q.copy();self.delta=np.diff(q,axis=0)
        # Exact maximum |s''|=10/sqrt(3), slightly above the rounded 5.7735.
        self.durations=np.array([max(minimum_duration,float(np.max(1.875*np.abs(d)/self.vmax)),
                           float(np.max(np.sqrt((10/np.sqrt(3))*np.abs(d)/self.amax)))) for d in self.delta])
        self.boundaries=np.r_[0.,np.cumsum(self.durations)];self.duration=float(self.boundaries[-1])
        self.path.setflags(write=False)

    def sample(self,t):
        if not np.isscalar(t) or not np.isfinite(t):raise ValueError('Finite scalar time required')
        t=float(np.clip(t,0,self.duration))
        if not len(self.delta):return TrajectorySample(t,self.path[0].copy(),np.zeros(7),np.zeros(7),0,1.)
        i=min(len(self.delta)-1,int(np.searchsorted(self.boundaries,t,side='right')-1))
        T=self.durations[i];u=float(np.clip((t-self.boundaries[i])/T,0,1))
        s=u**3*(10+u*(-15+6*u));ds=30*u*u*(1-u)**2/T
        dds=60*u*(1-u)*(1-2*u)/(T*T)
        return TrajectorySample(t,self.path[i]+s*self.delta[i],ds*self.delta[i],dds*self.delta[i],i,t/self.duration)
