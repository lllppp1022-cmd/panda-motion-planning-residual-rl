from collections import deque
from pathlib import Path
import json
import numpy as np
from src.common import ROOT
from src.rl.dependencies import activate
activate()
import torch
from torch import nn


def design():return json.loads((ROOT/'configs/phase3d_bc.json').read_text(encoding='utf-8'))


class ObservableIntegral:
    """Identical causal rule to frozen IntegralM3Controller, no teacher object needed."""
    def __init__(self):self.value=np.zeros(7);self.blocked=np.zeros(7,dtype=bool);self.clamped=0;self.blocked_steps=0
    def advance(self,reference,controller,holding):
        if holding:self.value[:]=0.;self.blocked[:]=False;return self.value.copy()
        q=controller.last['observed_q'] if controller.last else controller.robot.positions()
        dt=float(controller.robot.model.opt.timestep);change=.3*(np.asarray(reference)-q)*dt;blocked=np.zeros(7,dtype=bool)
        if controller.last:
            blocked=abs(controller.last['delayed_target']-controller.command_q)>.35*dt+1e-12
            force=controller.robot.data.actuator_force[controller.robot.actuator_ids];limits=controller.gravity.force_limits
            blocked|=(force<=limits[:,0]+1e-8)|(force>=limits[:,1]-1e-8)
        change[blocked]=0.;raw=self.value+change;self.value=np.clip(raw,-.002,.002);self.blocked=blocked
        self.clamped+=int(np.any(raw!=self.value));self.blocked_steps+=int(np.any(blocked))
        if not np.isfinite(self.value).all():raise FloatingPointError('Invalid observable integral')
        return self.value.copy()


class ShadowController:
    def __init__(self,inner,holding,expert=False):self.inner=inner;self.memory=ObservableIntegral();self.holding=holding;self.expert=expert;self.max_teacher_difference=0.
    def __getattr__(self,name):return getattr(self.inner,name)
    def step(self,reference):
        self.memory.advance(reference,self.inner,self.holding());out=self.inner.step(reference)
        if self.expert:
            difference=float(np.max(abs(self.memory.value-self.inner.integral)));self.max_teacher_difference=max(self.max_teacher_difference,difference)
            if difference>1e-12:raise RuntimeError('Observable integral differs from original teacher')
        return out


class ObservationBuilder:
    def __init__(self):self.history=deque([np.zeros(7) for _ in range(3)],maxlen=3);self.last_action=np.zeros(7);self.first=True
    def make(self,e,row):
        c=e.controller;last=c.last;middle=np.mean(e.robot.limits,axis=1);half=(e.robot.limits[:,1]-e.robot.limits[:,0])/2
        qref=np.array([row[f'q_ref_{j}'] for j in range(1,8)]);dqref=np.array([row[f'dq_ref_{j}'] for j in range(1,8)])
        err=(last['observed_q']-qref)/.05
        if not self.first:self.history.append(np.clip(err,-1,1))
        self.first=False
        obs=np.r_[(last['observed_q']-middle)/half,last['observed_velocity'],(qref-middle)/half,dqref/.15,err,
            (last['base_target']-middle)/half,*list(self.history),row['progress']*2-1,self.last_action,last['residual_rad']/.002]
        obs=np.clip(obs,-1,1).astype(np.float32);extra=np.clip(c.memory.value/.002,-1,1).astype(np.float32)
        result=np.r_[obs,extra].astype(np.float32)
        if result.shape!=(85,) or not np.isfinite(result).all():raise FloatingPointError('Invalid causal BC observation')
        return result


class BCActor(nn.Module):
    def __init__(self,dimension):
        super().__init__();self.dimension=dimension;self.net=nn.Sequential(nn.Linear(dimension,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,7),nn.Tanh())
    def forward(self,obs):return self.net(obs)


class Policy:
    def __init__(self,checkpoint):
        c=torch.load(checkpoint,map_location='cpu',weights_only=True);self.dimension=c['dimension'];self.mean=c['mean'];self.std=c['std']
        self.model=BCActor(self.dimension);self.model.load_state_dict(c['model']);self.model.eval()
    def predict(self,obs):
        x=torch.as_tensor(np.asarray(obs)[:self.dimension],dtype=torch.float32)
        with torch.no_grad():out=self.model((x-self.mean)/self.std).numpy()
        if out.shape!=(7,) or not np.isfinite(out).all():raise FloatingPointError('Invalid BC action')
        return np.clip(out,-1,1)
