"""Frozen R0/R1 rewards and a compatible instrumentation-only Gym extension."""
from src.rl.panda_residual_env import PandaResidualEnv
from src.control.trajectory_executor import CSVStream
from src.common import ROOT,write_json
import json
import numpy as np

def design():return json.loads((ROOT/'configs/phase3c_pilot.json').read_text(encoding='utf-8'))

def reward_r1(rows,action,previous_action,previous_progress,duration,success,ended,contact):
    c=design()['R1'];dt=len(rows)*.002;scale=dt/duration
    qe=np.array([[r[f'q_actual_{j}']-r[f'q_ref_{j}'] for j in range(1,8)] for r in rows])
    ve=np.array([[r[f'dq_actual_{j}']-r[f'dq_ref_{j}'] for j in range(1,8)] for r in rows])
    p=max(0.,rows[-1]['progress']-previous_progress)
    components=dict(joint=-scale*c['joint_weight']*min(1.,float(np.mean((qe/c['joint_scale_rad'])**2))),
        tcp=-scale*c['tcp_weight']*min(1.,float(np.mean([(r['tcp_error_m']/c['tcp_scale_m'])**2 for r in rows]))),
        velocity=-scale*c['velocity_weight']*min(1.,float(np.mean((ve/c['velocity_scale_rad_s'])**2))),
        action=-scale*c['action_weight']*float(np.mean(action**2)),
        smoothness=-scale*c['smooth_weight']*min(1.,float(np.mean((action-previous_action)**2))),
        time=-scale*c['time_weight'],progress=c['progress_total']*p,
        success=c['success_bonus'] if ended and success else 0.,failure=-c['failure_penalty'] if ended and not success else 0.,
        contact=-c['contact_penalty'] if ended and contact else 0.)
    return float(sum(components.values())),components

class DiagnosticEnv(PandaResidualEnv):
    def __init__(self,sources,dest,version='R0',seed=42,episode_offset=0,**kwargs):
        super().__init__(sources,dest,conditions=design()['training_conditions'],**kwargs)
        if version not in ('R0','R1'):raise ValueError('Frozen reward version required')
        self.version=version;self.training_seed=seed;self.episode_offset=episode_offset;self.schedule=design()['training_conditions']
        self._diag_stream=None
    def _on_step(self,e,row):
        if self._diag_stream is None:self._diag_stream=CSVStream(self.folder/'phase3c_physics_diagnostics.csv')
        last=e.controller.last;force=e.data.actuator_force[e.robot.actuator_ids];limits=e.controller.gravity.force_limits
        self._diag_stream.append(dict(physics_step=e.steps,time=row['time'],total_command_slew_intervened=last['rate_limited'],
            residual_slew_intervened=bool(np.any(abs(last['residual_target_rad']-last['residual_rad'])>1e-12)),
            force_saturated=bool(np.any((force<=limits[:,0]+1e-8)|(force>=limits[:,1]-1e-8)))))
        super()._on_step(e,row)
    def reset(self,*,seed=None,options=None):
        if self.mode=='training':
            index=self.episode+1+self.episode_offset
            # Guaranteed balanced valid mismatch exposure, same schedule for R0/R1.
            condition=self.schedule[(index+self.training_seed%len(self.schedule))%len(self.schedule)]
            options=dict(options or {},perturbation=condition,physical_seed=self.training_seed*10000+index)
        obs,info=super().reset(seed=seed,options=options)
        self.c_stream=CSVStream(self.folder/'phase3c_decisions.csv');self.c_return=0.
        return obs,info
    def step(self,action):
        prior_action=self.last_action.copy();prior_progress=self.previous_clock
        obs,r0,terminated,truncated,info=super().step(action)
        clipped=np.clip(np.asarray(action,dtype=float),-1,1);ended=terminated or truncated
        r1,parts=reward_r1(self.rows,clipped,prior_action,prior_progress,self.e.trajectory.duration,
            info['success'],ended,info['prohibited_contact'])
        reward=r0 if self.version=='R0' else r1;self.c_return+=reward
        info['R0_reward']=r0;info['R1_reward']=r1
        info['reward_components_selected']=info['reward_components'] if self.version=='R0' else parts
        info['condition_name']=self.e.controller.perturbation['name']
        self.c_stream.append(dict(decision=self.episode_decisions,physics_step=self.completed,R0_reward=r0,R1_reward=r1,
            selected_reward=reward,selected_return=self.c_return,version=self.version,terminated=terminated,truncated=truncated,
            **{f'R1_{k}':v for k,v in parts.items()},action=clipped.tolist()))
        if not np.isfinite(reward):raise FloatingPointError('Invalid selected reward')
        return obs,reward,terminated,truncated,info
    def close_episode(self,keep_stream=False):
        super().close_episode(keep_stream=keep_stream)
        if not keep_stream:
            if hasattr(self,'decision_stream'):self.decision_stream.close()
            if hasattr(self,'c_stream'):self.c_stream.close()
            if getattr(self,'_diag_stream',None):self._diag_stream.close();self._diag_stream=None
