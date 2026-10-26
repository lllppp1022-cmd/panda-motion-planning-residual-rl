"""Read-only logs -> explicit-fidelity replay, never policy sampling or mj_step."""
from pathlib import Path
from bisect import bisect_right
import json
import numpy as np
import mujoco
from scripts.phase3c_common import rows,read
from scripts.run_phase3a import load_new
from src.control.uncertain_models import model_pair
from src.training_observer.observer import numbers,annotate,STATE


class RecordedEpisode:
    def __init__(self,run,episode=0,mode='actual'):
        self.run=Path(run);self.mode=mode;self.episode=episode
        if (self.run/'actual_training').exists():
            folders=sorted((self.run/'actual_training').glob('episode_*'));self.folder=folders[episode]
            samples=rows(self.run/'training_samples.csv');offset=0;self.actions={};self.mapping={};self.end_steps=[]
            for folder in folders:
                ds=rows(folder/'decisions.csv');previous=750;total=0.
                for d in ds:
                    sample=samples[offset];offset+=1;global_decision=int(sample['decision']);end=int(d['physics_step'])
                    record=dict(sample,global_decision=global_decision,policy_version=(global_decision-1)//256,
                        interval_reward=float(sample['reward']),previous_cumulative_reward=total,
                        terminated=d['terminated'],truncated=d['truncated'],start_step=previous+1,end_step=end)
                    total+=float(sample['reward']);record['cumulative_reward']=total
                    if folder==self.folder:
                        self.actions[global_decision]=record;self.end_steps.append(end)
                        for step in range(previous+1,end+1):self.mapping[step]=record
                    previous=end
            self.reset_version=next(iter(self.actions.values()))['policy_version'] if self.actions else 0
        else:
            self.folder=self.run;self.mapping={};self.actions={};self.end_steps=[];self.reset_version=None
            if (self.folder/'decisions.csv').exists():
                previous=750;total=0.
                for d in rows(self.folder/'decisions.csv'):
                    end=int(d['physics_step']);value=float(d['reward']);number=int(d['decision'])
                    a=dict(global_decision=number,policy_version='fixed checkpoint / evaluation',interval_reward=value,cumulative_reward=total+value,
                        normalized_executed_action=d['action'],terminated=d['terminated'],truncated=d['truncated'],
                        **{k:v for k,v in d.items() if k.startswith('reward_')})
                    for step in range(previous+1,end+1):self.mapping[step]=a
                    self.actions[number]=a;self.end_steps.append(end);previous=end;total+=value
        self.t=rows(self.folder/'tracking.csv');self.u=rows(self.folder/('integral_control_signals.csv' if (self.folder/'integral_control_signals.csv').exists() else 'control_signals.csv'))
        self.summary=read(self.folder/'episode_physics_summary.json') if (self.folder/'episode_physics_summary.json').exists() else read(self.folder/'execution_summary.json')
        reset=read(self.folder/'reset.json') if (self.folder/'reset.json').exists() else None
        frozen=read(self.folder/'frozen_run_config.json');source=reset['source'] if reset else frozen['path_evidence']['source_json']
        parameters=read(self.folder/'model_parameters.json');b=load_new(source)
        _,self.model,_=model_pair(b.checker.model,parameters['perturbation']);self.data=mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model,self.data,b.checker.home_id)
        self.data.mocap_pos[:]=b.checker.mocap_pos;self.data.mocap_quat[:]=b.checker.mocap_quat
        self.qindices=np.array([x['qpos_address'] for x in frozen['actuators']['arm_joints']]);self.dindices=np.array([x['dof_address'] for x in frozen['actuators']['arm_joints']]);self.aindices=np.array([x['actuator_id'] for x in frozen['actuators']['arm_joints']])
        self.site=frozen['actuators']['tool_site_id'];self.data.qpos[b.checker.fingers]=.04
        self.full=np.load(self.folder/'integration_states.npz') if (self.folder/'integration_states.npz').exists() else None
        self.raw_frames=read(self.folder/'rgb_frames.json') if (self.folder/'rgb_frames.json').exists() else []
        self.camera=mujoco.MjvCamera();self.camera.lookat[:]=[.2,0,.4];self.camera.distance=1.85;self.camera.azimuth=135;self.camera.elevation=-25
        self.renderer=None;self.reader=None
        self.set_rgb_mode(bool(self.raw_frames))

    def set_rgb_mode(self,enabled):
        self.use_rgb=bool(enabled and self.raw_frames)
        self.fidelity='TRAINING-SITE RGB' if self.use_rgb else 'FULL RECORDED STATE RECONSTRUCTION' if self.full is not None else 'LOG RECONSTRUCTION / missing historical fingers and full state'

    def set_state(self,index):
        r=self.t[index]
        if self.full is not None:mujoco.mj_setState(self.model,self.data,self.full['states'][index],STATE)
        else:
            self.data.qpos[self.qindices]=[float(r[f'q_actual_{j}']) for j in range(1,8)]
            self.data.qvel[self.dindices]=[float(r[f'dq_actual_{j}']) for j in range(1,8)]
            self.data.ctrl[self.aindices]=[float(r[f'ctrl_{j}']) for j in range(1,8)];self.data.time=float(r['time'])
        mujoco.mj_forward(self.model,self.data)

    def information(self,index):
        r=self.t[index];u=self.u[index];action=self.mapping.get(int(r['physics_step']))
        summary='SUCCESS' if self.summary['success'] else self.summary['failure_reason']
        if summary=='rl_external_truncation':summary='BUDGET / EXTERNAL STOP (unfinished, not a task failure)'
        return dict(row=r,signal=u,action=action,status=summary,source=self.fidelity,mode=self.mode,
            policy_version=action['policy_version'] if action else self.reset_version,
            global_decision=action['global_decision'] if action else None)

    def lines(self,index):
        info=self.information(index);r=info['row'];u=info['signal'];a=info['action'] or {}
        name={'actual':'Actual Training','evaluation':'Checkpoint Evaluation / deterministic','comparison':'Baseline Comparison'}[self.mode]
        raw=json.loads(a.get('raw_sample_action','[]'));clipped=json.loads(a.get('normalized_executed_action','[]'))
        parts=[f'{k[7:]}={float(v):+.3g}' for k,v in a.items() if k.startswith('reward_')]
        return [f'{name} | {self.fidelity} | {self.run.name}',
            f'Episode {self.episode} | step {r["physics_step"]} | decision {info["global_decision"]} | policy v{info["policy_version"]} | t={float(r["time"]):.3f}s',
            'q_ref: '+numbers([float(r[f'q_ref_{j}']) for j in range(1,8)]),
            'q_actual: '+numbers([float(r[f'q_actual_{j}']) for j in range(1,8)]),
            'Actor mean: '+numbers(json.loads(a.get('actor_mean','[]'))),'Actor std: '+numbers(json.loads(a.get('actor_std','[]'))),
            'Actor raw: '+numbers(raw),'clipped: '+numbers(clipped),
            'target residual: '+numbers([float(u.get(f'residual_target_rad_{j}',0)) for j in range(1,8)]),
            'applied residual: '+numbers([float(u.get(f'residual_rad_{j}',0)) for j in range(1,8)]),
            f'TCP error={float(r["tcp_error_m"])*1000:.5f}mm | RECORDED actual gap={float(r["minimum_clearance_m"])*1000:.5f}mm',
            f'Recorded interval reward={a.get("interval_reward","N/A")} | cumulative={a.get("cumulative_reward","N/A")}',
            'Parts: '+' | '.join(parts[:5]),
            'Parts: '+' | '.join(parts[5:]),
            f'{r["phase"]} | RECORDED FINAL outcome: {info["status"]}',
            ]

    def image(self,index):
        if self.use_rgb:
            import imageio.v2 as imageio
            if self.reader is None:self.reader=imageio.get_reader(str(self.folder/'actual_rgb.mp4'))
            number=max(0,bisect_right([f['time'] for f in self.raw_frames],float(self.t[index]['time']))-1)
            self.displayed_index=int(self.raw_frames[number]['physics_step'])-1
            return annotate(self.reader.get_data(number),self.lines(self.displayed_index))
        self.displayed_index=index
        self.set_state(index)
        if self.renderer is None:self.renderer=mujoco.Renderer(self.model,height=544,width=960)
        self.renderer.update_scene(self.data,camera=self.camera)
        return annotate(self.renderer.render().copy(),self.lines(index))

    def close(self):
        if self.renderer:self.renderer.close()
        if self.reader:self.reader.close()
        if self.full is not None:self.full.close()
