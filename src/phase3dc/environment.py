from pathlib import Path
import numpy as np
from gymnasium import spaces
from src.training_observer.observer import ObservedEnv
from src.bc_residual.core import ShadowController
from src.rl.residual_controller import ResidualM3Controller
from src.control.uncertain_models import model_pair
from src.planning.collision_checker import CollisionChecker
from src.control.validated_path import ValidatedPath
from src.control.trajectory_executor import TrajectoryExecutor,CSVStream
from scripts.phase2a_cases import restore_scene
from scripts.run_phase3a import load_new
from src.common import ROOT,write_json
import json


def design():return json.loads((ROOT/'configs/phase3dc.json').read_text(encoding='utf-8'))


class GatedShadow(ShadowController):
    def __init__(self,inner,holding,gate):super().__init__(inner,holding);self.gate=gate
    def step(self,target):
        # Pause BEFORE any integral/physics mutation, preserving BC decision chronology.
        self.gate()
        return super().step(target)


class DCEnv(ObservedEnv):
    def __init__(self,sources,dest,method='A',seed=42,conditions=None,episode_offset=0,**kwargs):
        self.method=method;self.dc_conditions=conditions or design()['train_conditions'];self.dc_seed=seed;self.offset=episode_offset
        super().__init__(sources,dest,version='R1',seed=seed,**kwargs)
        self.observation_space=spaces.Box(-1.,1.,shape=(85,),dtype=np.float32)
        self.schedule=self.dc_conditions;self.conditions=self.dc_conditions;self._extra=None

    def reset(self,*,seed=None,options=None):
        self.integral_request=np.zeros(7);self.correction=np.zeros(7)
        index=self.episode+1+self.offset
        if self.mode=='training':
            options=dict(options or {},source=self.sources[index%len(self.sources)])
        # Reuse DiagnosticEnv's balanced schedule with NEW conditions and an
        # independent seed-derived physical seed, identical across A/B pairs.
        self.episode_offset=self.offset;self.training_seed=self.dc_seed
        return super().reset(seed=seed,options=options)

    def _observation(self):
        obs=super()._observation()
        return np.r_[obs,np.clip(self.snapshot['own_integral']/.002,-1,1)].astype(np.float32)

    def _on_step(self,e,row):
        # This stream exposes target/composition separately from applied residual and ctrl.
        if self._extra is None:self._extra=CSVStream(self.folder/'dc_control.csv')
        self._extra.append(dict(physics_step=e.steps,time=row['time'],own_integral=e.controller.memory.value.tolist(),
            integral20_request=getattr(self,'integral_request',np.zeros(7)).tolist(),rl_correction_target=getattr(self,'correction',np.zeros(7)).tolist(),
            combined_target=e.controller.last['residual_target_rad'].tolist(),applied=e.controller.last['residual_rad'].tolist(),
            force=e.data.actuator_force[e.robot.actuator_ids].tolist()))
        # Parent notifies main thread; append memory to copied snapshot BEFORE releasing lock.
        with self.cv:
            if self.mode!='training' and self.record:
                # Evaluation RGB has an explicit Baseline Comparison label;
                # never pass deterministic runs to Actual Training capture.
                if self.recorder is None:
                    from src.bc_residual.recording import Recording
                    self.recorder=Recording(e,self.folder,getattr(self,'display_name',self.method),e.controller.perturbation)
                try:self.recorder.capture(row,self.action.copy())
                except Exception as exc:self.record_errors.append(dict(episode=self.episode,error=repr(exc)))
                self.record=False
                try:super()._on_step(e,row)
                finally:self.record=True
            else:super()._on_step(e,row)
            self.snapshot['own_integral']=e.controller.memory.value.copy()

    def step(self,action):
        raw=np.asarray(action,dtype=float)
        if raw.shape!=(7,) or not np.isfinite(raw).all():raise ValueError('Seven finite normalized corrections required')
        raw=np.clip(raw,-1,1)
        # Integral20's historical interface quantizes its normalized request to
        # float32 BEFORE mapping to radians. Preserve that exact zero-RL path.
        integral_normalized=np.clip(self.snapshot['own_integral']/.002,-1,1).astype(np.float32).astype(float)
        self.integral_request=integral_normalized*.002
        self.correction=raw*(design()['correction_cap_rad'] if self.method=='B' else .002)
        normalized=np.clip(integral_normalized+raw*(design()['correction_cap_rad']/.002),-1,1) if self.method=='B' else raw
        target=normalized*.002
        obs,reward,term,trunc,info=super().step(normalized)
        info.update(actor_request=raw.tolist(),integral20_request=self.integral_request.tolist(),correction_target=self.correction.tolist(),
            composed_target=target.tolist(),effective_correction=(target-self.integral_request).tolist() if self.method=='B' else target.tolist())
        return obs,reward,term,trunc,info

    def _worker(self,source,perturbation,seed,folder):
        try:
            b=load_new(source);nominal,physics,parameters=model_pair(b.checker.model,perturbation)
            checker=CollisionChecker(physics,options={'minimum_clearance_m':.005});restore_scene(checker,b.evidence['scene_configuration']['scene'])
            self.e=TrajectoryExecutor(ValidatedPath(b.path,checker,b.evidence),self.cfg['reference_speed_rad_s'],folder)
            inner=ResidualM3Controller(self.e.robot,nominal,perturbation,seed,lambda:bool(self.e.failure_reason),.002,None,residual_slew_rad_s=.01)
            self.e.controller=GatedShadow(inner,lambda:bool(self.e.failure_reason),self._gate)
            write_json(folder/'model_parameters.json',dict(**parameters,seed=seed,source=source,note='nominal parameters ONLY in controller/actor'))
            self.control_stream=CSVStream(folder/'control_signals.csv')
            try:result=self.e.run(on_step=self._on_step,stop_requested=lambda:'rl_external_truncation' if self.aborting else '')
            finally:
                self.control_stream.close()
                if self.recorder:
                    try:self.recorder.close()
                    except Exception as exc:self.record_errors.append(dict(episode=self.episode,error=repr(exc)))
                    self.recorder=None
                if self._extra:self._extra.close();self._extra=None
            if result['errors']:raise RuntimeError(str(result['errors']))
            write_json(folder/'episode_physics_summary.json',result)
            with self.cv:self.result=result;self.episode_results.append(result)
        except BaseException as exc:
            write_json(folder/'worker_error.json',dict(error=repr(exc)));self.worker_error=exc
        finally:
            with self.cv:self.finished=True;self.cv.notify_all()
