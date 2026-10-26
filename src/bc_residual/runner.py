"""Original ctrl+mj_step execution, causal shadow integral and 20Hz actor adapter."""
from pathlib import Path
import numpy as np
from src.bc_residual.core import design,ShadowController,ObservationBuilder,Policy
import torch
from src.experiments.integral_control import IntegralM3Controller
from src.rl.residual_controller import ResidualM3Controller
from src.control.trajectory_executor import CSVStream
from src.common import write_json
from scripts.run_phase3a import load_new,UncertainRun


def run_case(job,record_video=False):
    source,condition,seed,method,checkpoint,directory=job;dest=Path(directory);torch.set_num_threads(1)
    run=UncertainRun(load_new(source),'M3',condition,seed,dest);e=run.e;run.signals.close()
    if method=='Integral500':inner=IntegralM3Controller(e.robot,run.nominal,condition,seed,lambda:bool(e.failure_reason))
    else:inner=ResidualM3Controller(e.robot,run.nominal,condition,seed,lambda:bool(e.failure_reason))
    e.controller=ShadowController(inner,lambda:bool(e.failure_reason),expert=method=='Integral500')
    builder=ObservationBuilder();policy=Policy(checkpoint) if checkpoint else None
    stream=CSVStream(dest/'control_signals.csv');decision_stream=CSVStream(dest/'bc_decisions.csv');observations=[];labels=[];actions=[];steps=[];times=[];control_errors=[]
    qmax=0.;rmax=0.;force_saturated=0;command_rate_steps=0;residual_rate_steps=0;native_forces=[];video=None
    if record_video:
        from src.bc_residual.recording import Recording
        video=Recording(e,dest,method,condition)
    if len(job)>6:raise ValueError('Unexpected job')
    def observe(ex,row):
        nonlocal qmax,rmax,force_saturated,command_rate_steps,residual_rate_steps
        c=e.controller;last=c.last;force=e.data.actuator_force[e.robot.actuator_ids];bounds=c.gravity.force_limits
        signal=dict(time=row['time'],physics_step=e.steps)
        for name in ('observed_q','observed_velocity','base_target','residual_rad','residual_target_rad','delayed_target','desired_before_delay','noise'):
            for j,v in enumerate(last[name]):signal[f'{name}_{j+1}']=float(v)
        for j in range(7):
            signal[f'observable_integral_{j+1}']=float(c.memory.value[j]);signal[f'expert_integral_{j+1}']=float(inner.integral[j]) if method=='Integral500' else None
            signal[f'actuator_force_{j+1}']=float(force[j]);signal[f'antiwindup_{j+1}']=bool(c.memory.blocked[j])
        signal['algorithm_s']=last['algorithm_s'];stream.append(signal)
        qmax=max(qmax,float(np.max(abs(last['residual_target_rad']))));rmax=max(rmax,float(np.max(abs(last['residual_rad']))))
        force_saturated+=int(np.any((force<=bounds[:,0]+1e-8)|(force>=bounds[:,1]-1e-8)));command_rate_steps+=int(last['rate_limited'])
        residual_rate_steps+=int(np.any(abs(last['residual_rad']-last['residual_target_rad'])>1e-12))
        if e.steps>=750 and (e.steps-750)%25==0 and not e.success and not e.failure_reason:
            try:
                obs=builder.make(e,row);label=np.clip(c.memory.value/.002,-1,1).astype(np.float32)
                if policy:action=policy.predict(obs)
                elif method in ('Integral500','Integral20'):action=label.copy()
                elif method=='M3':action=np.zeros(7,dtype=np.float32)
                else:raise ValueError('Unknown frozen method')
                observations.append(obs);labels.append(label);actions.append(action.copy());steps.append(e.steps);times.append(float(row['time']))
                decision_stream.append(dict(decision=len(observations),physics_step=e.steps,time=row['time'],observation85=obs.tolist(),label=label.tolist(),action=action.tolist(),
                    observable_integral=c.memory.value.tolist(),expert_desired_target=last['residual_target_rad'].tolist(),applied=last['residual_rad'].tolist(),ctrl=e.data.ctrl[e.robot.actuator_ids].tolist(),
                    label_semantics='causal current desired integral / cap; pre-next-interval, not future/post-slew'))
                builder.last_action=action.copy()
                if method!='Integral500':inner.set_residual(action)
            except Exception as exc:control_errors.append(repr(exc));inner.set_residual(np.zeros(7))
        if video:video.capture(row,actions[-1] if actions else np.zeros(7))
    try:r=e.run(on_step=observe,stop_requested=lambda:'bc_control_error' if control_errors else '')
    finally:
        stream.close();decision_stream.close()
        if video:video.close()
    np.savez_compressed(dest/'bc_samples.npz',observations85=np.array(observations),labels=np.array(labels),actions=np.array(actions),physics_step=np.array(steps),time=np.array(times))
    valid=not r['errors'] and not control_errors and not np.any(e.data.warning.number)
    quality=bool(valid and r['success'] and r['safety_violation_steps']==0 and r['prohibited_contact_observations']==0 and qmax<=.002+1e-12 and rmax<=.002+1e-12 and ccheck(e))
    r.update(method=method,condition=condition,physical_seed=seed,source=source,checkpoint=checkpoint,samples=len(observations),dataset_quality_accepted=quality,
        max_target_residual_rad=qmax,max_applied_residual_rad=rmax,force_saturation_steps=force_saturated,effective_command_slew_steps=command_rate_steps,
        residual_slew_steps=residual_rate_steps,observable_integral_teacher_max_difference=e.controller.max_teacher_difference,
        control_errors=control_errors,model_warnings=e.data.warning.number.tolist(),valid_model_and_execution=valid,
        closed_loop_target_mse=float(np.mean((np.array(actions)-np.array(labels))**2)) if labels else None,
        joint_rmse_scalar_rad=float(np.sqrt(np.mean(np.array(r['joint_rmse_rad'])**2))) if r['joint_rmse_rad'] is not None else None,
        maximum_completed_reference_progress=float(e.trajectory.sample(min(e.trajectory.duration,e.last_row['trajectory_time'])).progress),
        samples_file=str(dest/'bc_samples.npz'))
    write_json(dest/'bc_summary.json',r);write_json(dest/'comparison_summary.json',r);return r


def ccheck(e):return e.controller.max_teacher_difference<=1e-12
