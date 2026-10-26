"""Recompute dataset/closed-loop evidence, without filtering failed student episodes."""
from _bootstrap import ROOT,main_guard
from pathlib import Path
import hashlib
import numpy as np
from src.bc_residual.core import design,Policy
from scripts.phase3c_common import read,rows
from src.common import write_json,write_csv,sha


def vectors(records,key):return np.array([[float(r[f'{key}_{j}']) for j in range(1,8)] for r in records])


def main():
    manifest=rows(ROOT/'results/phase3d_bc_dataset_manifest.csv');data=[];trainarrays=[]
    for r in manifest:
        z=np.load(r['samples_file']);x=z['observations85'];y=z['labels'];u=rows(Path(r['run_directory'])/'control_signals.csv');t=rows(Path(r['run_directory'])/'tracking.csv')
        assert len(t)==int(r['actual_physics_steps']) and len(x)==int(r['samples']) and np.isfinite(x).all() and np.isfinite(y).all()
        assert np.array_equal(x[:,-7:],y) and np.array_equal(z['actions'],y)
        assert np.all(np.diff(z['physics_step'])==25) and z['physics_step'][0]==750
        for j,step in enumerate(z['physics_step']):
            signal=u[int(step)-1];expected=np.array([float(signal[f'expert_integral_{k}']) for k in range(1,8)])/.002
            assert np.max(abs(expected-y[j]))<1e-7
            assert np.array_equal(np.array([float(signal[f'observable_integral_{k}']) for k in range(1,8)]),np.array([float(signal[f'expert_integral_{k}']) for k in range(1,8)]))
        content=hashlib.sha256(x.tobytes()+y.tobytes()+z['physics_step'].tobytes()).hexdigest()
        d=dict(r,content_sha256=content,target_applied_input_max_difference=float(np.max(abs(y-x[:,71:78]))),
            label_mean_abs_normalized=float(np.mean(abs(y))),label_max_normalized=float(np.max(abs(y))),observation_integral_identity=True)
        data.append(d)
        if r['split']=='train' and r['quality_accepted']=='True':trainarrays.append(x)
        z.close()
    write_csv(ROOT/'results/phase3d_bc_dataset_audit.csv',data);train=np.concatenate(trainarrays);minimum=train.min(axis=0);maximum=train.max(axis=0)
    allcases=[]
    for stage in ('validation','final'):
        run=Path(read(ROOT/f'results/phase3d_bc_eval_{stage}_latest.json')['run'])
        for r in read(run/'complete.json')['results']:
            p=Path(r['run_directory']);t=rows(p/'tracking.csv');motion=[x for x in t if x['phase'] in ('trajectory','settle')];u=rows(p/'control_signals.csv');z=np.load(p/'bc_samples.npz')
            tcp=float(np.sqrt(np.mean([float(x['tcp_error_m'])**2 for x in motion])));joint=float(np.sqrt(np.mean((vectors(motion,'q_actual')-vectors(motion,'q_ref'))**2)))
            assert abs(tcp-r['tcp_rmse_m'])<1e-12 and abs(joint-r['joint_rmse_scalar_rad'])<1e-12
            actual=vectors(u,'residual_rad');target=vectors(u,'residual_target_rad');ctrl=vectors(t,'ctrl');force=vectors(u,'actuator_force');frozen=read(p/'frozen_run_config.json');parameters=read(p/'model_parameters.json')
            limits=np.array([x['range_rad'] for x in frozen['actuators']['arm_joints']]);force_limits=np.array([x['forcerange'] for x in frozen['actuators']['arm_joints']])
            assert np.max(abs(actual))<=.002+1e-12 and np.max(abs(target))<=.002+1e-12
            assert np.max(abs(np.diff(ctrl,axis=0)/.002))<=.35+1e-9
            assert np.all(force>=force_limits[:,0]-1e-8) and np.all(force<=force_limits[:,1]+1e-8)
            assert np.all(ctrl>=limits[:,0]) and np.all(ctrl<=limits[:,1]) and r['live_qpos_writes_after_initialization']==0
            # A protective hold may reset residual bookkeeping; report active slew separately.
            active=np.array([x['phase']!='protective_hold' for x in t]);rate=np.diff(actual,axis=0)/.002;mask=active[1:]&active[:-1]
            assert np.max(abs(rate[mask]))<=.01+1e-9
            student_mse=float(np.mean((z['actions']-z['labels'])**2));assert abs(student_mse-r['closed_loop_target_mse'])<1e-12
            dim=78 if r['method']=='BC78' else 85
            outside=np.mean((z['observations85'][:,:dim]<minimum[:dim]-1e-6)|(z['observations85'][:,:dim]>maximum[:dim]+1e-6))
            allcases.append(dict(split=stage,scene=r['scene'],condition=r['condition']['name'],method=r['method'],physical_seed=r['physical_seed'],success=r['success'],
                failure_reason=r['failure_reason'],completion_progress=r['maximum_completed_reference_progress'],tcp_rmse_mm=tcp*1000,joint_rmse_rad=joint,
                final_tcp_mm=r['final_tcp_error_m']*1000,minimum_clearance_mm=r['minimum_actual_clearance_m']*1000,
                safety_violation_steps=r['safety_violation_steps'],prohibited_contact_observations=r['prohibited_contact_observations'],
                max_target_residual_rad=float(np.max(abs(target))),max_applied_residual_rad=float(np.max(abs(actual))),
                residual_mean_squared_rate=float(np.mean(rate**2)),command_mean_squared_rate=float(np.mean((np.diff(ctrl,axis=0)/.002)**2)),
                max_active_residual_rate_rad_s=float(np.max(abs(rate[mask]))),force_saturation_steps=r['force_saturation_steps'],
                actual_physics_steps=r['physics_steps'],simulation_s=r['simulation_time_s'],own_state_target_mse=student_mse,own_state_target_rmse_rad=np.sqrt(student_mse)*.002,
                feature_values_outside_training_range_fraction=float(outside),full_record_tcp_rmse_mm=float(np.sqrt(np.mean([float(x['tcp_error_m'])**2 for x in t])))*1000,
                checkpoint=r['checkpoint'] or '',checkpoint_sha256=sha(r['checkpoint']) if r['checkpoint'] else '',
                nominal_signature=parameters['nominal_signature'],physics_signature=parameters['physics_signature'],path_sha256=frozen['path_evidence']['path_csv_sha256'],run_directory=str(p)))
            z.close()
    for key in {(r['split'],r['scene'],r['condition']) for r in allcases}:
        group=[r for r in allcases if (r['split'],r['scene'],r['condition'])==key]
        assert len(group)==5 and len({r['physical_seed'] for r in group})==len({r['physics_signature'] for r in group})==len({r['path_sha256'] for r in group})==1
    write_csv(ROOT/'results/phase3d_bc_evaluation.csv',allcases)
    duplicates={s:dict(runs=sum(r['split']==s for r in data),unique_observation_label_trajectories=len({r['content_sha256'] for r in data if r['split']==s})) for s in ('train','validation')}
    write_json(ROOT/'results/phase3d_bc_analysis.json',dict(passed=True,dataset_runs=len(data),dataset_samples=sum(int(r['samples']) for r in data),
        dataset_actual_physics_steps=sum(int(r['actual_physics_steps']) for r in data),duplicates=duplicates,all_evaluations=allcases,
        maximum_teacher_target_applied_input_difference=max(r['target_applied_input_max_difference'] for r in data),
        limitation='The78D input already contains previous action and applied residual. Teacher desired/applied can coincide;85D contains an independently observable integral. No claim of entirely new learned control.'))
    learning_plot()
    print('ANALYZED',duplicates,len(allcases),flush=True)


def learning_plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tr=rows(ROOT/'results/phase3d_bc_training.csv');fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    for dim,color in [(78,'tab:blue'),(85,'tab:orange')]:
        group=[r for r in tr if int(r['dimension'])==dim];epochs=[int(r['epoch']) for r in group]
        ax.plot(epochs,[float(r['training_mse']) for r in group],color=color,label=f'BC{dim} train')
        ax.plot(epochs,[float(r['validation_mse']) for r in group],color=color,ls='--',label=f'BC{dim} validation')
        best=min(group,key=lambda r:float(r['validation_mse']));ax.scatter([int(best['epoch'])],[float(best['validation_mse'])],color=color,marker='*',s=110)
    ax.set(yscale='log',xlabel='Supervised epoch (no PPO)',ylabel='Normalized target action MSE',title='Trajectory-disjoint validation; train-only normalization; best val checkpoint');ax.legend();ax.grid(alpha=.2)
    fig.savefig(ROOT/'assets/plots/phase3d_bc_learning_curve.png',dpi=150);plt.close(fig)


if __name__=='__main__':main_guard(main)
