from _bootstrap import ROOT,main_guard
from src.rl.dependencies import activate
activate()
import numpy as np
from src.phase3dc.common import *


def vector(rows,key):return np.array([[float(r[f'{key}_{j}']) for j in range(1,8)] for r in rows])


def main():
    stages=['initial'];gate=read(ROOT/'results/phase3dc_2048_gate.json')
    if gate['all_four_passed']:stages.append('continue')
    training=[];actions=[]
    for stage in stages:
        for r in read(Path(read(ROOT/f'results/phase3dc_training_{stage}_latest.json')['run'])/'complete.json')['results']:
            d=Path(r['run_directory']);samples=csvrows(d/'training_samples.csv');action=np.array([json.loads(x['correction_target']) for x in samples]);effective=np.array([json.loads(x['effective_correction']) for x in samples]);mask=abs(action)>1e-12
            raw=np.array([json.loads(x['raw_sample_action']) for x in samples]);clipped=np.array([json.loads(x['normalized_executed_action']) for x in samples])
            physical=sum(len(csvrows(f/'tracking.csv')) for f in (d/'actual_training').glob('episode_*'));assert physical==r['total_physics']
            assert physical==r['reset_physics']+r['decision_physics']+r['cleanup_physics']
            complete=[e for e in r['episode_results'] if e['failure_reason']!='rl_external_truncation']
            training.append(dict(stage=stage,method=r['method'],seed=r['seed'],decisions=r['stage_decisions'],cumulative_decisions=r['cumulative_decisions'],physics=physical,
                reset_physics=r['reset_physics'],decision_physics=r['decision_physics'],cleanup_physics=r['cleanup_physics'],full_episodes=len(complete),successful_full_episodes=sum(e['success'] for e in complete),
                budget_truncated_episodes=sum(e['failure_reason']=='rl_external_truncation' for e in r['episode_results']),safety_violations=sum(e['safety_violation_steps'] for e in r['episode_results']),
                prohibited_contacts=sum(e['prohibited_contact_observations'] for e in r['episode_results']),actor_delta=r['actor_delta_l2'],critic_delta=r['critic_delta_l2'],
                update_calls=len(r['updates']),mean_reward=float(np.mean([float(x['reward']) for x in samples])),checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256'],run_directory=str(d)))
            actions.append(dict(stage=stage,method=r['method'],seed=r['seed'],normalized_action_rms=float(np.sqrt(np.mean(clipped**2))),gaussian_clip_fraction=float(np.mean(raw!=clipped)),
                correction_target_rms_rad=float(np.sqrt(np.mean(action**2))),effective_target_rms_rad=float(np.sqrt(np.mean(effective**2))),
                any_effect_fraction=float(np.mean(abs(effective[mask])>1e-8)),mean_abs_effective_to_requested_ratio=float(np.mean(abs(effective[mask]/action[mask]))),
                composition_clipping_fraction=float(np.mean(abs(action[mask]-effective[mask])>1e-10))))
    evaluations=[]
    for stage in ('baseline','gate','post','final'):
        if stage=='post' and not gate['all_four_passed']:continue
        run=Path(read(ROOT/f'results/phase3dc_evaluation_{stage}_latest.json')['run'])
        for r in read(run/'complete.json')['results']:
            d=Path(r['run_directory']);rows=csvrows(d/'tracking.csv');motion=[x for x in rows if x['phase'] in ('trajectory','settle')];signals=csvrows(d/'control_signals.csv')
            tcp=float(np.sqrt(np.mean([float(x['tcp_error_m'])**2 for x in motion])));assert abs(tcp-r['tcp_rmse_m'])<1e-12
            force=vector(signals,'actuator_force');limits=np.array([87,87,87,87,12,12,12]);ctrl=vector(rows,'ctrl');act=vector(signals,'residual_rad');target=vector(signals,'residual_target_rad')
            assert np.max(abs(target))<=.002+1e-12 and np.max(abs(act))<=.002+1e-12 and np.all(abs(force)<=limits+1e-8)
            assert np.max(abs(np.diff(ctrl,axis=0)))<=.35*.002+1e-9 and r['live_qpos_writes_after_initialization']==0
            evaluations.append(dict(split='test' if stage=='final' else 'validation',stage=stage,scene=read(r['source'])['layout']['name'],condition=r['condition']['name'],method=r['evaluation_method'],physical_seed=r['physical_seed'],
                success=r['success'],failure_reason=r['failure_reason'],tcp_rmse_mm=tcp*1000,joint_rmse_rad=float(np.sqrt(np.mean((vector(motion,'q_actual')-vector(motion,'q_ref'))**2))),
                final_tcp_mm=r['final_tcp_error_m']*1000,minimum_clearance_mm=r['minimum_actual_clearance_m']*1000,completion_progress=max(float(x['progress']) for x in rows),
                safety_violation_steps=r['safety_violation_steps'],prohibited_contacts=r['prohibited_contact_observations'],
                max_actual_velocity=r['actual_max_velocity_rad_s'],max_actual_acceleration=r['actual_max_acceleration_fd_rad_s2'],
                max_target_rad=float(np.max(abs(target))),max_applied_rad=float(np.max(abs(act))),command_squared_rate=float(np.mean((np.diff(ctrl,axis=0)/.002)**2)),
                residual_squared_rate=float(np.mean((np.diff(act,axis=0)/.002)**2)),torque_saturation_samples=int(np.sum(np.any(abs(force)>=limits-1e-8,axis=1))),
                physics_steps=r.get('physics_steps',r.get('actual_physics_steps')),simulation_s=r['simulation_time_s'],wall_s=r['wall_time_s'],checkpoint=r['checkpoint'] or '',checkpoint_sha256=r['checkpoint_sha256'],run_directory=str(d)))
    write_csv(ROOT/'results/phase3dc_training_comparison.csv',training);write_csv(ROOT/'results/phase3dc_action_audit.csv',actions);write_csv(ROOT/'results/phase3dc_controller_evaluation.csv',evaluations)
    write_json(ROOT/'results/phase3dc_analysis.json',dict(training=training,actions=actions,evaluations=evaluations,training_physics=sum(t['physics'] for t in training),total_training_decisions=sum(t['decisions'] for t in training),gate=gate))
    plot(training,evaluations,stages)
    print('ANALYSIS COMPLETE',len(training),len(evaluations),sum(t['physics'] for t in training),flush=True)


def plot(training,evaluations,stages):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for m in ('A','B'):
        for seed in (42,123):
            group=[t for t in training if t['method']==m and t['seed']==seed];updates=[];samples=[]
            for g in group:
                updates+=read(Path(g['run_directory'])/'complete.json')['updates'];samples+=csvrows(Path(g['run_directory'])/'training_samples.csv')
            label=f'{m} seed{seed}';dec=[u['decision_steps'] for u in updates]
            axs[0,0].plot(dec,[np.mean([float(s['reward']) for s in samples[i*256:(i+1)*256]]) for i in range(len(updates))],label=label)
            for ax,key in [(axs[0,1],'value_loss'),(axs[1,0],'explained_variance'),(axs[1,1],'approx_kl')]:ax.plot(dec,[u[key] for u in updates],label=label)
    for ax,title in zip(axs.flat,['Actual stochastic reward (256 decision mean)','Value Loss','Explained Variance','Approx KL']):ax.set(title=title,xlabel='PPO decision');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(ROOT/'assets/plots/phase3dc_learning_curves.png',dpi=150);plt.close(fig)
    methods=['M3','Integral500','Integral20','BC85','A_seed42','A_seed123','B_seed42','B_seed123'];fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True)
    final=[r for r in evaluations if r['split']=='test'];ax.bar(methods,[np.mean([r['tcp_rmse_mm'] for r in final if r['method']==m]) for m in methods],alpha=.6)
    for j,m in enumerate(methods):
        y=[r['tcp_rmse_mm'] for r in final if r['method']==m];ax.scatter(j+np.linspace(-.1,.1,len(y)),y,c='black',s=20)
    ax.set(ylabel='TCP trajectory RMSE mm',title='All frozen test cases / both independent PPO training seeds');fig.savefig(ROOT/'assets/plots/phase3dc_final_comparison.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for i,m in enumerate(('A','B')):
        r=next(t for t in training if t['stage']=='initial' and t['method']==m and t['seed']==42);folder=Path(r['run_directory'])/'actual_training/episode_0000';t=csvrows(folder/'tracking.csv');x=csvrows(folder/'dc_control.csv');times=[float(v['time']) for v in t]
        axs[0,i].plot(times,[float(v['tcp_error_m'])*1000 for v in t],label='TCP error mm');axs[0,i].plot(times,[float(v['minimum_clearance_m'])*1000 for v in t],label='gap mm');axs[0,i].axhline(5,ls='--',c='red',label='5mm failure')
        for v in read(Path(r['run_directory'])/'complete.json')['updates']:
            step=v['decision_steps']*25+750
            if step<len(t):axs[0,i].axvline(float(t[step-1]['time']),c='gray',ls=':')
        for key,label in [('integral20_request','integral20 target'),('rl_correction_target','RL correction'),('combined_target','combined target'),('applied','ACTUAL applied')]:axs[1,i].plot(times,[json.loads(v[key])[1]*1000 for v in x],label=label)
        axs[0,i].set(title=f'{m} seed42 actual first stochastic episode',xlabel='simulation s');axs[1,i].set(xlabel='simulation s',ylabel='joint2 residual mrad')
        axs[0,i].legend(fontsize=8);axs[1,i].legend(fontsize=8)
    fig.savefig(ROOT/'assets/plots/phase3dc_action_tcp_clearance.png',dpi=150);plt.close(fig)

if __name__=='__main__':main_guard(main)
