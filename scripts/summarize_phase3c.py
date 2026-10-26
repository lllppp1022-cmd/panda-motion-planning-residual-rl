"""Read-only numerical re-audit and Chinese Phase3C reports from actual artifacts."""
from _bootstrap import ROOT,main_guard
from pathlib import Path
import csv,json,math
import numpy as np
from src.common import write_json,write_csv,sha
from scripts.phase3c_common import read,rows
from src.experiments.phase3c import reward_r1,design


def latest(name):return Path(read(ROOT/f'results/phase3c_{name}_latest.json')['run'])
def vector(records,key):return np.array([[float(r[f'{key}_{j}']) for j in range(1,8)] for r in records])
def yn(x):return str(x)=='True'
def mdtable(headers,records):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in records)


def audit_stage(g):
    """Recompute both rewards, physical counts and actual control constraints."""
    folder=Path(g['run_directory']);samples=rows(folder/'training_samples.csv');items=[];decisions=[];offset=0
    for ep in sorted((folder/'actual_training').glob('episode_*')):
        t=rows(ep/'tracking.csv');u=rows(ep/'control_signals.csv');diag=rows(ep/'phase3c_physics_diagnostics.csv')
        d=rows(ep/'decisions.csv');c=rows(ep/'phase3c_decisions.csv')
        assert len(t)==len(u)==len(diag)
        assert len(d)==len(c)
        qerr=vector(t,'q_actual')-vector(t,'q_ref');ctrl=vector(t,'ctrl');applied=vector(u,'residual_rad');target=vector(u,'residual_target_rad')
        initial=np.array(read(ep/'initialization.json')['q']);ctrlrate=np.max(abs(np.diff(np.vstack([initial,ctrl]),axis=0))/.002)
        # Protective stop resets residual bookkeeping; separately preserve that discontinuity.
        phases=np.array([r['phase']!='protective_hold' for r in t]);diff=np.diff(np.vstack([np.zeros(7),applied]),axis=0)/.002
        active=phases&np.r_[True,phases[:-1]]
        residualrate=float(np.max(abs(diff[active]))) if np.any(active) else 0.
        reward_error=0.;selected_error=0.;previous=750;old_action=np.zeros(7);old_progress=0.
        duration=read(ep/'frozen_run_config.json')['trajectory_duration_s']
        for a,b in zip(d,c):
            end=int(a['physics_step']);segment=t[previous:end];action=np.array(json.loads(a['action']))
            actual=[{k:(float(v) if k not in ('phase','safety_valid') else v) for k,v in x.items()} for x in segment]
            cfg=read(ROOT/'configs/phase3b_smoke.json')['reward']
            errors=vector(segment,'q_actual')-vector(segment,'q_ref');verrors=vector(segment,'dq_actual')-vector(segment,'dq_ref')
            r0parts=dict(joint=-cfg['joint_max_penalty']*min(1.,float(np.mean((errors/.01)**2))),
                velocity=-cfg['velocity_max_penalty']*min(1.,float(np.mean((verrors/.1)**2))),
                action=-cfg['action_penalty']*float(np.mean(action**2)),
                smoothness=-cfg['smoothness_max_penalty']*min(1.,float(np.mean((action-old_action)**2))),
                time=-cfg['time_penalty'],progress=cfg['progress_total']*max(0.,float(segment[-1]['progress'])-old_progress),
                success=float(a['reward_success']),failure=float(a['reward_failure']),contact=float(a['reward_contact']))
            assert all(abs(v-float(a['reward_'+k]))<1e-9 for k,v in r0parts.items())
            total0=sum(r0parts.values())
            terminated=yn(a['terminated']);truncated=yn(a['truncated']);success=float(a['reward_success'])>0
            r1,p=reward_r1(actual,action,old_action,old_progress,duration,success,terminated or truncated,float(a['reward_contact'])<0)
            assert abs(total0-float(b['R0_reward']))<1e-9
            reward_error=max(reward_error,abs(r1-float(b['R1_reward'])),*(abs(p[k]-float(b['R1_'+k])) for k in p))
            sample=samples[offset];selected=total0 if g['version']=='R0' else r1
            selected_error=max(selected_error,abs(selected-float(sample['reward'])))
            assert np.allclose(np.clip(json.loads(sample['raw_sample_action']),-1,1),action,rtol=0,atol=1e-8)
            assert np.allclose(vector(u[previous:end],'residual_target_rad')[0],action*.002,rtol=0,atol=1e-9)
            decisions.append(dict(version=g['version'],seed=g['seed'],decision=int(sample['decision']),episode=ep.name,
                reward=selected,reward_R0=total0,reward_R1=r1,**{'R0_'+k:v for k,v in r0parts.items()},**{'R1_'+k:v for k,v in p.items()}))
            offset+=1;previous=end;old_action=action;old_progress=float(segment[-1]['progress'])
        if len(t)>750:
            assert np.isfinite(qerr).all() and np.isfinite(ctrl).all()
        item=dict(episode=ep.name,physics_rows=len(t),decisions=len(d),reset_steps=750,
            max_actual_residual_rad=float(np.max(abs(applied))),max_target_residual_rad=float(np.max(abs(target))),
            max_active_residual_rate_rad_s=residualrate,max_ctrl_rate_rad_s=float(ctrlrate),
            residual_target_applied_rmse_rad=float(np.sqrt(np.mean((target-applied)**2))),
            residual_slew_steps=sum(yn(x['residual_slew_intervened']) for x in diag),
            effective_total_command_slew_steps=sum(yn(x['total_command_slew_intervened']) for x in diag),
            force_saturation_steps=sum(yn(x['force_saturated']) for x in diag),
            minimum_clearance_m=min(float(x['minimum_clearance_m']) for x in t),
            forbidden_contact_observations=sum(int(x['prohibited_contact_count']) for x in t),
            tcp_rmse_m=float(np.sqrt(np.mean([float(x['tcp_error_m'])**2 for x in t]))),
            joint_rmse_rad=float(np.sqrt(np.mean(qerr**2))),reward_recompute_max_abs_error=reward_error,
            float32_sample_reward_max_abs_error=selected_error)
        assert item['max_actual_residual_rad']<=.002+1e-12
        assert residualrate<=.01+1e-9 and ctrlrate<=.35+1e-9
        assert reward_error<1e-9 and selected_error<2e-5
        items.append(item)
    assert len(samples)==offset==g['stage_decisions']
    assert sum(x['physics_rows'] for x in items)==g['stage_total_physics']
    assert g['stage_training_physics']+g['stage_reset_physics']+g['stage_cleanup_physics']==g['stage_total_physics']
    buffers=[]
    # Independent internal GAE recurrence; the last bootstrap is audited using its saved target,
    # rather than pretending an inferred bootstrap is an independently predicted critic value.
    for file in sorted(folder.glob('rollout_*.npz')):
        z=np.load(file);a=z['advantages'].ravel();v=z['values'].ravel();r=z['rewards'].ravel();starts=z['episode_starts'].ravel()
        mismatch=[]
        for i in range(len(a)-2,-1,-1):
            live=1-starts[i+1];expected=r[i]+.995*v[i+1]*live-v[i]+.995*.95*live*a[i+1]
            mismatch.append(abs(float(a[i])-float(expected)))
        assert np.max(mismatch)<3e-5
        assert np.allclose(z['returns'],z['advantages']+z['values'],atol=2e-5,rtol=0)
        buffers.append(dict(rollout=file.name,gae_internal_max_abs_error=float(np.max(mismatch)),
            last_bootstrap_independently_verified=False,advantage_mean=float(a.mean()),advantage_std=float(a.std()),
            advantage_min=float(a.min()),advantage_max=float(a.max()),target_max=float(z['returns'].max())))
    return items,decisions,buffers


def history_report():
    audit=read(latest('3b_audit')/'complete.json');updates=read(latest('3b_updates')/'complete.json')['results'];effects=rows(ROOT/'results/phase3c_action_effectiveness.csv')
    values=rows(ROOT/'results/phase3c_reward_value_audit.csv');means=np.array([json.loads(r['actor_mean']) for r in values]);std=np.array([json.loads(r['actor_std']) for r in values])
    text='# Phase3B真实训练诊断\n\n日期：2026-10-22。依据原1024决策、物理CSV与更新前后checkpoint重新计算；历史文件未覆盖。\n\n'
    text+='## Value Loss约1460的来源\n\n第二个rollout第一次包含真正到达后的+200终止奖励。该奖励沿GAE反传（γλ=0.995×0.95=0.94525），Critic尚未学到这个稀疏大目标，产生约1463的更新前均方误差。记录的Value Loss是5个epoch、20个minibatch的均值，不能与更新前整块MSE混为一个数。\n\n'
    text+=mdtable(['更新','优化前目标MSE','日志Value Loss','EV','优势均值/标准差','最大优势','最大价值目标'],[[u['update'],f"{u['preoptimization_target_mse']:.6f}",f"{u['logged_mean_minibatch_value_loss']:.6f}",f"{u['explained_variance_recomputed']:.6f}",f"{u['advantage_mean']:.5f}/{u['advantage_std']:.5f}",f"{u['advantage_max']:.5f}",f"{u['max_value_target']:.5f}"] for u in audit['updates']])
    text+='\n四次原优化使用重构rollout、原优化器状态和原seed42排列离线重放：每次Value Loss差值及全部网络参数最大差值均为0。没有新增在线采样，也未写回原模型。当前证据支持“终止奖励尺度大且价值网络拟合不足”，不支持“Critic已发散”。EV始终接近0，说明预测能力仍弱，不能以参数更新代替学习有效性。\n\n'
    text+='## 逐决策奖励、值与动作\n\n'
    text+=f"已重算{audit['recomputed_decisions']}个决策的全部R0分量并与CSV逐项匹配；在相同真实轨迹离线重算R1。逐步奖励标准差R0={audit['reward_R0_std']:.6f}、R1={audit['reward_R1_std']:.6f}。完整结果见`results/phase3c_reward_value_audit.csv`。\n\n"
    text+='GAE采用每块优化前Critic、真实episode_start/done以及末观察值bootstrap；价值目标为优势+旧值。完整回合计算实际折扣回报；最后一个未完成回合仅报告已观察的折扣前缀，不能当作完整Monte Carlo回报。优化前目标与优化后预测的统计口径明确保留在重构NPZ及complete.json。\n\n'
    text+=f"Actor均值全维范围[{means.min():.7f},{means.max():.7f}]，高斯标准差范围[{std.min():.7f},{std.max():.7f}]（归一化动作）。训练使用真实随机采样，评价使用确定性均值。±1裁剪后乘0.002rad，再经过0.01rad/s残差限速和0.35rad/s总目标限速；原始采样、裁剪目标和实际补偿不是同一变量。\n\n"
    text+=f"残差限速介入{audit['residual_slew_steps']}个物理步，有效总指令限速介入{audit['total_command_slew_steps']}步。旧摘要`command_rate_limit_steps`比较q_ref与含重力前馈的ctrl，因此会把正常补偿也计入，不能解释为真实限速介入次数。本轮根据delayed_target与前次ctrl差重算，并新增真实控制器rate_limited日志；不改变任何控制行为。\n\n"
    text+=f"逐决策目标/实际残差RMSE最大{max(float(r['target_applied_rmse_rad']) for r in effects):.8f}rad。完整数据见`results/phase3c_action_effectiveness.csv`，包括每决策真实TCP与关节误差。\n\n"
    text+='## 每次更新前后独立确定性评价\n\n同一历史验证路径、相同目标与物理种子；0和4复用冻结的真实评价，1～3各补做两次，共6次新增实际执行。以下为Evaluation Replay，绝不是训练随机rollout。\n\n'
    text+=mdtable(['更新后','条件','TCP轨迹RMSE(mm)','最终TCP(mm)','成功','复用旧评价'],[[x['update'],x['condition']['name'],f"{x['result']['tcp_rmse_m']*1000:.6f}",f"{x['result']['final_tcp_error_m']*1000:.6f}",x['result']['success'],x['reused_frozen_actual_evaluation']] for x in sorted(updates,key=lambda x:(x['condition']['name'],x['update']))])
    text+='\n微小的正负变化均保留，不能据此宣称PPO优于M3。R0在当前亚毫米误差下跟踪项很弱，+200终止项主导价值学习；R1改成毫米敏感尺度与较小终止奖励，但是否改善真实控制必须由新Pilot检验。\n\n'
    text+=f"原始诊断：`{latest('3b_audit')}`；逐更新评价：`{latest('3b_updates')}`。\n"
    (ROOT/'docs/22_Phase3B训练诊断.md').write_text(text,encoding='utf-8')


def evaluation_table(stage):
    records=read(latest('eval_'+stage)/'complete.json')['results'];out=[]
    for r in records:
        parameters=read(Path(r['run_directory'])/'model_parameters.json');frozen=read(Path(r['run_directory'])/'frozen_run_config.json')
        assert parameters['seed']==9001 and parameters['perturbation']==r['condition']
        assert frozen['physics_dt_s']==.002 and frozen['monitor_frequency_hz']==500
        assert frozen['path_evidence']['source_json']==r['source']
        t=rows(Path(r['run_directory'])/'tracking.csv');motion=[x for x in t if x['phase'] in ('trajectory','settle')]
        qe=vector(motion,'q_actual')-vector(motion,'q_ref')
        tcp=float(np.sqrt(np.mean([float(x['tcp_error_m'])**2 for x in motion])))
        assert abs(tcp-r['tcp_rmse_m'])<1e-12
        joint=float(np.sqrt(np.mean(qe**2)));assert np.allclose(np.sqrt(np.mean(qe**2,axis=0)),r['joint_rmse_rad'],atol=1e-12)
        minimum=min(float(x['minimum_clearance_m']) for x in t);assert abs(minimum-r['minimum_actual_clearance_m'])<1e-12
        assert len(t)==r['physics_steps'] and r['live_qpos_writes_after_initialization']==0
        signal=Path(r['run_directory'])/('integral_control_signals.csv' if r['method']=='Integral' else 'control_signals.csv')
        sig=rows(signal);forces=vector(sig,'actuator_force');maxforce=float(np.max(abs(forces)))
        arm=frozen['actuators']['arm_joints'];native=np.array([a['forcerange'] for a in arm]);force_bounds=np.max(abs(native),axis=1)
        assert np.all(forces>=native[:,0]-1e-8) and np.all(forces<=native[:,1]+1e-8)
        assert parameters['nominal']['actuator_forcerange']==parameters['physics']['actuator_forcerange']
        limits=np.array([a['range_rad'] for a in arm]);qactual=vector(t,'q_actual');ctrl=vector(t,'ctrl')
        assert np.all(qactual>=limits[:,0]) and np.all(qactual<=limits[:,1])
        assert np.all(ctrl>=limits[:,0]) and np.all(ctrl<=limits[:,1])
        checkpoint=r['checkpoint']
        out.append(dict(split=stage,scene=r['scene'],condition=r['condition']['name'],method=r['method'],physical_seed=9001,
            success=r['success'],termination_reason=r['failure_reason'],completion_progress=r['completion_progress'],
            safety_violation_steps=r['safety_violation_steps'],forbidden_contact_observations=r['prohibited_contact_observations'],contact_intervals=r['contact_intervals'],
            tcp_rmse_mm=tcp*1000,joint_rmse_rad=joint,final_tcp_mm=r['final_tcp_error_m']*1000,
            full_record_tcp_rmse_mm=float(np.sqrt(np.mean([float(x['tcp_error_m'])**2 for x in t])))*1000,
            full_record_joint_rmse_rad=float(np.sqrt(np.mean((vector(t,'q_actual')-vector(t,'q_ref'))**2))),
            tracking_metric_physics_steps=len(motion),
            minimum_clearance_mm=minimum*1000,maximum_actual_residual_rad=r['maximum_actual_residual_rad'],
            command_smoothness_mean_squared_rate=r['command_smoothness_mean_squared_rate'],command_max_rate_rad_s=r['command_max_rate_rad_s'],
            max_actual_velocity_rad_s=r['actual_max_velocity_rad_s'],max_actual_acceleration_rad_s2=r['actual_max_acceleration_fd_rad_s2'],
            actual_physics_steps=len(t),execution_simulation_s=r['simulation_time_s'],max_native_force_nm=maxforce,
            native_force_saturation_steps=int(np.sum(np.any(abs(forces)>=force_bounds-1e-8,axis=1))),
            nominal_signature=parameters['nominal_signature'],physics_signature=parameters['physics_signature'],
            path_csv_sha256=frozen['path_evidence']['path_csv_sha256'],
            checkpoint=checkpoint or '',checkpoint_sha256=sha(checkpoint) if checkpoint else '',run_directory=r['run_directory']))
    assert len(out)==21
    assert len({(x['method'],x['condition']) for x in out})==21
    for condition in ('nominal','gain90','mass110'):
        case=[x for x in out if x['condition']==condition]
        assert len({x['physics_signature'] for x in case})==len({x['nominal_signature'] for x in case})==len({x['path_csv_sha256'] for x in case})==1
    return out


def probe_state_effects():
    records=read(latest('validation')/'complete.json')['results'];out=[]
    for scene in ('c_train_sphere','c_validation_cylinder'):
        base=next(x for x in records if x['scene']==scene and x['method']=='M3');b=rows(Path(base['run_directory'])/'tracking.csv')
        for method in ('FeedbackProbe','CounterProbe'):
            candidate=next(x for x in records if x['scene']==scene and x['method']==method);c=rows(Path(candidate['run_directory'])/'tracking.csv')
            assert len(b)==len(c)
            for before,after in zip(b,c):
                out.append(dict(scene=scene,condition='gain90',method=method,physics_step=int(before['physics_step']),
                    phase=before['phase'],progress=float(before['progress']),
                    zero_tcp_error_mm=float(before['tcp_error_m'])*1000,nonzero_tcp_error_mm=float(after['tcp_error_m'])*1000,
                    tcp_error_change_mm=(float(after['tcp_error_m'])-float(before['tcp_error_m']))*1000,
                    **{f'zero_q_{j}':float(before[f'q_actual_{j}']) for j in range(1,8)}))
    write_csv(ROOT/'results/phase3c_probe_state_effects.csv',out)
    summary=[]
    for scene in ('c_train_sphere','c_validation_cylinder'):
        for method in ('FeedbackProbe','CounterProbe'):
            for phase in sorted({x['phase'] for x in out}):
                group=[x for x in out if x['scene']==scene and x['method']==method and x['phase']==phase]
                if not group:continue
                summary.append(dict(scene=scene,method=method,phase=phase,physics_steps=len(group),
                    smaller_error_fraction=sum(x['tcp_error_change_mm']<0 for x in group)/len(group),
                    mean_tcp_error_change_mm=float(np.mean([x['tcp_error_change_mm'] for x in group]))))
    write_csv(ROOT/'results/phase3c_probe_effects_by_phase.csv',summary)


def main():
    history_report();probe_state_effects();gate=read(ROOT/'results/phase3c_2048_gate.json');stages=['initial','continue'] if gate['all_four_passed'] else ['initial']
    allstages={s:read(latest('training_'+s)/'complete.json')['results'] for s in stages};training=[];raw_rewards=[];audits=[]
    for version in ('R0','R1'):
        for seed in (42,123):
            groups=[next(x for x in allstages[s] if x['version']==version and x['seed']==seed) for s in stages]
            episode_audits=[];updates=[]
            for s,g in zip(stages,groups):
                a,d,b=audit_stage(g);episode_audits.extend(a);raw_rewards.extend(d);updates.extend(g['updates'])
                audits.append(dict(version=version,seed=seed,stage=s,episodes=a,buffers=b))
            end=groups[-1];row=dict(reward=version,training_seed=seed,decision_steps=sum(g['stage_decisions'] for g in groups),
                decision_physics_steps=sum(g['stage_training_physics'] for g in groups),reset_physics_steps=sum(g['stage_reset_physics'] for g in groups),
                cleanup_physics_steps=sum(g['stage_cleanup_physics'] for g in groups),total_physics_steps=sum(g['stage_total_physics'] for g in groups),
                completed_successful_episodes=sum(len(g['completed_episodes']) for g in groups),budget_interrupted_episodes=sum(len(g['episode_results'])-len(g['completed_episodes']) for g in groups),
                ppo_updates=len(updates),adam_optimizer_steps=len(updates)*20,
                actor_parameter_delta_l2_last_stage=end['actor_delta_l2'],critic_parameter_delta_l2_last_stage=end['critic_delta_l2'],
                final_value_loss=updates[-1]['value_loss'],final_explained_variance=updates[-1]['explained_variance'],
                max_approx_kl=max(u['approx_kl'] for u in updates),maximum_actual_residual_rad=max(x['max_actual_residual_rad'] for x in episode_audits),
                max_active_residual_rate_rad_s=max(x['max_active_residual_rate_rad_s'] for x in episode_audits),
                residual_slew_steps=sum(x['residual_slew_steps'] for x in episode_audits),effective_total_command_slew_steps=sum(x['effective_total_command_slew_steps'] for x in episode_audits),
                native_force_saturation_steps=sum(x['force_saturation_steps'] for x in episode_audits),
                minimum_actual_clearance_mm=min(x['minimum_clearance_m'] for x in episode_audits)*1000,
                training_safety_violation_steps=sum(r['safety_violation_steps'] for g in groups for r in g['episode_results']),
                training_forbidden_contact_observations=sum(r['prohibited_contact_observations'] for g in groups for r in g['episode_results']),
                checkpoint=end['checkpoint'],checkpoint_sha256=end['checkpoint_sha256'],run_directory=end['run_directory'])
            sampled=[r for g in groups for r in rows(Path(g['run_directory'])/'training_samples.csv')]
            mean=np.array([json.loads(r['actor_mean']) for r in sampled]);std=np.array([json.loads(r['actor_std']) for r in sampled])
            random=np.array([json.loads(r['raw_sample_action']) for r in sampled]);clipped=np.array([json.loads(r['normalized_executed_action']) for r in sampled])
            row.update(actor_mean_rms_normalized=float(np.sqrt(np.mean(mean**2))),gaussian_std_min_normalized=float(std.min()),gaussian_std_max_normalized=float(std.max()),
                sampled_action_rms_normalized=float(np.sqrt(np.mean(random**2))),clipped_action_fraction=float(np.mean(np.any(random!=clipped,axis=1))),
                target_applied_residual_rmse_rad=float(np.sqrt(np.average([x['residual_target_applied_rmse_rad']**2 for x in episode_audits],weights=[x['physics_rows'] for x in episode_audits]))))
            assert row['decision_steps']==end['cumulative_decisions'] and sha(end['checkpoint'])==end['checkpoint_sha256']
            training.append(row)
    evaluation=evaluation_table('validation')+evaluation_table('final')
    write_csv(ROOT/'results/phase3c_training_comparison.csv',training);write_csv(ROOT/'results/phase3c_evaluation_comparison.csv',evaluation)
    write_csv(ROOT/'results/phase3c_training_reward_components.csv',raw_rewards)
    distributions=[]
    for version in ('R0','R1'):
        selected=[r for r in raw_rewards if r['version']==version]
        for key in selected[0]:
            if key.startswith(('R0_','R1_')) or key=='reward':
                v=np.array([r[key] for r in selected],dtype=float)
                distributions.append(dict(training_reward=version,component=key,count=len(v),mean=float(v.mean()),std=float(v.std()),
                    minimum=float(v.min()),p05=float(np.quantile(v,.05)),median=float(np.median(v)),p95=float(np.quantile(v,.95)),maximum=float(v.max())))
    write_csv(ROOT/'results/phase3c_reward_component_distributions.csv',distributions)
    write_json(ROOT/'results/phase3c_raw_log_reaudit.json',dict(passed=True,stages=audits,
        caveat='NPZ内部GAE递推独立复算；末bootstrap未声称独立验证。Phase3B完整bootstrap与优化器已逐参数精确复放。'))
    plot(allstages,evaluation);tensorboard_diagnostics(allstages);report(training,evaluation,allstages)
    write_json(ROOT/'results/phase3c_acceptance.json',dict(completed=True,training=training,evaluation=evaluation,
        gate=gate,final_opening=read(latest('eval_final')/'final_test_opening.json'),
        no_further_training=True,checkpoint_selection='fixed last common allowed budget, no seed filtering'))
    print('REPORTS COMPLETE',sum(x['total_physics_steps'] for x in training),flush=True)


def plot(allstages,evaluation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    for s,groups in allstages.items():
        for g in groups:
            u=g['updates'];label=f"{g['version']} seed{g['seed']}";color={'R0':'tab:blue','R1':'tab:orange'}[g['version']];style='-' if g['seed']==42 else '--'
            x=[v['decision_steps'] for v in u]
            axes[0,0].plot(x,[v['value_loss'] for v in u],style,color=color,label=label if s=='initial' else None)
            axes[0,1].plot(x,[v['explained_variance'] for v in u],style,color=color)
            samples=rows(Path(g['run_directory'])/'training_samples.csv');vals=np.array([float(r['reward']) for r in samples]);xx=np.array([int(r['decision']) for r in samples])
            index=0 if g['version']=='R0' else 1
            axes[1,index].plot(xx[63::64],np.convolve(vals,np.ones(64)/64,mode='valid')[::64],style,color=color,alpha=.8)
    axes[0,0].set(title='Critic loss (different reward scales)',yscale='log',ylabel='Value loss');axes[0,0].legend()
    axes[0,1].set(title='Explained variance',ylabel='EV')
    axes[1,0].set(title='R0 sampled reward, 64-decision moving mean',ylabel='R0 reward')
    axes[1,1].set(title='R1 sampled reward, 64-decision moving mean',ylabel='R1 reward')
    for a in axes.flat:
        a.set_xlabel('Actual PPO decision steps');a.grid(alpha=.25);a.axvline(2048,color='gray',linestyle=':',alpha=.6)
    path=ROOT/'assets/plots/phase3c_learning_curves.png';path.parent.mkdir(exist_ok=True);fig.savefig(path,dpi=150);plt.close(fig)
    methods=['M1','M3','Integral','R0_seed42','R0_seed123','R1_seed42','R1_seed123'];fig,axes=plt.subplots(1,2,figsize=(13,4),constrained_layout=True)
    for ax,split in zip(axes,['validation','final']):
        for i,cond in enumerate(['nominal','gain90','mass110']):
            vals=[next(r['tcp_rmse_mm'] for r in evaluation if r['split']==split and r['method']==m and r['condition']==cond) for m in methods]
            ax.bar(np.arange(7)+(i-1)*.25,vals,width=.25,label=cond)
        ax.set_xticks(np.arange(7),methods,rotation=35,ha='right');ax.set(title=split+' deterministic actual execution',ylabel='TCP trajectory RMSE (mm)');ax.legend();ax.grid(axis='y',alpha=.2)
    fig.savefig(ROOT/'assets/plots/phase3c_control_comparison.png',dpi=150);plt.close(fig)


def tensorboard_diagnostics(allstages):
    from torch.utils.tensorboard import SummaryWriter
    for stage,groups in allstages.items():
        for g in groups:
            sample=rows(Path(g['run_directory'])/'training_samples.csv')
            with SummaryWriter(log_dir=str(Path(g['run_directory'])/'tensorboard/recomputed_diagnostics')) as writer:
                for i in range(0,len(sample),256):
                    part=sample[i:i+256];step=int(part[-1]['decision'])
                    for key in ('reward','R0_reward','R1_reward'):
                        values=np.array([float(x[key]) for x in part]);writer.add_scalar('actual_logged/'+key+'_mean',float(values.mean()),step)
                        writer.add_scalar('actual_logged/'+key+'_std',float(values.std()),step)
                    for key in part[0]:
                        if key.startswith('reward_'):writer.add_scalar('actual_logged/'+key+'_mean',float(np.mean([float(x[key]) for x in part])),step)
                for u in g['updates']:
                    for key in ('advantage_mean','advantage_std','advantage_min','advantage_max','target_min','target_max','preoptimization_value_mse'):
                        writer.add_scalar('saved_rollout/'+key,u[key],u['decision_steps'])


def report(training,evaluation,allstages):
    text='# Phase3C有限Pilot验收报告\n\n日期2026-10-22。完成后暂停；未启动100k PPO决策训练。以下所有数值来自真实动力学CSV及checkpoint，不修改Phase3B历史。\n\n'
    text+='## 运行范围与共同门禁\n\n四组R0/R1×42/123均先执行2048决策；中期两个失配条件分别确定性评价，完整记录见`results/phase3c_2048_gate.json`。只按预先共同门禁决定是否四组一起继续，最后checkpoint按共同预算固定，不按最终测试挑选。\n\n'
    text+=mdtable(['奖励','种子','PPO决策','真实积分步','其中reset/cleanup','成功完整回合/预算中断','更新/Adam步','末Value Loss','末EV'],[[r['reward'],r['training_seed'],r['decision_steps'],r['total_physics_steps'],f"{r['reset_physics_steps']}/{r['cleanup_physics_steps']}",f"{r['completed_successful_episodes']}/{r['budget_interrupted_episodes']}",f"{r['ppo_updates']}/{r['adam_optimizer_steps']}",f"{r['final_value_loss']:.6f}",f"{r['final_explained_variance']:.6f}"] for r in training])
    text+=f"\n训练合计{sum(r['decision_steps'] for r in training)}个PPO决策、{sum(r['total_physics_steps'] for r in training)}个真实MuJoCo积分步；包含reset静态保持、决策执行、各阶段关闭时保护保持。每组阶段边界未结束的回合标记为预算中断，不当作到达成功，也不伪造失败的完整回报。继续阶段从保存的网络、Adam、Python/NumPy/Torch随机状态恢复，环境明确重置并推进工况序号，不伪称无缝跨episode继续。\n\n"
    text+='网络两支每次均更新，所有rollout NPZ含旧值、GAE、价值目标、原始动作及logprob；checkpoint、TensorBoard、参数SHA、随机状态、完整CSV均保存。模型的确定性评价仍需读入机器人状态，不读取实际质量、增益或惯量。\n\n'
    text+='## 新场景与隔离\n\n新训练sphere、新验证cylinder、新最终box的障碍布局和起终点均不同，RRT路径均0.005rad复核，规划间隙20mm、执行门槛5mm；不是历史heldout_box更名。名义、gain90、mass110单因素共三工况，名义控制器不读真实扰动参数。最终测试开启前写入冻结方法、配置、四个checkpoint哈希及验证路径；没有根据最终结果再选奖励或模型。这里只覆盖每split一个几何任务与三工况，不代表广泛泛化。\n\n'
    text+='## 真实确定性评价\n\n各方法在每split三工况配对比较，物理seed9001；传统基线不重复当作两个独立随机种子。主TCP/关节RMSE与原执行器口径一致，包含trajectory和settle阶段；另列full_record指标纳入prehold和保护保持。安全、接触和实际积分数量统计全记录，所有失败也保留，完成进度与终止原因另列CSV。未用仅成功回合过滤误差。\n\n'
    methods=['M1','M3','Integral','R0_seed42','R0_seed123','R1_seed42','R1_seed123']
    for split in ('validation','final'):
        text+='### '+split+'\n\n'
        text+=mdtable(['方法','nominal TCP RMSE(mm)','gain90 TCP RMSE(mm)','mass110 TCP RMSE(mm)','稳定到达','5mm违规/禁止接触'],[[m,*[f"{next(x['tcp_rmse_mm'] for x in evaluation if x['split']==split and x['method']==m and x['condition']==c):.6f}" for c in ('nominal','gain90','mass110')],f"{sum(x['success'] for x in evaluation if x['split']==split and x['method']==m)}/3",f"{sum(x['safety_violation_steps'] for x in evaluation if x['split']==split and x['method']==m)}/{sum(x['forbidden_contact_observations'] for x in evaluation if x['split']==split and x['method']==m)}"] for m in methods])
        text+='\n'
    text+='## 对研究问题的回答\n\n'
    for version in ('R0','R1'):
        for seed in (42,123):
            pairs=[]
            for r in evaluation:
                if r['method']==f'{version}_seed{seed}' and r['condition']!='nominal':
                    base=next(x for x in evaluation if x['split']==r['split'] and x['condition']==r['condition'] and x['method']=='M3')
                    integ=next(x for x in evaluation if x['split']==r['split'] and x['condition']==r['condition'] and x['method']=='Integral')
                    pairs.append((100*(base['tcp_rmse_mm']-r['tcp_rmse_mm'])/base['tcp_rmse_mm'],r['tcp_rmse_mm']<integ['tcp_rmse_mm']))
            text+=f"- {version} seed{seed}：验证/最终共4个失配条件相对M3的TCP RMSE变化（正值改善）为{', '.join(f'{v[0]:+.3f}%' for v in pairs)}；超过积分基线{sum(v[1] for v in pairs)}/4。\n"
    robust={v:all(next(r for r in evaluation if r['method']==f'{v}_seed{s}' and r['split']==split and r['condition']==c)['tcp_rmse_mm']<next(r for r in evaluation if r['method']=='M3' and r['split']==split and r['condition']==c)['tcp_rmse_mm'] for s in (42,123) for split in ('validation','final') for c in ('gain90','mass110')) for v in ('R0','R1')}
    text+=f"\n严格按两个种子、两个几何任务及两个失配条件均改善的描述性条件：{robust}。这不是显著性检验，也未预先设定最小有用效应，微小变化不能等同工程收益。\n\n"
    text+='R1的毫米敏感跟踪项和较小终止奖励更便于解释价值目标，旧+200主导的尺度问题已改善；更小的Value Loss不能作为R1控制优越的证据。受限积分基线保留M3前馈，ki=0.3/s、±0.002rad、0.01rad/s，抗饱和与延迟观测均独立记录；同一套参数先通过12次动作/传统控制门禁，最终测试后未调参。PPO必须同时超过M3和这个传统基线才有充分的额外收益证据，不能只挑某一工况或一粒种子。\n\n'
    text+=f"训练5mm违规{sum(r['training_safety_violation_steps'] for r in training)}物理步、禁止接触观察{sum(r['training_forbidden_contact_observations'] for r in training)}次；42次评价中违规回合{sum(r['safety_violation_steps']>0 for r in evaluation)}，禁止接触回合{sum(r['forbidden_contact_observations']>0 for r in evaluation)}。全部实际残差最大{max(r['maximum_actual_residual_rad'] for r in training):.8f}rad，未扩大探索边界；新动作探针中反馈降低误差、反向反馈增大误差，说明通道有效，但不构成全状态安全证明。\n\n"
    text+='## 当前决策与限制\n\n不自动扩大训练预算。只有在预先冻结的新失配任务上出现跨种子的、相对强传统控制器可重复且有实际意义的收益，同时维持安全条件，才值得进入下一阶段。当前有限Pilot只能检验这批安全较宽、轻度失配任务；未研究强组合扰动、时延/噪声泛化、更广场景，也不声称统计显著、连续碰撞安全或实机安全认证。\n\n'
    text+='原173项与新增回归必须实际执行，测试详情另见`results/phase3c_tests.xml`和`results/phase3c_tests.log`；日志生成脚本不会把尚未运行的测试写成通过。`results/phase3c_raw_log_reaudit.json`独立复算训练奖励、内部GAE递推、动作映射、残差/总指令限速、积分步、评价RMSE、实际间隙及力矩约束。新NPZ末bootstrap未单独重建Critic预测，未虚称这一项独立精确复算；Phase3B四块bootstrap和优化器已完整逐参数复放。\n\n'
    text+='## 运行与查看\n\n以下为本轮实际使用的入口，重新运行会生成独立目录；训练命令会启动新实验，查看现有结果请先使用TensorBoard/报告。\n\n```powershell\ncd D:\\mujoco_learning\\panda_safe_residual_rl\n& D:\\miniconda\\envs\\panda_safe_rl\\python.exe scripts/train_phase3c.py --stage initial\n& D:\\miniconda\\envs\\panda_safe_rl\\python.exe scripts/evaluate_phase3c.py --stage gate\n# 仅共同门禁全部通过才允许继续：\n& D:\\miniconda\\envs\\panda_safe_rl\\python.exe scripts/train_phase3c.py --stage continue\n& D:\\miniconda\\envs\\panda_safe_rl\\python.exe scripts/evaluate_phase3c.py --stage validation\n& D:\\miniconda\\envs\\panda_safe_rl\\python.exe scripts/evaluate_phase3c.py --stage final\n& D:\\miniconda\\envs\\panda_safe_rl\\python.exe scripts/summarize_phase3c.py\n& D:\\miniconda\\envs\\panda_safe_rl\\python.exe -m pytest -q\n```\n\n'
    text+='### 最终模型与真实日志\n\n'
    for r in training:text+=f"- {r['reward']} seed{r['training_seed']}：`{r['checkpoint']}`；SHA256 `{r['checkpoint_sha256']}`。同目录`tensorboard/`、`training_samples.csv`、`rollout_*.npz`及`actual_training/episode_*/`。\n"
    text+='\n结果索引：`results/phase3c_acceptance.json`；规划源：`results/phase3c_prepare_latest.json`；2048安全评价：`results/phase3c_eval_gate_latest.json`；验证/最终评价：`results/phase3c_eval_validation_latest.json`、`results/phase3c_eval_final_latest.json`。\n\n'
    text+='未执行：100k决策训练、更多随机种子、强组合域随机化、观测噪声/时延专项、实机部署。本阶段没有视频交付要求；未用确定性回放冒充随机训练。Phase3B真实试错观察系统及历史视频原样保留。\n'
    (ROOT/'docs/24_Phase3C验收报告.md').write_text(text,encoding='utf-8')


if __name__=='__main__':main_guard(main)
