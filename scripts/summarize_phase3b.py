"""Honest smoke report: interaction counts, failed candidate, paired evaluation."""
from _bootstrap import ROOT,main_guard
import csv,json,xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.common import write_json,sha
from scripts.validate_phase3b import read,trace,sources

def main():
    training=Path(read(ROOT/'results/phase3b_training_latest.json')['run']);tr=read(training/'complete.json');cfg=read(training/'frozen_training_config.json')
    for name,h in cfg['source_hashes'].items():
        if sha(ROOT/name)!=h:raise ValueError('Training source changed: '+name)
    prep=Path(read(ROOT/'results/phase3b_prepare_latest.json')['run']);manifest=read(prep/'history_sha256.json')
    changed=[name for name,h in manifest.items() if not (ROOT/name).exists() or sha(ROOT/name)!=h]
    if changed:raise ValueError('Historical artifact changed: '+str(changed))
    validation=Path(read(ROOT/'results/phase3b_eval_validation_latest.json')['run']);final=Path(read(ROOT/'results/phase3b_eval_final_latest.json')['run'])
    evaluations=read(validation/'complete.json')['results']+read(final/'complete.json')['results'];audit=read(ROOT/'results/phase3b_csv_audit.json')
    assert audit['passed'] and audit['evaluation_runs']==31
    tests=ET.parse(ROOT/'results/phase3b_release_tests.xml').getroot();suite=tests.find('testsuite') if tests.tag!='testsuite' else tests
    assert all(int(suite.attrib.get(k,0))==0 for k in ('failures','errors','skipped'));testcount=int(suite.attrib['tests'])
    samples=trace(training/'training_samples.csv');updates=read(training/'ppo_update_events.json')
    initial_safety=[]
    for run in sorted((ROOT/'results/runs').glob('phase3b_safety_*')):
        if (run/'complete.json').exists() and not read(run/'complete.json')['passed']:initial_safety.append(dict(run=str(run),record=read(run/'complete.json')))
    failed_probes=[r for e in initial_safety for r in e['record']['results'] if not r['success']]
    safety=read(ROOT/'results/phase3b_action_safety_tests.json');assert safety['passed']
    # Like-for-like main comparison: every M1/M3/final policy has the same nine tasks.
    groups=[]
    for method in ('M1','M3','M3_PPO'):
        rs=[r for r in evaluations if r['comparison_method']==method]
        groups.append(dict(method=method,runs=len(rs),success=sum(r['success'] for r in rs),
            violation_runs=sum(r['safety_violation_steps']>0 for r in rs),contact_runs=sum(r['prohibited_contact_observations']>0 for r in rs),
            median_tcp_rmse_m=float(np.median([r['tcp_rmse_m'] for r in rs])),
            median_final_tcp_error_m=float(np.median([r['final_tcp_error_m'] for r in rs])),
            max_final_tcp_error_m=max(r['final_tcp_error_m'] for r in rs)))
    pairs=[]
    for r in evaluations:
        if r['comparison_method']!='M3_PPO':continue
        b=next(x for x in evaluations if x['comparison_method']=='M3' and x['scene']==r['scene'] and x['condition']==r['condition'])
        pairs.append(dict(scene=r['scene'],condition=r['condition']['name'],M3_tcp_rmse_m=b['tcp_rmse_m'],PPO_tcp_rmse_m=r['tcp_rmse_m'],
            rmse_change_fraction=r['tcp_rmse_m']/b['tcp_rmse_m']-1,success_M3=b['success'],success_PPO=r['success']))
    initial_equals=[]
    for r in evaluations:
        if r['comparison_method']!='PPO_initial':continue
        b=next(x for x in evaluations if x['comparison_method']=='M3' and x['scene']==r['scene'] and x['condition']==r['condition'])
        old,new=trace(Path(b['run_directory'])/'tracking.csv'),trace(Path(r['run_directory'])/'tracking.csv')
        equal=len(old)==len(new) and all(x[f'q_actual_{j}']==y[f'q_actual_{j}'] and x[f'ctrl_{j}']==y[f'ctrl_{j}'] for x,y in zip(old,new) for j in range(1,8))
        initial_equals.append(dict(scene=r['scene'],condition=r['condition']['name'],bitwise_equal_to_M3=equal))
    assert all(x['bitwise_equal_to_M3'] for x in initial_equals)
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True);x=np.arange(1,len(samples)+1);reward=np.array([float(r['reward']) for r in samples])
    axes[0].plot(x,reward,lw=.7,label='Actual stochastic training decision rewards')
    axes[0].set_ylabel('Reward / decision');axes[0].legend();axes[0].grid(alpha=.25)
    per_episode=[]
    for folder in sorted((training/'actual_training').glob('episode_*')):
        dr=trace(folder/'decisions.csv')
        if dr and dr[-1]['terminated']=='True':per_episode.append(sum(float(r['reward']) for r in dr))
    for key in ('joint','velocity','action','smoothness','time','progress'):
        values=np.array([float(r['reward_'+key]) for r in samples]);smooth=np.convolve(values,np.ones(50)/50,mode='valid')
        axes[1].plot(x[49:],smooth,label=key)
    axes[1].set_ylabel('50-decision mean component');axes[1].set_xlabel('Actual PPO decision steps (20 Hz)');axes[1].legend(ncol=3);axes[1].grid(alpha=.25)
    for ax in axes:
        for u in updates:ax.axvline(u['decision_steps'],ls='--',color='gray',alpha=.4)
    fig.suptitle('Phase3B: 1024 real PPO decisions, four updates; goal bonuses are sparse, NOT convergence evidence')
    fig.tight_layout();(ROOT/'assets/plots').mkdir(exist_ok=True);fig.savefig(ROOT/'assets/plots/phase3b_reward_curve.png',dpi=150);plt.close(fig)
    summary=dict(training_run=str(training),training=tr,paired_evaluation=groups,paired_changes=pairs,initial_policy_equivalence=initial_equals,
        failed_original_candidate_runs=[r['run_directory'] for r in failed_probes],failed_candidate_count=len(failed_probes),
        corrected_gate_successes=sum(r['success'] for r in safety['results']),test_count=testcount,history_files_unchanged=len(manifest),
        conclusion='环境和烟雾训练基本验收通过；不能证明PPO超过M3，暂不具备直接大规模正式训练的证据',
        next_requirements=['multiple bounded training seeds','bounded integral/adaptive baseline','reward/observation analysis','more safety states and unseen tasks'],
        training_complete_episode_returns=per_episode,final_test_already_opened=True)
    write_json(ROOT/'results/phase3b_experiment_summary.json',summary)
    table='\n'.join(f"|{r['method']}|{r['success']}/{r['runs']}|{r['violation_runs']}|{r['contact_runs']}|{r['median_tcp_rmse_m']*1000:.6f}|{r['max_final_tcp_error_m']*1000:.6f}|" for r in groups)
    pairtable='\n'.join(f"|{r['scene']}|{r['condition']}|{r['M3_tcp_rmse_m']*1000:.6f}|{r['PPO_tcp_rmse_m']*1000:.6f}|{r['rmse_change_fraction']*100:+.3f}%|" for r in pairs)
    update_table='\n'.join(f"|{u['update']}|{u['decision_steps']}|{u['actor_parameter_delta_l2']:.6f}|{u['critic_parameter_delta_l2']:.6f}|{u['policy_gradient_loss']:.6f}|{u['value_loss']:.4f}|{u['approx_kl']:.7f}|" for u in updates)
    resets=[read(p/'reset.json') for p in sorted((training/'actual_training').glob('episode_*'))]
    actual_conditions=Counter(p['perturbation']['name'] for p in resets)
    train_safety=sum(r['safety_violation_steps']>0 for r in tr['episode_results']);train_contacts=sum(r['prohibited_contact_observations']>0 for r in tr['episode_results'])
    medchange=float(np.median([x['rmse_change_fraction'] for x in pairs]))*100
    report=f'''# Phase 3B 首次在线残差PPO训练验收报告

日期：2026-10-22。工程：`{ROOT}`。本轮短训练已完成并暂停，没有启动100k或其他长训练。

## 结论

**确实完成了真实PPO交互与Actor/Critic优化。基础环境与烟雾训练验收通过；尚不能证明PPO优于M3，不建议未经下一轮验证直接开展大规模正式训练。** 当前只有一个训练种子、1024个决策步、3个完整采样回合，主要结论是工程链路可用。小幅误差变化不是统计显著性或鲁棒性证据。

## 历史与新场景

历史冻结清单：`{prep/'history_sha256.json'}`；{len(manifest)}个文件在交付时逐一SHA256一致，包括Phase3A成果、旧模型、原控制器、157项测试和历史RRT路径，未覆盖。新train/validation/final三布局以新起终点及障碍位置实际规划，规划种子73；20mm规划门槛、0.005rad独立复核、5mm实际失败门槛。没有把旧heldout_box冒充新测试。

各路径真实配置、树、Shortcut CSV、场景及模型指纹在`{prep}`。最终盒形布局只在冻结最终模型、完成25次验证评价后打开，执行6次最终评价。该测试集现在已经查看，未来若调参应另建新的最终未见集。

## 首轮非零动作失败与修复

原候选±0.002rad没有独立残差变化率约束。12次全幅探针中{len(failed_probes)}次交替动作未能稳定到达，原因均为`goal_not_stably_reached`，不是5mm或接触违规；原始失败、全部动作、物理和奖励日志完整保留。未放宽0.02rad/10mm/0.02rad/s保持0.5秒的原成功标准，也未在失败门禁下训练。

随后在M3之后、原总指令限速之前加入**0.01rad/s独立残差变化率约束**，把实际残差状态加入Actor观测，防止隐藏滤波状态。重新运行Gym检查、两个完整零残差等价对照和12次有界全幅测试。修复后12/12稳定到达、无5mm违规/禁止接触；另一次运动中退出保留了真实受限保持及额外关节运动。安全作用仅限这些状态和输入，不能推广为±0.002rad全局安全保证。

原候选目录：{', '.join(e['run'] for e in initial_safety)}。

## 环境、观测与奖励

物理0.002s、500Hz原位置伺服；PPO20Hz、通常每动作25次真实积分。环境用条件变量授权原TrajectoryExecutor状态机积分，没有重新编造完成判据。每个reset的750步保持、失败/退出后的保持、最后不足25步的决策均实际计数。live qpos仅允许reset初始化，运行中通过data.ctrl及mj_step运动。原274个几何配对和真实夹爪/原生接触每步检查保持不变。

78维固定归一化观察只包括可观测q/dq、参考q/dq、跟踪误差、M3基底目标、3次历史误差、进度、上一归一化动作及实际残差状态。Actor没有真实质量、惯量、Kp、阻尼、工况名或精确障碍距离。真实仿真间隙用于奖励/安全日志，属于评价信息而非Actor特征。噪声/观测延迟仍是上层模型，不冒充底层编码器或网络化原生伺服链。

奖励分量逐决策记录：关节/速度误差、残差幅度/变化、时间、参考进度；稳定完成+200、失败或Gym超时−200、禁止接触另−400。800步内最差完成的未折扣回报仍高于提前失败。未完轨迹、保护停止不发完成奖励。真正失败/到达terminated，Gym决策预算truncated；训练总预算结束引发的清理保持独立归档，不回填为成功或虚构PPO终止样本。

详见[环境设计](20_Phase3B残差RL环境设计.md)。

## 零残差等价与依赖

名义、Kp×0.95两组，各完整8919个物理步，原Phase3A M3与新环境零残差在q_ref/dq_ref/q_actual/dq_actual/ctrl、时间、TCP误差及间隙逐项字符串/浮点记录完全一致。初始策略确定性评价的四个共同任务同样与M3逐步一致。

实际解释器为`D:\\miniconda\\envs\\panda_safe_rl\\python.exe`。本机指定环境原来缺少RL包，通过追加已有`mujoco_learn` site-packages只读复用torch 2.14.1+cpu、Gymnasium1.4.0、SB3 2.9.0、TensorBoard2.21.0；仍优先使用Panda numpy2.4.6及MuJoCo3.16.0。不修改任何Reacher工程/历史环境，依赖来源完整写入训练配置。独立安装流程没有执行或验证，后备路径是本机复现的一个限制。

## 真实训练量与参数更新

训练目录：`{training}`。种子42，64×64 Actor/Critic，初始Actor均值层权重/偏置0、高斯标准差0.1；训练使用随机动作，评价用deterministic=True。学习率1e-4，rollout256、batch64、5epoch、gamma0.995、GAE0.95、clip0.1；残差目标±0.002rad并受0.01rad/s限制。

- 实际PPO决策：**{tr['actual_ppo_decisions']}**。
- 决策交互中的实际MuJoCo积分：**{tr['training_step_integrations']}**，包含正常完整回合最后不足25步的决策。
- 4次reset初始保持积分：**{tr['reset_hold_integrations']}**。
- 训练结束清理保持：**{tr['cleanup_hold_integrations']}**；其中1步已进入原控制器的门控等待，随后执行原250步保持，未隐藏继续运动。
- 总实际积分：**{tr['actual_physics_integrations_total']} = {tr['training_step_integrations']} + {tr['reset_hold_integrations']} + {tr['cleanup_hold_integrations']}**。
- 完整真实回合：3个，均成功，结束于决策327/654/981；第4回合在43个决策后因总预算结束，记未完成，不计成功率分母或失败回合冒充学习结果。
- reset实际工况次数：`{dict(actual_conditions)}`。这只是本轮真实抽样，不能声称每种工况都得到充分学习。
- 训练及退出期间5mm违规回合{train_safety}、禁止接触回合{train_contacts}。
- PPO train调用{tr['ppo_update_calls']}次，累计{tr['optimizer_epochs']}epoch；每epoch4个minibatch，合计80次Adam优化步。
- Actor总参数L2变化{tr['actor_delta_l2']:.8f}，Critic {tr['critic_delta_l2']:.8f}；均值输出层最终权重范数{tr['action_head_weight_norm']:.8f}，不是仅修改了日志。
- 随机动作非零比例{tr['stochastic_action_nonzero_fraction']:.1%}，最大已执行归一化动作{tr['maximum_executed_normalized_action']:.6f}。保存原始高斯动作和裁剪动作，可对应到真实位置残差与ctrl。

|更新|决策步|Actor L2增量|Critic L2增量|策略梯度loss|value loss|approx KL|
|---|---:|---:|---:|---:|---:|---:|
{update_table}

最终模型：`{tr['final_model']}`，SHA256 `{tr['final_model_sha256']}`。初始、每次更新后、最终checkpoint均保留；最终模型保存/重载后确定性预测完全一致。TensorBoard事件：`{training/'tensorboard'}`。value loss及短样本表现只能说明发生优化，不能据此声称Critic已收敛。

## 初始与短训策略的配对评价

共31次实际评价：M1/M3/最终PPO各9个同任务条件，初始PPO另4个共同任务。主对比不混用不同任务集。所有物理种子8001，训练/验证/最终布局分别保留标识，禁止接触和失败都计入，没有筛掉较差结果。

|方法|稳定到达|5mm违规回合|禁止接触回合|TCP轨迹RMSE中位mm|最大最终TCP误差mm|
|---|---:|---:|---:|---:|---:|
{table}

初始确定性PPO的4个共同任务与M3逐步一致，不能把初始确定性回放称为随机训练。最终PPO相对M3的9个配对条件TCP RMSE变化中位数为{medchange:+.3f}%（负号改善，正号退化）：

|布局|条件|M3 TCP RMSE mm|短训PPO TCP RMSE mm|变化|
|---|---|---:|---:|---:|
{pairtable}

这一个训练种子的微小差异不支持“PPO必然超过M3”。评价详情与七关节RMSE、实际速度/加速度、指令变化率、饱和和失败原因均在`results/phase3b_evaluation_comparison.csv`；原始CSV可复算。M1保留为辅助基线，M3才是当前主要传统对照。受限积分/自适应尚未实现，不将其说成已被PPO超过。

## 自动验收与真实日志

全部**{testcount}项测试通过、0失败/0跳过**，原157项保留；JUnit/日志：`results/phase3b_release_tests.xml/log`。新增覆盖接口、完整零残差、动作映射及残差/总指令限速、非法动作、真实25步、预算截断和受限保持、原交替失败terminated分类、PPO双网络更新、保存重载、最终测试冻结、无参数特权输入及历史文件一致性。

开发期一次pytest模块导入失败已经修复并保留原日志。初始无残差slew的4次真实失败没有删除，修复后重复门禁而非放松验收标准。独立CSV审计核验31次评价和4个训练物理回合，实际力矩映射、原生力矩限幅、总指令0.35rad/s、残差0.01rad/s及训练各奖励分量均通过，训练物理步总数从原始CSV重新计数一致。

回报图：`assets/plots/phase3b_reward_curve.png`。图中的稀疏完成奖励尖峰反映回合结束位置，不代表学习收敛。原始随机样本：`results/phase3b_smoke_training.csv`；每个episode保留完整tracking/control_signals/clearance/events/decisions及模型参数JSON，未用独立predict回放冒充采样。

## 可复现命令与剩余事项

以下短训练命令本轮实际执行过；再次执行会建立新的独立运行目录和checkpoint，但仍只训练1024个决策步：

```powershell
Set-Location D:\\mujoco_learning\\panda_safe_residual_rl
& D:\\miniconda\\envs\\panda_safe_rl\\python.exe scripts/train_phase3b.py --decisions 1024
```

验证入口（均已实际执行）：

```powershell
& D:\\miniconda\\envs\\panda_safe_rl\\python.exe scripts/validate_phase3b.py --stage interface
& D:\\miniconda\\envs\\panda_safe_rl\\python.exe scripts/validate_phase3b.py --stage equivalence
& D:\\miniconda\\envs\\panda_safe_rl\\python.exe scripts/validate_phase3b.py --stage safety
& D:\\miniconda\\envs\\panda_safe_rl\\python.exe -m pytest -q
```

没有执行2048决策、100k或任何长训练，没有进行多种子训练、大规模组合随机化、受限积分/自适应对照、实机或连续安全认证。历史D窄间隙路径未进入本轮PPO训练或评价，不能声称已经修复Phase3A已知失败。新测试集已打开，下一轮调参需再保留新最终测试布局。完整环境链路可以支撑下一轮明确授权的小规模研究，但**当前证据不支持直接启动大规模正式训练**；先补多种子有限pilot、传统强基线、奖励/观察分析及更丰富安全验证。阶段到此暂停。
'''
    (ROOT/'docs/21_Phase3B首次训练验收报告.md').write_text(report,encoding='utf-8')
    required=['docs/20_Phase3B残差RL环境设计.md','docs/21_Phase3B首次训练验收报告.md','results/phase3b_smoke_training.csv',
        'results/phase3b_zero_residual_equivalence.json','results/phase3b_action_safety_tests.json','assets/plots/phase3b_reward_curve.png']
    write_json(ROOT/'results/phase3b_acceptance.json',dict(tests_passed=testcount,training_run=str(training),evaluation_validation_run=str(validation),evaluation_final_run=str(final),
        zero_residual_gate=True,nonzero_gate=True,prior_files_unchanged=len(manifest),training_decisions=1024,training_physics=tr['actual_physics_integrations_total'],
        final_model=tr['final_model'],final_model_sha256=tr['final_model_sha256'],recommend_long_training=False,
        required_files={name:dict(sha256=sha(ROOT/name),bytes=(ROOT/name).stat().st_size) for name in required}))
    print('REPORT COMPLETE',groups,flush=True)

if __name__=='__main__':main_guard(main)
