"""Actual-run ledger, conservative-stop analysis, figures and Chinese report."""
from _bootstrap import main_guard
import csv,json
from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.common import ROOT,sha,write_json,write_csv
from src.control.phase2e_provenance import verify_frozen_history


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def uid(r):return Path(r['run_directory']).name.split('_M')[0]
def safe_csv(rows):
    return [{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in rows]


def main():
    history=verify_frozen_history();base=Path(read(ROOT/'results/phase2e_batch_latest.json')['run'])
    runs=read(base/'complete.json')['results'];cohort=read(base/'cohort_before_execution.json')
    static=Path(read(ROOT/'results/phase2e_static_latest.json')['run']);statics=read(static/'complete.json')['results']
    shadow=Path(read(ROOT/'results/phase2e_shadow_latest.json')['run']);gate=read(shadow/'shadow_gate.json')
    active=Path(read(ROOT/'results/phase2e_active_pilot_latest.json')['run'])
    audits=read(ROOT/'results/phase2e_csv_audit.json');assert audits['all_metrics_match'] and len(runs)==66
    assert read(ROOT/'results/phase2e_m0_baseline_reproduction.json')['all_matched']
    tests=ET.parse(ROOT/'results/phase2e_release_tests.xml').getroot();n=len(list(tests.iter('testcase')))
    assert n==137 and not list(tests.iter('failure')) and not list(tests.iter('error'))
    ledger=[];safety=[];conditions=[]
    for r in runs:
        path_uid=uid(r);members=[c for c in cohort['conditions'] if c['path_uid']==path_uid]
        actual_id=Path(r['run_directory']).name
        record={**r,'actual_execution_id':actual_id,'path_uid':path_uid,'condition_membership':members,
             'legacy_reference_rate_limit_count':r['command_rate_limit_steps'],'command_rate_limit_steps':r['effective_rate_limit_steps']}
        ledger.append(record)
        audit=next(x for x in audits['audits'] if x['run_directory']==r['run_directory'])
        paired=next(x for x in runs if uid(x)==path_uid and x['method']=='M1' and x['reference_speed_rad_s']==r['reference_speed_rad_s'])
        conservative=r['outcome']=='protected_stop' and paired['success'] and paired['safety_violation_steps']==0
        safety.append(dict(actual_execution_id=actual_id,method=r['method'],speed_rad_s=r['reference_speed_rad_s'],outcome=r['outcome'],
            first_warning_time_s=r['first_warning_time_s'],guard_action_time_s=r['guard_action_time_s'],
            first_actual_violation_s=r['first_hard_violation_time_s'],first_contact_s=r['first_contact_time_s'],
            contact_intervals=r['contact_intervals'],post_stop_joint_displacement_rad=r['post_stop_joint_displacement_rad'],
            post_stop_minimum_gap_m=audit['post_stop_min_gap_m'],safely_stopped_without_goal=r['safely_stopped_without_goal'],
            conservative_stop_on_M1_completed_path=conservative,paired_M1_source=paired['run_directory'],
            guard_algorithm_s=r['guard_algorithm_s'],guard_calls=r['guard_calls'],run_directory=r['run_directory']))
        for i,c in enumerate(members):conditions.append(dict(**c,method=r['method'],speed_rad_s=r['reference_speed_rad_s'],
            actual_execution_id=actual_id,reuses_same_actual_execution=i>0,outcome=r['outcome'],success=r['success'],run_directory=r['run_directory']))
    write_csv(ROOT/'results/phase2e_controller_comparison.csv',safe_csv(ledger))
    write_csv(ROOT/'results/phase2e_safety_guard_analysis.csv',safety);write_csv(ROOT/'results/phase2e_condition_results.csv',conditions)
    groups=[];table=[];joints=[];costs=[]
    for method in ['M0','M1','M2']:
        rs=[r for r in runs if r['method']==method];success=[r for r in rs if r['success']]
        events=[p['computation_s'] for r in rs for p in read(Path(r['run_directory'])/'predictions.json') if 'computation_s' in p]
        g=dict(method=method,actual_runs=len(rs),outcomes=dict(Counter(r['outcome'] for r in rs)),
            contact_intervals=sum(r['contact_intervals'] for r in rs),hard_violation_runs=sum(r['first_hard_violation_time_s'] is not None for r in rs),
            minimum_actual_gap_m=min(r['minimum_actual_clearance_m'] for r in rs),torque_saturation_steps=sum(r['torque_saturation_steps'] for r in rs),
            maximum_successful_final_tcp_error_m=max(r['final_tcp_error_m'] for r in success),
            median_successful_final_tcp_error_m=float(np.median([r['final_tcp_error_m'] for r in success])),
            joint_rmse_mean_rad=np.mean([r['joint_rmse_rad'] for r in rs],axis=0).tolist(),
            actual_max_velocity_rad_s=max(r['actual_max_velocity_rad_s'] for r in rs),actual_max_acceleration_rad_s2=max(r['actual_max_acceleration_fd_rad_s2'] for r in rs),
            max_ctrl_rate_rad_s=max(r['max_ctrl_rate_rad_s'] for r in rs),max_ctrl_acceleration_rad_s2=max(r['max_ctrl_acceleration_rad_s2'] for r in rs),
            compensation_clipped_steps=sum(r['compensation_clipped_steps'] for r in rs),position_clipped_steps=sum(r['position_clipped_steps'] for r in rs),
            effective_rate_limit_steps=sum(r['effective_rate_limit_steps'] for r in rs),
            gravity_amortized_us_per_physics_step=sum(r['gravity_algorithm_s'] for r in rs)/sum(r['physics_steps'] for r in rs)*1e6,
            guard_amortized_us_per_physics_step=sum(r['guard_algorithm_s'] for r in rs)/sum(r['physics_steps'] for r in rs)*1e6,
            guard_call_p50_s=float(np.percentile(events,50)) if events else None,
            guard_call_p95_s=float(np.percentile(events,95)) if events else None,guard_call_max_s=max(events,default=None))
        groups.append(g)
        table.append(f"| {method} | {g['outcomes'].get('success',0)}/22 | {g['outcomes'].get('protected_stop',0)} | {g['outcomes'].get('failure',0)} | {g['hard_violation_runs']} | {g['contact_intervals']} | {g['minimum_actual_gap_m']*1000:.4f} | {g['median_successful_final_tcp_error_m']*1000:.8f} | {g['torque_saturation_steps']} |")
        joints.append('| '+method+' | '+' | '.join(f'{x:.6f}' for x in g['joint_rmse_mean_rad'])+' |')
        costs.append(f"| {method} | {g['gravity_amortized_us_per_physics_step']:.3f} | {g['guard_amortized_us_per_physics_step']:.3f} | {g['guard_call_p95_s'] if events else '—'} | {g['max_ctrl_rate_rad_s']:.6f} | {g['max_ctrl_acceleration_rad_s2']:.3f} | {g['actual_max_acceleration_rad_s2']:.6f} |")
    plots=ROOT/'assets/plots';plots.mkdir(exist_ok=True,parents=True)
    names=['home','D_start','D_critical_reference'];fig,ax=plt.subplots(figsize=(9,4));x=np.arange(3)
    for i,method in enumerate(['M0','M1']):
        values=[next(r['final_tcp_error_m']*1000 for r in statics if r['static_case']==name and r['method']==method) for name in names]
        ax.bar(x+(i-.5)*.32,values,.32,label=method)
    ax.set(xticks=x,xticklabels=names,ylabel='Final static TCP error (mm)',title='Same ideal nominal model; M1 settles to numerical precision');ax.legend();fig.tight_layout();fig.savefig(plots/'phase2e_static_accuracy.png',dpi=150);plt.close(fig)
    # Use the actual native video episodes for a matched original-D trace.
    vcontrol=Path(read(ROOT/'results/phase2e_controller_video_latest.json')['run']);vguard=Path(read(ROOT/'results/phase2e_guard_video_latest.json')['run'])
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    with (vcontrol/'M1'/'tracking.csv').open(encoding='utf-8-sig',newline='') as f:
        for last_reference in csv.DictReader(f):pass
    goal=np.array([float(last_reference[f'tcp_ref_{j}']) for j in range(3)])
    for method,folder in [('M0',vcontrol/'M0'),('M1',vcontrol/'M1'),('M2',vguard/'M2')]:
        with (folder/'tracking.csv').open(encoding='utf-8-sig',newline='') as f:data=list(csv.DictReader(f))[::5]
        t=np.array([float(r['time']) for r in data]);xyz=np.array([[float(r[f'tcp_actual_{j}']) for j in range(3)] for r in data])
        axs[0].plot(t,[float(r['minimum_clearance_m'])*1000 for r in data],label=method)
        axs[1].plot(t,np.linalg.norm(xyz-goal,axis=1)*1000,label=method)
    axs[0].axhline(5,color='red',ls='--',label='5mm hard threshold');axs[0].set(xlabel='Simulation time (s)',ylabel='Actual sampled gap (mm)')
    axs[1].set(xlabel='Simulation time (s)',ylabel='TCP distance to final goal (mm)',title='Protective stop is not goal completion')
    for ax in axs:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(plots/'phase2e_guard_and_goal.png',dpi=160);plt.close(fig)
    static_table=[]
    for r in statics:static_table.append(f"| {r['static_case']} | {r['method']} | {r['final_tcp_error_m']*1000:.9g} | {r['final_joint_max_error_rad']:.9g} | {max(r['steady_joint_peak_to_peak_rad']):.3g} | {r['actual_max_velocity_rad_s']:.6f} | {r['actual_max_acceleration_fd_rad_s2']:.6f} | {r['torque_saturation_steps']} |")
    videoaudit=read(ROOT/'results/phase2e_video_audit.json');conservative=sum(x['conservative_stop_on_M1_completed_path'] for x in safety)
    performance=dict(date='2026-10-22',groups=groups,actual_dynamic_executions=66,condition_assessments=len(conditions),unique_paths=11,
        static_executions=6,authoritative_shadow_executions=4,active_pilot_executions=2,video_executions=4,
        conservative_stops=conservative,shadow_gate=gate,provenance=history,csv_all_metrics_recomputed=True,
        parameter_perturbations_executed=False,reinforcement_learning_executed=False,tests_passed=n,video_audit=videoaudit)
    write_json(ROOT/'results/phase2e_performance.json',performance)
    doc=f'''# Panda Phase 2E：传统重力补偿与提前保护验收报告

日期：2026-10-22。项目 `D:\\mujoco_learning\\panda_safe_residual_rl`，Python `D:\\miniconda\\envs\\panda_safe_rl\\python.exe`，MuJoCo 3.16.0。本轮实际完成独立补偿控制器、shadow预测验证、主动保护pilot、有界正式对照、完整CSV复算及原生live动力学视频；没有训练PPO/SAC，没有修改旧RRT、模型、伺服增益、力限、障碍或原5 mm执行标准。

## 1. 历史审计与冻结

首次冻结目录见 results/phase2e_freeze_latest.json。740项文件有冻结SHA256；739项原位置保持一致，另1项旧回归测试因取数错误作了明确归档修复，原文件逐字节保存在冻结目录，匹配原SHA；所有实验CSV、失败记录、模型、控制器、旧视频、旧latest指针均未改动。修复说明见第9节及 results/phase2e_test_artifact_fix.json。

实际执行器是直接关节位置伺服：Kp=[4500,4500,3500,3500,2000,2000,2000]，Kd=[450,450,350,350,200,200,200]，gear=1；力矩前四关节±87 N·m、后三关节±12 N·m。ctrl是rad，不是力矩。body_gravcomp及stiffness均为0，无流体项。独立零速度数据计算重力，按真实dof地址取值并核对被动力，防止将普通动态bias或重复补偿误称为纯重力。

## 2. 三组冻结控制方案

- M0：原 JointPositionController 及原实际保护，源文件未修改。
- M1：新增重力前馈 ctrl_target=q_ref+gravity(q_actual)/Kp；每关节补偿限0.02 rad，总命令变化率0.35 rad/s，原位置/力矩限保持。通过原controller.step完成一次真正mj_step，补偿可关闭。
- M2：与M1相同，再以0.05 s周期对独立shadow预测0.30 s，在预测最低间隙≤7 mm时提前暂停参考、转入受限补偿保持。7 mm含2 mm预警缓冲；实际失败仍为原5 mm。未实现有制动距离证明的减速曲线，不宣称严格安全。

设计细节见 docs/15_重力补偿控制器设计.md 和 docs/16_提前保护策略设计.md。所有正式参考、模型、初始配置、五次时间缩放、0.002 s物理步、500 Hz物理控制/监控及原目标验收条件相同。只有仿真开始前明确初始化，执行期间没有live qpos传送；shadow积分不计作live积分或实际录像。

## 3. 按顺序实际执行的门槛

1. 静态M0/M1：3合法无障碍构型，共6次。要求均稳定、无力矩饱和、最后0.5 s关节峰峰值<1e-4 rad，且M1逐构型TCP静差改善。
2. 仅shadow：原D seed42与8 mm seed123、0.15 rad/s，M0/M1共4次；不干预命令。{gate['comparisons']}个未来样本对齐，最大关节误差 {gate['max_joint_prediction_error_rad']:.3g} rad、间隙误差 {gate['max_gap_prediction_error_m']:.3g} m；名义模型高度吻合不等于未知扰动鲁棒性。两例已知M0失败提前0.320/0.488 s预警，预测计算P95 {gate['p95_prediction_time_s']:.6f} s，通过预先规定门槛。
3. 主动pilot：2例M2，一例正常到达、一例稳定保护停止；无实际违规/接触/预测异常，再进入正式阶段。开发期静态与shadow初稿运行亦保留，最终统计使用latest登记的复核版本，不混作额外独立样本。
4. 正式：原Phase2C的A/B/C/D路径，以及Phase2D全部5/8/10/15 mm、42/123/2026种子路径，两档速度0.10/0.15，M0/M1/M2全部纳入。相同路径及场景去重后11条不同路径，实际66次执行。对应16个来源条件、96个条件评估条目，复用映射明确记录；不把96条当成96次独立执行或随机场景。没有只挑有效果的路径。

正式目录：`{base}`。各执行目录保留原tracking/clearance/events、控制信号、预测、配置与summary；失败和保护停止均保留。

## 4. 静态位置保持结果

| 构型 | 方法 | 最终TCP误差 mm | 最大稳态关节误差 rad | 稳态关节最大峰峰值 rad | 实际最大速度 rad/s | 初始化/全段最大加速度 rad/s² | 饱和步数 |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(static_table)}

M1的极小稳态误差是匹配理想仿真模型的数值结果，不能宣称实机同精度。七关节实际稳态偏差、真实力矩与速度在静态源CSV中完整保留；稳态幅度及速度门槛检查没有发现不合理振荡。初始化瞬态单独保留，参考加速度约束不是实际加速度的硬保证。

![静态精度](../assets/plots/phase2e_static_accuracy.png)

## 5. 正式66次实际执行

| 方法 | 到达成功 | 保护停止未到达 | 实际失败 | 有5 mm违规的回合 | 连续禁止接触区间 | 最低实际间隙 mm | 成功回合最终TCP误差中位数 mm | 力矩饱和步数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

没有人为中止回合。M0的6次失败及2次实际接触区间完整保留；M1名义样本22/22到达，没有5 mm违规或接触；M2为20/22到达，2次安全保护停止未到目标。不能将保护停止计作成功。

原M0的22个不同路径/速度条件与Phase2C/2D对应历史的成功标记、失败原因、物理步数、时间、最终q、RMSE和间隙逐项一致，见 results/phase2e_m0_baseline_reproduction.json。没有削弱M0或重写旧实验。

七关节跟踪RMSE按每回合已执行部分统计，再对方法内回合取均值；失败早停的统计窗口不同，不能据此单独宣称完整路径误差改善比例或统计显著：

| 方法 | J1 rad | J2 | J3 | J4 | J5 | J6 | J7 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(joints)}

逐回合RMSE、每关节最大误差、TCP跟踪RMSE/最大/最终误差、完成时间、间隙、违规/接触、饱和、命令平滑性与算法成本均在 phase2e_controller_comparison.csv。成功、失败与未到达分开，未用局部跟踪小误差掩盖到目标剩余距离。

## 6. 提前保护的实际效果及代价

shadow确实提前发现了已知M0失败；M2也在对应名义路径上于5 mm违规前稳定停住，停止后额外位移、实际间隙和速度全部记录。但配对M1在这两次都能完成且无违规，因此当前M2有 {conservative} 次“原本M1可以完成的保守中止”。不能将M0到M2的改善全归因于guard，重力补偿已解决这些名义失败；本样本没有证明guard比M1额外避免了真实违规。

预警7 mm与实际失败5 mm分离，当前控制器依旧保留原受限保持窗口。M1/M2保持时补偿重力有助于减少新的重力静差，仍不能保证消除所有惯性、参数误差或制动后碰撞。本轮未为了达到目标而事后缩小缓冲或修改阈值。

![真实间隙与到目标距离](../assets/plots/phase2e_guard_and_goal.png)

安全细表 phase2e_safety_guard_analysis.csv 分别列首次warning、guard动作、第一次实际5 mm违规、第一次接触、保持后额外关节位移/最低间隙、是否稳定停住、配对M1及保守停止标记。predictions.json是独立shadow数据，不能当实际轨迹；实际CSV和视频仅来自live。

## 7. 力矩、平滑性与计算成本

所有实际力矩已从ctrl/q/qvel重新按原affine映射和原force limits核对；三组实际均无力矩饱和。本轮没有出现补偿偏移裁剪或位置裁剪造成的隐蔽路径改变，完整介入次数见performance.json。仍保留原命令变化率限制。

| 方法 | 重力算法摊销 μs/物理步 | guard摊销 μs/物理步 | 单次预测P95 s | 最大ctrl变化率 rad/s | 最大ctrl二阶差分 rad/s² | 最大实际加速度 rad/s² |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(costs)}

补偿提高稳态精度，同时建立补偿时命令变化率可能达到0.35 rad/s，命令二阶差分可明显大于参考加速度；这些不是机器人实际加速度，不将其混用。实际初始化加速度峰值保留于物理日志。

500 Hz表示仿真步频，不是墙钟每2 ms的硬实时保证。预测会阻塞仿真推进，正式三进程和原生渲染负载影响墙钟开销；单次预测耗时及摊销值均为实测。没有用更少必要实际碰撞检查换取性能。

## 8. 原始CSV与实际视频验证

66次tracking/control_signals已独立复算七关节RMSE/最大误差、TCP指标、最低间隙、违规、连续接触、最终稳定成功判定、力矩饱和、实际力矩映射、命令变化率/总变差；66/66一致，见 phase2e_csv_audit.json。原执行器legacy命令限速计数器对补偿ctrl不再有相同语义；本阶段CSV明确保存legacy值，并以实际corrected-target限速标记的effective计数作为正式指令限速指标。

两段原生Windows窗口视频都经过实际运行、完整解码和逐帧CSV核对：

- phase2e_controller_comparison.mp4：M0/M1分别独立初始化，原D seed42、0.15 rad/s；原版失败、补偿版到达。
- phase2e_safety_guard.mp4：M1/M2分别独立初始化、相同条件；M1到达，M2明确显示预警和保护未到达。

{chr(10).join('- '+v['kind']+'：'+str(v['decoded_frames'])+'帧，与实际tracking的时间/q_ref/q_actual/ctrl全部匹配。' for v in videoaudit['videos'])}

帧直接读取积分中的live MjData，没有渲染shadow预测或通过qpos播放伪造控制。窗口均正常关闭；分段比较时间会明确随每次独立初始化重新开始。

## 9. 自动测试与可复现性修复

最终 {n} passed / 0 failed / 0 skipped，包含原120项全部回归与17项新增测试。新增覆盖纯重力/gear/力矩映射、独立零速查询、拒绝重复补偿/弹簧/不支持模型、关闭补偿等价原版、偏移/位置/力矩限制、shadow不改live或加物理步、预测异常停止归档、静态稳定、全部条件/CSV、M0历史复现、两段实际视频。

首轮完整测试为136通过、1失败，日志 phase2e_full_tests.log/xml 保留。原因是任务开始前可变Phase2D latest指针已指向一次用户关闭窗口（viewer_closed）的演示，而旧测试却要求第二段成功。原验收固定录像仍存在且SHA为61dd0e6b6470d84a122b0a5bc75e1a490a35cd42d0b915033073df78500ab733。修复仅将该测试读取的manifest和视频路径改为Phase2D验收登记的固定版本；全部断言的AST完全一致，原测试字节与SHA备份在冻结目录。没有改latest、旧视频或用户中止记录，也没有将中止算成算法失败。兼容审计模块又修复了Windows反斜杠路径规范化；加载旧版模块的第二轮测试日志 phase2e_final_tests.log/xml 同样保留。最终全套验证采用新进程，日志为 phase2e_release_tests.log/xml。

## 10. 可直接运行的PowerShell命令

```powershell
Set-Location D:\\mujoco_learning\\panda_safe_residual_rl
& D:\\miniconda\\envs\\panda_safe_rl\\python.exe scripts/demo_compensated_control.py --viewer --speed 0.15
& D:\\miniconda\\envs\\panda_safe_rl\\python.exe scripts/demo_safety_guard.py --viewer --speed 0.15
```

两条命令均已在Windows原生MuJoCo窗口实际运行。程序真实执行并默认录制，Esc/Q/关闭窗口归档中止。方法以M0/M1/M2和DYNAMIC SERVO EXECUTION明确标注，不要将旧场景绿色标记误当本轮TCP目标，参考和实际位置以明确数值为准。

重做分级实验：

```powershell
& D:\\miniconda\\envs\\panda_safe_rl\\python.exe scripts/run_phase2e.py --stage static
& D:\\miniconda\\envs\\panda_safe_rl\\python.exe scripts/run_phase2e.py --stage shadow
& D:\\miniconda\\envs\\panda_safe_rl\\python.exe scripts/run_phase2e.py --stage active_pilot
& D:\\miniconda\\envs\\panda_safe_rl\\python.exe scripts/run_phase2e.py --stage batch
& D:\\miniconda\\envs\\panda_safe_rl\\python.exe scripts/audit_phase2e.py
& D:\\miniconda\\envs\\panda_safe_rl\\python.exe -m pytest -q
& D:\\miniconda\\envs\\panda_safe_rl\\python.exe scripts/summarize_phase2e.py
```

每次创建独立目录，保留旧版本。首次建立新冻结记录可运行 freeze_phase2e.py；本轮冻结指针和原始版本已保留，不必重复覆盖。Codex受限Windows环境的多进程需允许本地管道；本轮有界66次正式试验通过3个隔离进程实际完成。

## 11. 基线选择、未执行事项与后续残差接口

在已测试名义条件中，M1最适合作为**控制精度基线**：静态更准、22/22目标完成且无实际违规/接触；保留原5 mm监控、模型与限制。M2当前可作保守保护实验组，但两次不必要停止降低目标完成率，不能自动宣称比M1更适合作为默认任务控制器。两者都不是实机安全认证。

未执行：增益/阻尼/负载参数扰动、未知外力或感知延迟、更多规划随机种子、随机执行扰动、持续速度重定时、制动距离证明、长期稳定/实机测试，以及所有PPO/SAC训练。这些不是已经获得的鲁棒性结果。

当前名义数据尚未证明必须引入RL，已知重力静差可由传统模型补偿解决。剩余运动滞后应先考虑独立验证的传统速度/加速度前馈与参考重定时；未知载荷等困难工况需真实诊断后再判断是否有传统方法无法充分处理的残差。不能削弱M0制造优势，也不能让RL替代基础保护。

将来若满足独立场景划分和安全验证条件，可从实际q/qvel、q_ref/dq_ref/ddq_ref、TCP误差、补偿量、原ctrl、连杆间隙/趋势和保护状态构建观测，输出受限残差偏移，在重力前馈后统一施加位置/变化率及native力矩约束。需继续保留无RL的M1/M2消融。本轮仅说明连接方式，没有实现或启动残差策略。

离散检查、同模型短时预测及当前有限样本都不构成全局安全保证。本阶段到此暂停，等待用户审核。
'''
    (ROOT/'docs/17_Phase2E验收报告.md').write_text(doc,encoding='utf-8')
    write_json(ROOT/'results/phase2e_acceptance.json',dict(date='2026-10-22',phase='2E',tests_passed=n,actual_formal_runs=66,
        groups=groups,history=history,static_run=static,shadow_run=shadow,active_pilot_run=active,batch_run=base,
        controller_video_run=vcontrol,guard_video_run=vguard,rl_executed=False,parameter_perturbations_executed=False,
        required_files={p:dict(sha256=sha(ROOT/p),bytes=(ROOT/p).stat().st_size) for p in ['docs/15_重力补偿控制器设计.md','docs/16_提前保护策略设计.md',
        'docs/17_Phase2E验收报告.md','results/phase2e_controller_comparison.csv','results/phase2e_safety_guard_analysis.csv','results/phase2e_performance.json',
        'assets/videos/phase2e_controller_comparison.mp4','assets/videos/phase2e_safety_guard.mp4']}))
    print('COMPLETE Phase2E:',n,'tests, 66 actual formal runs, all CSV audited, no RL.',flush=True)


if __name__=='__main__':main_guard(main)
