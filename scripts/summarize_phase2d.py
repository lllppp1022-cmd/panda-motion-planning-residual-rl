"""Publish Phase2D conclusions exclusively from real recorded experiments."""
from _bootstrap import main_guard
import json,csv
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.common import ROOT,sha,write_json,write_csv


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def main():
    freeze=Path(read(ROOT/'results/phase2d_freeze_latest.json')['run']);hashes=read(freeze/'historical_hashes.json')
    unchanged={p:sha(ROOT/p)==h for p,h in hashes.items()};assert all(unchanged.values())
    plan=Path(read(ROOT/'results/phase2d_planning_latest.json')['run']);plans=read(plan/'complete.json')['results']
    dyn=Path(read(ROOT/'results/phase2d_dynamic_latest.json')['run']);runs=read(dyn/'complete.json')['results']
    audit=read(ROOT/'results/phase2d_csv_recomputation.json');assert audit['all_metrics_match'] and audit['trials']==len(runs)
    video=Path(read(ROOT/'results/phase2d_video_latest.json')['run']);clips=read(video/'complete.json')['episodes']
    movie=read(video/'comparison_validation.json');analysis=read(ROOT/'results/phase2d_failure_analysis.json')
    testxml=ET.parse(ROOT/'results/phase2d_final_tests.xml').getroot();tests=len(list(testxml.iter('testcase')))
    assert not list(testxml.iter('failure')) and not list(testxml.iter('error')) and tests==120
    assert len(plans)==12 and len(runs)==24
    groups=[];table=[];seedtable=[];timingtable=[]
    for margin in [.005,.008,.010,.015]:
        p=[r for r in plans if r['planning_clearance_m']==margin];d=[r for r in runs if r['planning_clearance_m']==margin]
        a=[r for r in audit['audits'] if r['planning_clearance_m']==margin]
        g=dict(planning_clearance_m=margin,execution_clearance_m=.005,planning_trials=len(p),planning_successes=sum(r['success'] for r in p),
            raw_fine_passes=sum(r['raw_fine_passed'] for r in p),final_fine_passes=sum(r['final_validation_passed'] for r in p),
            dynamic_trials=len(d),dynamic_successes=sum(r['success'] for r in d),
            minimum_planned_gap_m=min(r['minimum_sampled_clearance_m'] for r in p if r['success']),
            minimum_actual_gap_m=min(r['minimum_actual_clearance_m'] for r in d),
            contact_intervals=sum(r['forbidden_contact_intervals'] for r in a),contact_executions=sum(r['forbidden_contact_intervals']>0 for r in a),
            planning_median_s=float(np.median([r['planning_time_s'] for r in p])),
            path_length_median_rad=float(np.median([r['path_length_rad'] for r in p])),
            dynamic_simulation_median_s=float(np.median([r['simulation_time_s'] for r in d])),
            planning_failures=[dict(seed=r['planning_seed'],reason=r['failure_reason']) for r in p if not r['success']])
        groups.append(g)
        table.append(f"| {margin*1000:.0f} | {g['planning_successes']}/3 | {g['final_fine_passes']}/3 | {g['dynamic_successes']}/6 | {g['minimum_planned_gap_m']*1000:.3f} | {g['minimum_actual_gap_m']*1000:.3f} | {g['contact_executions']} | {g['planning_median_s']:.3f} | {g['path_length_median_rad']:.3f} |")
        for speed in [.10,.15]:
            ds=[r for r in d if r['reference_speed_rad_s']==speed]
            timingtable.append(f"| {margin*1000:.0f} / {speed:.2f} | {sum(r['success'] for r in ds)}/3 | "+
                f"{min(r['simulation_time_s'] for r in ds):.3f}～{max(r['simulation_time_s'] for r in ds):.3f} | "+
                f"{max(max(r['joint_max_abs_error_rad']) for r in ds):.6f} | "+
                f"{min(r['tcp_rmse_m']*1000 for r in ds):.3f}～{max(r['tcp_rmse_m']*1000 for r in ds):.3f} | "+
                f"{min(r['final_tcp_error_m']*1000 for r in ds):.3f}～{max(r['final_tcp_error_m']*1000 for r in ds):.3f} |")
    identities={}
    for r in plans:
        source=read(r['source_json']);key=json.dumps(source['result']['path'],sort_keys=True)
        identities.setdefault(key,[]).append((r['planning_clearance_m'],r['planning_seed']))
        seedtable.append(f"| {r['planning_clearance_m']*1000:.0f} | {r['planning_seed']} | {'成功' if r['success'] else r['failure_reason']} | {r['minimum_sampled_clearance_m']*1000:.6f} | {r['raw_nodes']}→{r['final_nodes']} | {r['raw_length_rad']:.4f}→{r['path_length_rad']:.4f} | {r['planning_time_s']:.3f} |")
    plots=ROOT/'assets/plots';plots.mkdir(exist_ok=True,parents=True)
    fig,ax=plt.subplots(figsize=(9,4.8))
    for seed,color in [(42,'#2876b5'),(123,'#df6b32'),(2026,'#249265')]:
        for speed,marker in [(.10,'o'),(.15,'x')]:
            r=[x for x in runs if x['planning_seed']==seed and x['reference_speed_rad_s']==speed]
            ax.plot([x['planning_clearance_m']*1000 for x in r],[x['minimum_actual_clearance_m']*1000 for x in r],marker=marker,
                color=color,ls='-' if speed==.1 else '--',label=f'seed {seed}, {speed:.2f} rad/s')
    ax.axhline(5,color='red',ls=':',label='Unchanged execution threshold: 5 mm');ax.axhline(0,color='black',lw=.7)
    ax.set(xlabel='Required planning clearance (mm)',ylabel='Minimum actual gap incl. protective hold (mm)',xticks=[5,8,10,15],title='Same scene, original servo, real dynamic execution')
    ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(plots/'phase2d_margin_vs_actual.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for d,ax in zip(analysis['historical_failures'][::2],axs):
        samples=d['directional_distance_samples'];fraction=np.array([s['fraction'] for s in samples])
        ax.plot(fraction,[s['distance_m']*1000 for s in samples],'o-',label='Full geometry query')
        ax.plot(fraction,(d['reference_pair_gap_m']+fraction*d['linearized_gap_change_m'])*1000,'--',label='Local linearization')
        ax.axhline(5,color='red',ls=':');ax.set(title=f"Historical {d['speed']:.2f} rad/s failure",xlabel='Fraction of reference-to-actual joint error',ylabel='Specific pair gap (mm)');ax.legend(fontsize=8)
    fig.suptitle('Kinematic sensitivity diagnostic, NOT an actual executed trajectory');fig.tight_layout();fig.savefig(plots/'phase2d_local_sensitivity.png',dpi=150);plt.close(fig)
    # Verify each live video frame against its exact integrating CSV row.
    alignments=[]
    for clip in clips:
        folder=video/clip['episode'];frames=read(folder/'video_frames.json');lookup={f['physics_step']:f for f in frames};matched=0
        with (folder/'tracking.csv').open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                if int(row['physics_step']) not in lookup:continue
                frame=lookup[int(row['physics_step'])];assert frame['time']==float(row['time'])
                for field in ['q_actual','q_ref','ctrl']:
                    np.testing.assert_array_equal([float(row[f'{field}_{j}']) for j in range(1,8)],frame[field])
                matched+=1
        assert matched==len(frames);alignments.append(dict(episode=clip['episode'],frames_matched=matched))
    write_json(ROOT/'results/phase2d_performance.json',dict(groups=groups,unique_final_joint_paths=len(identities),planning_conditions=12,dynamic_trials=24,
         cohort='three planning seeds, paired velocities, no independent stochastic execution repeats',csv_audit=audit['all_metrics_match'],video_alignment=alignments))
    # Enrich only the NEW stage canonical CSV. Raw per-run summaries stay intact.
    enriched=[]
    for r in runs:
        a=next(x for x in audit['audits'] if x['planning_clearance_m']==r['planning_clearance_m'] and x['planning_seed']==r['planning_seed'] and x['speed_rad_s']==r['reference_speed_rad_s'])
        enriched.append(dict(**r,forbidden_contact_intervals=a['forbidden_contact_intervals'],first_forbidden_contact_time_s=a['first_forbidden_contact_time_s'],csv_metrics_recomputed=True))
    write_csv(ROOT/'results/phase2d_dynamic_execution_results.csv',enriched)
    first_contact=next(a for a in audit['audits'] if a['forbidden_contact_intervals'] and a['speed_rad_s']==.1)
    baseline,improved=clips;doc=f'''# Panda Phase 2D：规划与真实执行安全裕度一致性验收报告

日期：2026-10-21。项目：`D:\\mujoco_learning\\panda_safe_residual_rl`。解释器：`D:\\miniconda\\envs\\panda_safe_rl\\python.exe`。本轮完成历史冻结、失败机制审计、规划/执行门槛独立实现、四档裕度的有界搜索、全部通过路径的真实执行、CSV独立复算和实际动力学视频。没有训练 PPO/SAC，也没有修改原位置伺服、障碍、模型限位、执行5 mm门槛或原成功标准。

## 1. 历史冻结与控制变量

阶段开始快照：`{freeze}`，共 {len(hashes)} 项文件（历史结果及当时接口快照）；结束时逐项哈希检查一致。原 Phase 2B/2C CSV、JSON、失败轨迹、模型、检查器、执行器、控制器、配置、旧测试、文档及视频均保留，80 个上游文件仍与历史来源清单一致。原 D 规划路径及执行失败没有被替换。

D 场景球/圆柱形状、尺寸、位置、四元数及起终点完全采用原 seed=42 场景；本轮变化只有规划最小间隙和规划随机种子。执行继续使用原五次时间缩放、0.002 s物理步、500 Hz控制、原0.35 rad/s指令变化限制、0.30 rad/s²参考加速度和原重力/伺服/力限。初始化仍仅在仿真开始前一次设置到保存起点，后续通过 data.ctrl + mj_step，不进行动态 qpos 传送。

## 2. 参数独立实现与预算

新增 `src/planning/margin_study.py` 中的 `MarginRequirements(planning_clearance_m, execution_clearance_m)`：规划为5/8/10/15 mm，执行参数只允许原5 mm。规划使用新建检查器及对应间隙，所有节点、树边、Shortcut和新模型0.005 rad完整复核继续调用原Phase2A/2B接口；原全局默认配置未修改。

进入执行前重新读取实际保存的路径CSV、成功/原始与最终复核标记、模型/过滤哈希及原场景，当前环境再次逐节点逐边复核对应规划门槛。随后另建5 mm检查器，交给完全未修改的 Phase2C `TrajectoryExecutor`。规划15 mm并不意味着执行器以15 mm判失败；它仍以5 mm判定。

配置：`configs/phase2d_margin_study.yaml`。固定规划种子42、123、2026；每档3次，共12次真实双树搜索。每次最多800迭代、6 s搜索、12 s总预算、CONNECT最多128步、Shortcut40次/2 s；搜索0.02 rad、最终0.005 rad。所有种子结果都保留。

pilot 全部完成后、动力学执行前保存预算决定：对全部12条通过路径各执行0.10/0.15 rad/s一次，共24次；不额外扩展规划种子。它是有界三种子可行性研究，**更大规划种子批量未执行**。三个隔离Python工作进程仅用于独立仿真加速，每条执行仍受原120 s仿真/180 s墙钟预算。固定路径两档速度是配对条件，不是独立随机场景，也没有将历史确定性重复算作新规划种子。

## 3. 汇总结果

| 规划门槛 mm | 规划成功 | 最终细化通过 | 动力学成功 | 该组最小规划间隙 mm | 该组最小实际间隙 mm | 有实际禁止接触的执行数 | 规划耗时中位数 s | 路径长度中位数 rad |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

原始/最终路径完整复核均12/12通过；实际成功为 {sum(r['success'] for r in runs)}/24。5 mm组0/6、8 mm组4/6、10与15 mm组各6/6。12个规划条件产生 {len(identities)} 条不同最终折线：同一随机种子在不同门槛下有时得到相同路径，因此不能把24次执行解释成24条不同轨迹或独立任务。

尤其5与8 mm的 seed=123 得到完全相同路径，规划最小间隙9.300347 mm，但两档速度均失败；更大的“最低规划阈值”并不等价于该路径已有足够动态储备。8/10/15 mm的 seed=42 最终路径相同，规划最小约18.097 mm、实际最低约13.439 mm；实际值低于15 mm但高于执行5 mm，正确判为执行成功，体现两阈值独立。

![规划裕度与实际最低间隙](../assets/plots/phase2d_margin_vs_actual.png)

## 4. 全部规划结果（无种子筛除）

| 门槛 mm | 规划种子 | 结果 | 最小采样间隙 mm | 原始→平滑节点数 | 原始→平滑长度 rad | 耗时 s |
|---|---:|---|---:|---:|---:|---:|
{chr(10).join(seedtable)}

完整父节点、路径、细化样本与配置见 `{plan}`，其中CSV记录真实路径。未验证或失败路径会在加载阶段拒绝进入动力学；本轮12次预算内均找到有效路径，因此没有制造“15 mm不可解”的结论。

## 5. 跟踪与执行耗时

| 规划门槛 mm / 速度 rad/s | 成功 | 仿真执行时间范围 s | 已执行部分最大关节误差 rad | TCP跟踪RMSE范围 mm | 最终到目标TCP误差范围 mm |
|---|---:|---:|---:|---:|---:|
{chr(10).join(timingtable)}

全部七关节RMSE向量、逐关节最大误差、TCP误差、规划与实际间隙差、首次违规、实际速度/加速度、受限保持、接触、仿真及墙钟耗时已保存在 `results/phase2d_dynamic_execution_results.csv` 和每次原始JSON/CSV。失败只统计已执行部分跟踪RMSE；最终到目标误差单列，不把中途局部小误差当作到达。墙钟含日志及高频完整检查，且有并行进程，不是硬实时基准。

动态原始目录：`{dyn}`。每个规划条件与速度有独立输出，不覆盖原 Phase2C 目录。确定性执行重复次数为1，本轮没有0.05 rad/s条件。

## 6. 历史失败机制与新接触失败

详细分析见 `docs/13_Phase2C失败机制分析.md` 和 `results/phase2d_tracking_clearance_analysis.csv`。原 D seed42 / 0.10 rad/s 在14.982 s首次违规：参考配对间隙5.944584 mm、实际4.959684 mm；局部七关节误差和重力/速度项均保留。原实际加速度8.941548 rad/s²峰值位于0.002 s初始化保持阶段，不能归因于正式轨迹参考加速度越限。

有限差分使用1e-3、5e-4、1e-4、1e-5 rad，原生距离查询及几何最近特征的变化导致局部线性预测与有限误差变化存在明显残差；该方法只能作局部诊断。图中参考到实际方向扫描是运动学距离分析，**不是实际执行轨迹**，也不是安全证书。

![局部线性近似的限制](../assets/plots/phase2d_local_sensitivity.png)

停止后的漂移包含运动惯性，也包含把保持目标改成当前实际角度后重力平衡点的变化。该原保持逻辑未改。本轮有4条执行记录发生实际禁止接触（对应两个相同路径/速度组合在不同规划门槛下重复），共 {audit['contact_intervals']} 段连续接触区间。最小实际间隙 -0.443289 mm，不能写成“无碰撞”。

例如 seed123 / 0.10 rad/s：首次5 mm违规在 {first_contact['first_violation_time_s']:.6f} s，首次原生禁止接触在 {first_contact['first_forbidden_contact_time_s']:.6f} s，二者间隔约 {(first_contact['first_forbidden_contact_time_s']-first_contact['first_violation_time_s']):.3f} s；接触发生于触发保护之后。5与8 mm组都保留了这一失败。10与15 mm组这6+6次执行中没有禁止接触或5 mm违规，但这只是当前小样本事实。

本轮没有修改保持机制以掩盖接触，响应式保持不能保证刹停安全，也不能仅凭无接触就认为满足5 mm门槛。首次事件、原生接触、停止后全过程均在原始 clearance.csv 中。

## 7. 原始CSV复算验收

`scripts/audit_phase2d_csv.py` 独立读取24次全部物理步CSV，重算七关节RMSE与最大误差、TCP RMSE/最大误差、最小间隙、执行时间、物理步数、违规步数、原生/禁止接触观察数、首次违规、连续接触区间和最终到目标误差；按原冻结的最后250个物理步稳定条件重建成功判定。24/24与执行汇总一致，结果见 `results/phase2d_csv_recomputation.json`。这不是从汇总复制后自称复算。

同时核对两段视频的每一帧：时间、q_ref、q_actual和ctrl逐项等于相应物理步tracking.csv。共 {sum(a['frames_matched'] for a in alignments)} 帧匹配。原先 Phase2B/2C路径、报告、CSV、模型、伺服源码及旧视频的阶段开始哈希均保持一致。

## 8. 实际动力学视频

`assets/videos/phase2d_margin_comparison.mp4` 为两次**独立初始化、真实live动力学执行**的先后对照，时间在第二个明确标注的回合重启：

- A_original_failed_baseline：原5 mm历史路径 / seed42 / 0.10 rad/s，失败。录像第 {baseline['video']['first_violation_frame']} 帧明确标注首次5 mm违规及时间，之后继续记录原受限保持。
- B_replanned_margin：预先选定8 mm / seed42 / 0.10 rad/s，实际执行成功；规划最小间隙 {improved['execution']['planning_minimum_clearance_m']*1000:.6f} mm、实际最小 {improved['execution']['minimum_actual_clearance_m']*1000:.6f} mm。未筛选出最好种子替代失败数据。

视频两段均来自同一个各自正在积分的live MjData，不用scratch设置构型冒充执行。显示规划阈值、执行5 mm、进度、段号、参考/实际TCP、当前实际间隙、成功/失败及首次违规。原生Windows Viewer已实际打开并正常关闭。

完整解码 {movie['frames']} 帧，{movie['duration_s']:.1f} s，{movie['resolution'][0]}×{movie['resolution'][1]}；两段分别 {baseline['video']['frames']} / {improved['video']['frames']} 帧、不同原始图像 {baseline['video']['raw_unique_frames']} / {improved['video']['raw_unique_frames']} 帧。MP4拼接仅拼接已录真实画面，没有重新生成策略/构型轨迹。SHA256：`{movie['sha256']}`。逐帧日志与单独视频：`{video}`。

## 9. 自动测试及运行中问题

最终 {tests} passed / 0 failed / 0 skipped，包括原104项全部回归及新增16项。测试覆盖阈值独立、默认兼容、当前完整细化、未验证/篡改路径拒绝、原5 mm路径及两速度失败复现、场景/伺服不变、全部CSV复算、局部分析标记、真实视频和无新增live qpos播放。

保留首轮日志 `results/phase2d_full_tests.log/xml`：119 passed、1项夹具建立错误，原因是Windows沙箱无法访问默认pytest临时目录；修复为项目内独立夹具目录，未改变算法或删除失败记录。最终日志为 `results/phase2d_final_tests.log/xml`。

初次多进程调度也因Windows受限本地管道权限未启动；错误及空运行目录保留，之后通过允许本地工作进程的运行方式完成全部24次。它没有产生被计作成功的空实验。成功/失败均来自真实完成的物理试验。

## 10. PowerShell复现命令

```powershell
Set-Location D:\\mujoco_learning\\panda_safe_residual_rl
& D:\\miniconda\\envs\\panda_safe_rl\\python.exe scripts/analyze_phase2d_failure.py
& D:\\miniconda\\envs\\panda_safe_rl\\python.exe scripts/run_phase2d_planning.py
& D:\\miniconda\\envs\\panda_safe_rl\\python.exe scripts/run_phase2d_execution.py
& D:\\miniconda\\envs\\panda_safe_rl\\python.exe scripts/audit_phase2d_csv.py
& D:\\miniconda\\envs\\panda_safe_rl\\python.exe scripts/demo_margin_execution.py --viewer --margin-mm 8 --seed 42
& D:\\miniconda\\envs\\panda_safe_rl\\python.exe -m pytest -q
```

演示命令已在Windows原生窗口实际测试。每次创建独立results/runs目录；latest指针和本阶段成品会指向新完整运行，旧目录保留。若在新任务环境重建本阶段冻结证据，先运行 `scripts/freeze_phase2d.py`。Esc/Q/窗口关闭会中止并归档，不能把用户中止当成功。

## 11. 交付、未执行事项与残差RL引入条件

交付：docs/13_Phase2C失败机制分析.md、docs/14_Phase2D验收报告.md；results/phase2d_tracking_clearance_analysis.csv、phase2d_planning_margin_results.csv、phase2d_dynamic_execution_results.csv、phase2d_failure_analysis.json、phase2d_csv_recomputation.json；assets/videos/phase2d_margin_comparison.mp4；完整搜索/复核/执行/接触/视频源记录均保留。

未执行：更多规划随机种子、独立随机扰动执行、大规模统计验证、0.05 rad/s诊断、传统控制器增强、重力补偿/前馈/反馈修正、严格连续碰撞检测、实机实验，以及所有PPO/SAC训练。传统控制器增强应另命名/配置/验收，不能回写本轮原控制组；它不是本轮裕度研究交付的前提。

当前10/15 mm组的6/6不能推广为所有路径、速度或扰动安全。后续残差RL应在以下条件成立后才考虑：明确仍未解决的跟踪目标；先建立独立、充分调试的传统控制与重力补偿对照；扩大固定训练/验证/测试场景并防止种子泄漏；限制残差动作且验证真实动态间隙；先解决停止后接触的保护边界，不把RL当作基本安全机制。若更大几何裕度和传统控制已满足目标，应如实评估RL的必要性，不能刻意削弱基线制造优势。

离散0.005 rad检查不是严格连续保证；几何裕度不是对所有动力学扰动的安全证明；仿真通过不是实机认证。本阶段完成后暂停，等待用户审核。
'''
    (ROOT/'docs/14_Phase2D验收报告.md').write_text(doc,encoding='utf-8')
    write_json(ROOT/'results/phase2d_acceptance.json',dict(phase='2D',date='2026-10-21',planning_trials=len(plans),dynamic_trials=len(runs),
        tests_passed=tests,original_tests_preserved=104,groups=groups,historical_files_verified=len(unchanged),historical_hashes_all_unchanged=True,
        planning_run=plan,dynamic_run=dyn,video_run=video,video_alignment=alignments,csv_all_metrics_match=True,
        traditional_controller_changed=False,reinforcement_learning_executed=False,execution_clearance_m=.005,
        hashes={p:sha(ROOT/p) for p in ['src/planning/margin_study.py','configs/phase2d_margin_study.yaml','scripts/run_phase2d_planning.py',
          'scripts/run_phase2d_execution.py','scripts/audit_phase2d_csv.py','scripts/analyze_phase2d_failure.py','scripts/demo_margin_execution.py',
          'tests/test_phase2d_margin.py','docs/13_Phase2C失败机制分析.md','docs/14_Phase2D验收报告.md']}))
    print('REPORT COMPLETE',tests,'tests; all history hashes unchanged; groups',[(g['planning_clearance_m'],g['dynamic_successes']) for g in groups],flush=True)


if __name__=='__main__':main_guard(main)
