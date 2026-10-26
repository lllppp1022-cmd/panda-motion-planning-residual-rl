"""Evidence-only report, uncertainty failure diagnosis and scientific figures."""
from _bootstrap import ROOT,main_guard
import csv,json,xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.common import write_json,sha
from scripts.run_phase3a import check_lock,prepared,read,frozen_sources

def rows(p):
    with Path(p).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def main():
    check_lock();p=prepared();history=read(p/'history_sha256.json');changed=[name for name,h in history.items() if not (ROOT/name).exists() or sha(ROOT/name)!=h]
    if changed:raise ValueError('Historical files changed: '+str(changed))
    audit=read(ROOT/'results/phase3a_csv_audit.json');data=rows(ROOT/'results/phase3a_controller_comparison.csv');groups=defaultdict(list)
    assert audit['all_metrics_match'] and audit['all_pairs_fair']
    for r in data:groups[r['method']].append(r)
    num=lambda r,key:float(r[key])
    success=lambda r:r['success']=='True'
    summaries=[]
    for method,rs in sorted(groups.items()):
        ok=[r for r in rs if success(r)]
        summaries.append(dict(method=method,runs=len(rs),success=len(ok),failures=len(rs)-len(ok),
            gap_violation_runs=sum(r['hard_violation']=='True' for r in rs),contacts=sum(int(r['prohibited_contact_intervals']) for r in rs),
            minimum_gap_m=min(num(r,'minimum_actual_clearance_m') for r in rs),saturation_steps=sum(int(r['saturation_steps']) for r in rs),
            successful_tcp_rmse_median_m=float(np.median([num(r,'tcp_rmse_m') for r in ok])),
            successful_final_tcp_median_m=float(np.median([num(r,'final_tcp_error_m') for r in ok])),
            successful_final_tcp_max_m=max(num(r,'final_tcp_error_m') for r in ok),
            actual_max_velocity_rad_s=max(num(r,'max_velocity_rad_s') for r in rs),actual_max_acceleration_rad_s2=max(num(r,'max_acceleration_rad_s2') for r in rs),
            extra_cost_median_us_per_step=float(np.median([num(r,'controller_extra_us_per_step') for r in rs])),
            joint_rmse_mean_rad=np.mean([json.loads(r['joint_rmse_rad']) for r in ok],axis=0).tolist()))
    bycondition=[]
    for name in read(ROOT/'configs/phase3a_pilot.json')['perturbations']:
        for method in ('M1','M3'):
            rs=[r for r in data if r['method']==method and r['perturbation']==name['name']];ok=[r for r in rs if success(r)]
            bycondition.append(dict(perturbation=name['name'],method=method,runs=len(rs),success=len(ok),
                gap_violations=sum(r['hard_violation']=='True' for r in rs),contact_intervals=sum(int(r['prohibited_contact_intervals']) for r in rs),
                successful_tcp_rmse_median_m=float(np.median([num(r,'tcp_rmse_m') for r in ok])) if ok else None,
                successful_final_tcp_median_m=float(np.median([num(r,'final_tcp_error_m') for r in ok])) if ok else None))
    paired=rows(ROOT/'results/phase3a_paired_fairness.csv');both=[r for r in paired if r['M1_success']==r['M3_success']=='True']
    improvements=[1-float(r['M3_tcp_rmse_m'])/float(r['M1_tcp_rmse_m']) for r in both]
    nominalpairs=[r for r in both if r['perturbation']=='nominal']
    # Oracle calculations are offline diagnostics, never passed to the controller.
    import mujoco
    from src.control.gravity_compensation import GravityCompensation
    offline=[]
    for perturbation in ('gain_low','mass_high','combined'):
        row=next(r for r in data if r['scene']=='D_two_obstacles' and r['method']=='M1' and r['perturbation']==perturbation and r['disturbance_seed']=='3101')
        d=Path(row['run_directory']);summary=read(d/'phase3a_summary.json');first=summary['first_violation'];q=np.array(first['q'])
        nominal=mujoco.MjModel.from_binary_path(str(d/'nominal_controller_model.mjb'));physics=mujoco.MjModel.from_binary_path(str(d/'perturbed_physics_model.mjb'))
        gn,gp=GravityCompensation(nominal),GravityCompensation(physics)
        delta=gp.query(q)/gp.kp-gn.query(q)/gn.kp
        after=[x for x in rows(d/'tracking.csv') if float(x['time'])>=first['time']-1e-9]
        qactual=np.array([[float(x[f'q_actual_{j}']) for j in range(1,8)] for x in after])
        offline.append(dict(perturbation=perturbation,first_violation=first,offline_oracle_offset_difference_rad=delta,
            max_additional_joint_motion_after_stop_rad=float(np.max(abs(qactual-q))),
            minimum_gap_after_stop_m=min(float(x['minimum_clearance_m']) for x in after),
            warning='Oracle physical parameters only for offline diagnosis; local static offset estimate, NOT global safety guarantee'))
    failure=dict(summary=summaries,by_perturbation=bycondition,failed_runs=[r for r in data if not success(r)],
        invalid_runs=audit['invalid_runs'],user_aborts=[r for r in data if r['status']=='user_abort'],offline_diagnostics=offline,
        paired_successful_conditions=len(both),median_tcp_rmse_reduction_paired_success=float(np.median(improvements)),
        nominal_paired_rmse_reductions=[dict(scene=r['scene'],fraction=1-float(r['M3_tcp_rmse_m'])/float(r['M1_tcp_rmse_m'])) for r in nominalpairs],
        recommendation='暂不需要RL：先完成增益/负载辨识或受限积分补偿对照；允许以后独立设计残差研究，但本pilot未证明其必要性',
        limitations=['four paths only','two seeds; deterministic factors repeated are not independent scenarios','supervisory sensor effects only',
            'no integral/adaptive/robust MPC baseline','no lower-gain/mass opposite-direction sweep','no RL training','no real-robot safety certification'],
        history_unchanged=len(history),source_hashes=frozen_sources())
    write_json(ROOT/'results/phase3a_failure_analysis.json',failure)
    plots=ROOT/'assets/plots';plots.mkdir(exist_ok=True)
    labels=[x['name'] for x in read(ROOT/'configs/phase3a_pilot.json')['perturbations']]
    fig,axes=plt.subplots(1,2,figsize=(14,5));x=np.arange(len(labels))
    for offset,method,color in ((-.18,'M1','#3c77b3'),(.18,'M3','#df8a32')):
        rs=[next(r for r in bycondition if r['perturbation']==name and r['method']==method) for name in labels]
        axes[0].bar(x+offset,[r['success']/r['runs'] for r in rs],.36,label=method,color=color)
        axes[1].bar(x+offset,[1000*r['successful_tcp_rmse_median_m'] if r['successful_tcp_rmse_median_m'] is not None else 0 for r in rs],.36,label=method,color=color)
    for ax in axes:ax.set_xticks(x,labels,rotation=35,ha='right');ax.legend();ax.grid(axis='y',alpha=.25)
    axes[0].set_ylabel('Goal completion fraction');axes[0].set_ylim(0,1.1)
    axes[1].set_ylabel('TCP trajectory RMSE, successful runs only (mm)')
    fig.suptitle('Phase3A bounded pilot: paired nominal / uncertain physics; failures retained')
    fig.tight_layout();fig.savefig(plots/'phase3a_robustness_summary.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
    for method in ('M1','M3'):
        r=next(r for r in data if r['scene']=='validation_cylinder' and r['perturbation']=='combined' and r['disturbance_seed']=='3101' and r['method']==method)
        t=rows(Path(r['run_directory'])/'tracking.csv');times=[float(z['time']) for z in t]
        axes[0].plot(times,[1000*float(z['minimum_clearance_m']) for z in t],label=method)
        axes[1].plot(times,[1000*float(z['tcp_error_m']) for z in t],label=f"{method}: {r['status']}")
    axes[0].axhline(5,color='red',ls='--',label='Actual hard threshold 5 mm');axes[0].set_ylabel('Actual minimum gap (mm)')
    axes[1].set_ylabel('TCP tracking error (mm)');axes[1].set_xlabel('Actual simulation time (s)')
    for ax in axes:ax.legend();ax.grid(alpha=.25)
    fig.suptitle('Validation cylinder: fixed combined uncertainty, seed3101; actual physics CSV')
    fig.tight_layout();fig.savefig(plots/'phase3a_combined_tracking.png',dpi=150);plt.close(fig)
    video=Path(read(ROOT/'results/phase3a_video_latest.json')['run']);vm=read(video/'complete.json')
    assert vm['frames']>1 and len(vm['episodes'])==3 and all(e['validation']['all_frames_match_actual_CSV'] and not e['validation']['aborted'] for e in vm['episodes'])
    tests=ET.parse(ROOT/'results/phase3a_release_tests.xml').getroot();suite=tests.find('testsuite') if tests.tag!='testsuite' else tests
    assert int(suite.attrib.get('failures',0))==int(suite.attrib.get('errors',0))==int(suite.attrib.get('skipped',0))==0
    testcount=int(suite.attrib['tests'])
    table='\n'.join(f"|{r['method']}|{r['success']}/{r['runs']}|{r['gap_violation_runs']}|{r['contacts']}|{r['minimum_gap_m']*1000:.4f}|{r['successful_tcp_rmse_median_m']*1000:.4f}|{r['successful_final_tcp_median_m']*1000:.6f}|{r['extra_cost_median_us_per_step']:.2f}|" for r in summaries)
    conditiontable='\n'.join(f"|{r['perturbation']}|{r['method']}|{r['success']}/{r['runs']}|{r['gap_violations']}|{r['contact_intervals']}|{r['successful_tcp_rmse_median_m']*1000:.4f}|" for r in bycondition)
    scene_table=[]
    for split in ('historical','train','validation','test'):
        for method in ('M1','M3'):
            rs=[r for r in data if r['split']==split and r['method']==method]
            scene_table.append(f"|{split}|{method}|{sum(success(r) for r in rs)}/{len(rs)}|{sum(r['hard_violation']=='True' for r in rs)}|")
    plans=read(p/'plans.json');planstable='\n'.join(f"|{r['layout']}|{r['split']}|{r['seed']}|{r['success']}|{r['minimum_clearance_m']*1000:.4f}|{r['planning_s']:.3f}|" for r in plans)
    jointstable='\n'.join(f"|{r['method']}|"+'|'.join(f'{v:.6f}' for v in r['joint_rmse_mean_rad'])+'|' for r in summaries)
    diagnostics='\n'.join(f"- {x['perturbation']}：首次违规t={x['first_violation']['time']:.3f}s；停止后最大额外单关节运动{x['max_additional_joint_motion_after_stop_rad']:.6f}rad；停止后最小间隙{x['minimum_gap_after_stop_m']*1000:.4f}mm；离线静态补偿偏差最大{max(abs(x['offline_oracle_offset_difference_rad'])):.6f}rad。" for x in offline)
    nominaltext='；'.join(f"{r['scene']} {100*r['fraction']:.1f}%" for r in failure['nominal_paired_rmse_reductions'])
    report=f'''# Phase 3A 验收报告：模型不确定性与强传统控制基线

日期：2026-10-22。工作目录：`{ROOT}`。阶段已完成并暂停，没有训练PPO/SAC。

## 核心结论

**建议：暂不需要RL。** M3能显著减小名义模型下的运动跟踪滞后，但固定名义重力补偿不能抵消实际Kp、质量和惯量失配。当前发现的是可解释的模型偏差，尚未排除负载/增益辨识、受限积分或自适应补偿等传统方法；不能由“M1/M3在紧间隙路径失败”直接推出RL必要。后续若做残差PPO，应视为独立研究假设而非已证明的必需方案。

M3适合作为当前更强的运动跟踪传统基线，M1继续保留为零残差等价基线。未来若研究M1+残差，必须同时比较M3，不能仅超过较弱的M1就声称学习方法优越；模型失配下的最终静差仍需单独处理。

## 历史与冻结证据

准备目录：`{p}`。{len(history)}个冻结历史文件在交付时逐一SHA256复核完全一致。原模型、M0/M1/M2、RRT、旧结果和视频未改动。模型隔离、实验配置和控制器源码在本轮执行前冻结，每次读取核验。最初一次开发准备和静态目录保留；修正Python模块导入方式后重新冻结并重复静态门禁，再执行正式pilot。第一次进程池在Windows沙箱因管道权限失败，未执行动力学；获批本机运行后启动正式有界任务，启动失败日志未删除。

M0/M2未重跑辅助矩阵；主比较只使用M1/M3。共{audit['actual_runs']}次正式真实动力学执行，{audit['paired_conditions']}个配对条件，{audit['valid_recomputed']}次有效CSV复算，无效模型/运行{len(audit['invalid_runs'])}次。另有4次正式静态保持与3次视频演示执行，视频不计入正式样本；开发期更早的4次静态保持也真实执行并保留，但不重复计入正式静态门禁。名义每路径1种子，其他条件2种子。纯参数确定性条件中的重复是执行复现，不能当作独立随机场景或统计显著性证据。

## 模型隔离与控制推导

详见[冻结实验设计](18_Phase3A扰动实验设计.md)。实际物理模型独立复制，Kp变化同步改变gain和位置bias；质量/主惯量共同按比例缩放，保持物理一致性；无关节范围、原生力矩、碰撞几何、重力常数或积分器放宽。两份MJB及全部参数逐次保存。控制器只获得名义模型、参考与延迟/带噪观测，不获得实际扰动参数。实际模型只给积分与安全监控。名义物理副本也经过同样常量初始化，不以逐位复现Phase2E全部轨迹为结论。

位置伺服未饱和映射：`tau=Kp*(ctrl-q)-Kd*qvel`。M1为名义重力前馈`g_nom/Kp_nom`，限±0.02rad；M3另加`Kd_nom/Kp_nom*dq_ref`，限±0.02rad。比值单位秒、符号为正，抵消参考运动方向上的阻尼。实现用相邻已收到q_ref的后向差分，不读取未来真实状态；保护保持时速度项清零。随后总目标按原关节/ctrl范围裁剪、原0.35rad/s限速、真实data.ctrl写入及mj_step积分，保留±87Nm（前四轴）/±12Nm（后三轴）。

观测噪声和观测延迟仅作用上层补偿；MuJoCo原生内环仍使用内部真实状态。动作延迟指上层位置目标FIFO。没有试验底层编码器或速度传感器延迟，不能将本结果推广到完整硬件观测链。M1/M3每步的高斯噪声序列在共同执行区间完全相同；差异不是随机数重采样造成的。

## 新布局与分组

|布局|用途|规划种子|通过|最小规划采样间隙mm|规划秒|
|---|---|---:|---|---:|---:|
{planstable}

每个新布局均为不同障碍形状、位置和起终点构型；不是对旧轨迹重命名。均由本项目RRT-Connect规划，Shortcut后独立0.005rad复核，新规划门槛10mm。历史D保持原约5.756mm路径。实际执行始终5mm，全部274对、地面、自碰撞、夹爪及原生接触逐物理步监控，未验证路径禁止进入执行。完整配置、直线碰撞证据、树记录、路径CSV和当前复核保留在准备目录。

train/validation的90次pilot先完成，再打开test的30次；测试开启文件记录源码/配置哈希，开启后无控制器调参。没有RL训练；train名称只表示以后可能的任务用途。此次测试结果已打开，未来若据此改算法，应另建最终未见布局，不能重复称这些布局未见测试集。

## 静态门禁

两个合法无障碍构型，M1/M3各2次，四次全部稳定到达、无饱和。零参考速度时M3退化为M1。名义匹配模型的静态最终TCP误差约1.5–1.7e-13m，只是理想数值模型的平衡精度，不是实机精度。每次真实积分超过500步；保留初始化瞬态、实际速度/加速度及力矩，不用qpos写入消除误差。

## 正式总体结果

|方法|到达|5mm违规回合|禁止接触连续区间|最小实际间隙mm|成功回合TCP轨迹RMSE中位mm|成功最终TCP中位mm|新增计算中位us/步|
|---|---:|---:|---:|---:|---:|---:|---:|
{table}

TCP误差中位数只涵盖成功回合；失败回合仍完整列在CSV中，不能凭其早停后较短轨迹的低RMSE判优。所有正式回合均保留实际最终TCP到原目标的误差、完成进度和失败原因，不将保护保持参考替换后的局部低误差算成到达。额外计算耗时在本机三个工作进程及部分录像并发负载下测量，不作为硬实时或硬件无关的性能保证。

|扰动|方法|到达|5mm违规|接触区间|成功TCP RMSE中位mm|
|---|---|---:|---:|---:|---:|
{conditiontable}

|场景分组|方法|到达|5mm违规|
|---|---|---:|---:|
{chr(10).join(scene_table)}

共同成功的配对条件中，M3相对M1的TCP轨迹RMSE降低比例中位数为{100*np.median(improvements):.2f}%。名义配对分别为：{nominaltext}。这是当前任务的描述统计，不能称为统计显著、通用鲁棒或超过全部传统控制方法。

成功回合七关节RMSE均值（rad）：

|方法|J1|J2|J3|J4|J5|J6|J7|
|---|---:|---:|---:|---:|---:|---:|---:|
{jointstable}

三类安全概念分别记录：无原生禁止接触不代表满足5mm；离散路径复核不代表连续安全；名义模型上的预测不代表未知物理扰动下安全。实际速度/加速度峰值、力矩及命令限速计数、每回合墙钟时间见comparison.csv，不将参考vmax/amax当作实际硬保证。最高加速度包含开始保持的重力瞬态。

## 失败机制与停止后的运动

{diagnostics}

上述离线“实际参数理想补偿量”只用于解释，不输入控制器。Kp降低后相同位置偏移产生更少力矩，质量增加后所需重力力矩更大；M3速度项在静止时为零，因此不会消除这些稳态偏差。首次5mm违规后保持原失效判据及0.5s受限保持，仍记录惯性/重力引发的额外运动。失败详情（全部种子）见`results/phase3a_failure_analysis.json`，不只展示成功。

## 零残差、残差边界与M2定位

自动测试在相同物理状态、名义参数和无扰动条件下，对新M1与原CompensatedJointController逐物理步比较qpos/ctrl，零扩展50步完全一致。未来残差应在M1正确前馈之后加入，零残差不得更换观测、延迟、随机采样或限速顺序；应对完整新环境再做逐步零残差等价回归。

**目前没有经过验证的非零残差安全幅度。** 5mm要求对全身几何，不能把固定0.002rad或任何角度自动换算为全局安全。可以把±0.002rad作为未来离线候选上限，仍须逐状态/短预测检查、原目标限速和原生力矩限幅，并核验动态停止距离；它未被本实验执行或证明安全。需要在匹配名义控制、轻度合理失配且有额外几何裕度的布局中，与受限积分/辨识对照后再决定是否研究残差。

M2在Phase2E的两次停止发生于M1本可完成的路径，7mm预测预警与5mm实际门槛加短时模型造成保守成本。原Phase2E预测器会复制传入的执行模型；若直接接入本阶段扰动物理模型，就会得到实际参数，形成oracle信息。本阶段没有运行这条路径。未来非oracle保护应单独接入名义预测模型并验证失配和延迟误差，此时预测不再天然准确。M2保留为可选、单独校准的辅助实验，不作为默认RL控制器或未知扰动安全盾；本阶段未改M2，也未宣称预测覆盖新失配。

## 复算、测试及视频

`scripts/audit_phase3a.py`从实际tracking/control_signals逐行独立计算全部{audit['valid_recomputed']}回合RMSE、目标、接触区间、间隙、速度加速度、计算耗时，核验实际力矩与限幅映射，复核同路径/同模型/同随机数配对，共{audit['paired_conditions']}对通过。无效运行与人为中止按状态单列，未剔除安全失败。

原137项及新增测试实际执行，合计**{testcount}项通过，0失败、0跳过**。日志和JUnit为`results/phase3a_release_tests.log/xml`。开发期1项脚本导入失败保留在unit_tests日志，修复后13项单元测试先通过，再启动正式pilot。随后历史清单测试因Windows默认GBK无法读取UTF-8中文文件名失败，改为明确UTF-8读取，原失败日志保留在preheldout_tests中；没有放宽任何物理/安全限制或历史哈希断言。

视频：`assets/videos/phase3a_robustness_comparison.mp4`，{vm['frames']}帧、{vm['frames']/10:.1f}s。固定validation_cylinder依次演示M1名义、M1组合扰动、M3组合扰动，每段独立初始化并明确标注，所有画面来自真实live MjData的ctrl/mj_step执行，逐帧q_ref/q_actual/ctrl与实际CSV完全对应，并检查动态位移及解码。原生Windows Viewer已启动并正常关闭。视频重执行不混入正式成功率。

```powershell
Set-Location D:\\mujoco_learning\\panda_safe_residual_rl
& D:\\miniconda\\envs\\panda_safe_rl\\python.exe scripts/demo_robustness.py --viewer
& D:\\miniconda\\envs\\panda_safe_rl\\python.exe scripts/audit_phase3a.py
& D:\\miniconda\\envs\\panda_safe_rl\\python.exe -m pytest -q
```

## 未完成与边界

- 未训练PPO/SAC；未创建RL环境或验证残差非零安全幅度。
- 未测试负向质量误差、不同幅度、底层传感器/速度噪声、随机外力、实机、积分/自适应/MPC等更多传统基线。
- 四条路径、两个扰动种子不足以证明泛化；测试数据已开启，后续调参需新未见任务。
- 没有连续碰撞或形式化安全保证；实际接触与间隙违规均以真实记录为准。
- 强名义M3和隔离接口已交付；优先研究辨识及受限积分补偿，再审查RL必要性。本阶段到此暂停等待用户审核。
'''
    (ROOT/'docs/19_Phase3A验收报告.md').write_text(report,encoding='utf-8')
    required=['docs/18_Phase3A扰动实验设计.md','docs/19_Phase3A验收报告.md','results/phase3a_nominal_vs_perturbed.csv',
              'results/phase3a_controller_comparison.csv','results/phase3a_failure_analysis.json','assets/videos/phase3a_robustness_comparison.mp4']
    write_json(ROOT/'results/phase3a_acceptance.json',dict(tests_passed=testcount,groups=summaries,actual_formal_runs=len(data),history_files_unchanged=len(history),
        prepare_run=str(p),video_run=str(video),pilot_run=read(ROOT/'results/phase3a_pilot_latest.json')['run'],test_run=read(ROOT/'results/phase3a_test_latest.json')['run'],
        paired_audit=audit,video_frames=vm['frames'],recommendation=failure['recommendation'],rl_executed=False,
        required_files={name:dict(sha256=sha(ROOT/name),bytes=(ROOT/name).stat().st_size) for name in required}))
    print('REPORT COMPLETE',summaries,flush=True)

if __name__=='__main__':main_guard(main)
