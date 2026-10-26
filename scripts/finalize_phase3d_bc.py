"""Publish audited Phase3D-B reports from actual archived experiment records."""
from _bootstrap import ROOT, main_guard
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scripts.phase3c_common import read, rows
from src.common import sha, write_json


def latest(name):
    return Path(read(ROOT/f'results/phase3d_bc_{name}_latest.json')['run'])


def main():
    prepare=latest('prepare');training=latest('training');video=latest('video')
    analysis=read(ROOT/'results/phase3d_bc_analysis.json');trained=read(training/'complete.json');movie=read(video/'complete.json')
    history=read(prepare/'history_sha256.json');changed=[]
    for name,expected in history.items():
        p=ROOT/name
        if not p.exists() or sha(p)!=expected:changed.append(name)
    assert not changed, f'Historical content changed: {changed}'
    opening=read(latest('eval_final')/'final_test_opening.json')
    assert sha(ROOT/'configs/phase3d_bc.json')==opening['config_sha256']
    for name,expected in opening['controller_code_sha256'].items():assert sha(ROOT/name)==expected
    for c in opening['checkpoints'].values():assert sha(c['path'])==c['sha256']
    combined=ET.Element('testsuites');testcount=0
    for name in ('historical','new'):
        tree=ET.parse(ROOT/f'results/phase3d_bc_{name}_tests.xml')
        suites=[tree.getroot()] if tree.getroot().tag=='testsuite' else list(tree.getroot())
        for suite in suites:
            assert all(int(suite.get(k,0))==0 for k in ('errors','failures','skipped'))
            testcount+=int(suite.get('tests'));combined.append(suite)
    assert testcount==209
    ET.ElementTree(combined).write(ROOT/'results/phase3d_bc_all_tests.xml',encoding='utf-8',xml_declaration=True)
    manifest=rows(ROOT/'results/phase3d_bc_dataset_manifest.csv');evaluations=analysis['all_evaluations']
    unique={r['run_directory']:int(r['actual_physics_steps']) for r in manifest}
    unique.update({r['run_directory']:int(r['actual_physics_steps']) for r in evaluations})
    unique.update({r['result']['run_directory']:int(r['result']['physics_steps']) for r in movie['results']})
    assert len(unique)==79
    methods=['M3','Integral500','Integral20','BC78','BC85'];agg={}
    for split in ('validation','final'):
        agg[split]={}
        for method in methods:
            rr=[r for r in evaluations if r['split']==split and r['method']==method]
            agg[split][method]=dict(n=len(rr),success=sum(r['success'] for r in rr),tcp_rmse_mm=float(np.mean([r['tcp_rmse_mm'] for r in rr])),
                joint_rmse_rad=float(np.mean([r['joint_rmse_rad'] for r in rr])),final_tcp_mm=float(np.mean([r['final_tcp_mm'] for r in rr])),
                minimum_clearance_mm=min(r['minimum_clearance_mm'] for r in rr),violations=sum(r['safety_violation_steps'] for r in rr),
                prohibited_contacts=sum(r['prohibited_contact_observations'] for r in rr),own_state_action_mse=float(np.mean([r['own_state_target_mse'] for r in rr])))
    assert all(r['all_frames_match'] and r['tracking_exactly_matches_formal_evaluation'] for r in movie['results'])
    assert sha(ROOT/'assets/videos/phase3d_bc_vs_integral.mp4')==movie['sha256']
    acceptance=dict(passed=True,tests=testcount,historical_files_verified=len(history),historical_changes=changed,
        final_frozen_source_and_checkpoints_match=True,unique_dynamic_runs=len(unique),unique_experiment_physics_steps=sum(unique.values()),
        dataset_physics_steps=analysis['dataset_actual_physics_steps'],video_additional_physics_steps=movie['actual_video_execution_physics'],
        counts_exclude_automated_test_executions=True,ppo_optimizer_steps=0,aggregates=agg,video=movie['video'],video_frames=movie['frames'],
        video_duration_s=movie['duration_s'],video_all_tracking_matches=True,ready_for_automatic_ppo=False)
    write_json(ROOT/'results/phase3d_bc_acceptance.json',acceptance)
    plot(agg,evaluations)
    command='''```powershell
Set-Location D:\\mujoco_learning\\panda_safe_residual_rl
$py = 'D:\\miniconda\\envs\\panda_safe_rl\\python.exe'
# 播放真实动力学现场录像，不启动训练
Start-Process (Resolve-Path 'assets/videos/phase3d_bc_vs_integral.mp4')
# 复用3D-A播放器，完整记录状态重建，逐个窗口比较
& $py scripts/watch_ppo_training.py --mode comparison --replay-only --run results/runs/phase3d_bc_video_1792744038891360000/Integral500 --other-run results/runs/phase3d_bc_video_1792744038891360000/BC78 --other-run results/runs/phase3d_bc_video_1792744038891360000/BC85 --paused
```'''
    headless=command[command.index('& $py scripts/watch'):command.index('\n```')].replace(' --paused',' --headless-export')
    design_text=f'''# Phase 3D-B 专家示教与BC设计

本阶段只训练监督学习BC，未进行PPO更新。冻结协议见 `configs/phase3d_bc.json`，原控制器、模型、历史路径和日志保持原样；发布时重新核对了 {len(history)} 个历史文件的SHA256。

## 动作接口与因果时序

教师为原版 `IntegralM3Controller`（M3重力+速度前馈+受限积分），500 Hz。积分增益0.3/s、积分/残差范围±0.002 rad、残差速率0.01 rad/s、总位置指令速率0.35 rad/s，保留原生执行器力矩与关节限位。实际安全门槛仍为5 mm，规划裕度独立为20 mm。

在线积分I每个0.002 s物理步按当前参考与最新可观测关节位置之差更新；上一控制步总指令限速或原生力矩饱和时阻断相应积分分量；裁剪±0.002；回合开始与保护保持时清零。名义补偿模型只读取名义参数和可观测状态。额外的ObservableIntegral不访问教师对象，生成数据时逐物理步与原教师积分比对，全部差值为0。学生运行时使用自己的观测/控制状态执行同一规则。

物理第750步（1.5秒预保持后）开始，每25物理步构造一次观测并选择七维动作。标签是**当前因果期望积分I/0.002，下一段执行之前**，不是后续25步的未来积分，也不是经过残差限速后的执行残差。学生输出[-1,1]^7，映射为±0.002 rad目标，再经过原限速器、总指令限速器、位置伺服和mj_step。分别保存内部可观测积分、教师积分（仅教师记录）、期望残差、实际残差、最终ctrl和原生力矩。

原专家继续500 Hz更新，学生20 Hz持有目标。另设Integral20因果适配对照，避免把专家/学生频率差异误认为学习差距。所有闭环运动均来自data.ctrl与mj_step，动态阶段live qpos写入计数为0。

## 78维和85维输入

原78维保留关节位置/速度、当前参考位置/速度、误差、名义M3基础目标、三个历史误差、进度、上一动作和已施加残差。85维在末尾增加当前自己计算的七维I/0.002。两者均无真实质量、实际Kp或未来参考。对传感器延迟的处理遵循原控制器last observed状态；本pilot未额外加入噪声/延迟实验。

重要限制：这批教师示教中积分变化缓慢，期望残差与已施加残差恰好相同；经逐样本核对，标签也与78维中的上一实际残差字段相等。85维额外积分同样直接对应标签。这是可在线获得的反馈记忆，不是隐藏教师状态；但造成共线性和复制捷径，离线低MSE不能证明学会在自身状态上积累补偿。报告保留这一问题，不在看过测试后重新选标签或调模型。

## 隔离场景及预算

新建两条训练路径、一条验证路径、两条最终测试路径，障碍形状/位置、起终关节构型实际不同；完整参数、模型签名、规划种子和路径在 `{prepare.relative_to(ROOT)}`。RRT种子预算211、307；五条路径均首次211成功且通过0.005 rad独立复核，未启动第二种子搜索。

训练：nominal、Kp×0.95/0.90、质量及惯量×1.05、阻尼×1.10、组合(0.95/1.05/1.10)，两个物理种子4101/4102。验证：nominal、Kp×0.92、质量惯量×1.08、组合(0.94/1.07/1.08)，种子5101。最终：两个新布局×Kp×0.88、质量惯量×1.12、组合(0.90/1.10/1.15)，种子6101。未复用3C已开启的heldout_box作为新测试。

先4条pilot，再完成24训练+4验证示教。质量筛选冻结为：有效有限模型、稳定到达、无模型警告/控制异常/禁止接触/5 mm违规、残差范围正确、独立积分逐步一致。28条全部合格，0拒绝，原始记录全保留。8752样本：7716训练、1036验证；真实示教239536物理步。24训练记录仅12条独特观测—标签序列，另外12条为不同种子下确定性重复，不能视为24独立任务。最终集无示教标签参与优化。

## 监督训练及冻结

两层64单元Tanh，七维Tanh输出，输出层初始零；Adam 0.001，batch256，seed42，最大150 epoch，验证不改善25轮早停。输入均值/标准差仅用训练数据，标准差下限0.01，模型保存归一化参数。分别按独立验证动作MSE选择best，保存initial/best/final。BC78实际31轮961步优化，best第6轮；BC85实际37轮1147步优化，best第12轮。两次监督训练约3.54秒墙钟，不含采集和评价。

先冻结模型，再验证闭环；最终开启文件记录协议、BC源码和checkpoint哈希，此后未调参/重训/挑选。两模型实际各训练1个种子，不能声称跨训练种子稳定性。BC85不是现有78维Gym PPO的直接可加载权重，后续需单独验证85维适配、归一化及Tanh动作头兼容性。

## 运行及观看

{command}

播放器P/空格暂停，N/B逐决策前后，左右键逐记录，+/-调速，Esc/Q退出当前窗口；关闭一个比较窗口后打开下一个。该入口为Mode C Baseline Comparison，明确显示FULL RECORDED STATE RECONSTRUCTION；它恢复记录状态用于显示，不重新采样动作、不训练，也不冒充新的动力学执行。真正现场RGB来自上述MP4，五段依次M3、Integral500、Integral20、BC78、BC85。

实际执行过的无窗口复用验证：

```powershell
{headless}
```

该命令只将完整记录状态重建导出为比较视频；不将其称为训练现场RGB。GUI本轮未人工逐键验收，使用的是3D-A已有交互实现。

实际执行过的实验入口（重复运行会新建实验目录并消耗相同预算；本阶段已完成，无需再跑）：

```powershell
& $py scripts/prepare_phase3d_bc.py
& $py scripts/collect_phase3d_bc.py --stage pilot
& $py scripts/collect_phase3d_bc.py --stage complete
& $py scripts/train_phase3d_bc.py
& $py scripts/evaluate_phase3d_bc.py --stage validation
& $py scripts/evaluate_phase3d_bc.py --stage final
& $py scripts/analyze_phase3d_bc.py
& $py scripts/record_phase3d_bc.py
& $py -m pytest -q --ignore=tests/test_phase3d_bc.py --junitxml=results/phase3d_bc_historical_tests.xml
& $py -m pytest -q tests/test_phase3d_bc.py --junitxml=results/phase3d_bc_new_tests.xml
```

Windows多进程若受运行平台限制，需要允许本地Python创建工作进程；不需要安装新包或修改历史环境。依赖激活沿用项目现有只读路径机制。
'''
    (ROOT/'docs/27_Phase3D_B专家示教与BC设计.md').write_text(design_text,encoding='utf-8')
    table='| 方法 | 验证TCP RMSE mm | 最终TCP RMSE mm | 最终关节RMSE rad | 最终TCP到达误差均值 mm | 最终最小间隙 mm | 到达 | 违规/禁止接触 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
    for m in methods:
        a=agg['final'][m];v=agg['validation'][m]
        table+=f"| {m} | {v['tcp_rmse_mm']:.6f} | {a['tcp_rmse_mm']:.6f} | {a['joint_rmse_rad']:.8f} | {a['final_tcp_mm']:.6f} | {a['minimum_clearance_mm']:.6f} | {a['success']}/{a['n']} | {a['violations']}/{a['prohibited_contacts']} |\n"
    cases='| 最终布局/失配 | M3 | 积分500Hz | 积分20Hz | BC78 | BC85 |\n|---|---:|---:|---:|---:|---:|\n'
    for scene,condition in dict.fromkeys((r['scene'],r['condition']) for r in evaluations if r['split']=='final'):
        group={r['method']:r for r in evaluations if r['split']=='final' and r['scene']==scene and r['condition']==condition}
        cases+=f'| {scene}/{condition} | '+' | '.join(f"{group[m]['tcp_rmse_mm']:.6f}" for m in methods)+' |\n'
    hashes='\n'.join(f"- BC{r['dimension']} best SHA256：`{r['best_sha256']}`；final：`{r['final_sha256']}`。" for r in trained['results'])
    report=f'''# Phase 3D-B 验收报告

结论：已完成真实受限积分示教、78/85维监督学习和未见失配条件的真实闭环评价。85维优于78维，但仍未稳定复现原积分专家的精度；暂停本阶段，不启动BC初始化PPO。

## 真实执行清单与成本

- 五个新布局完成有界RRT搜索及0.005 rad复核；规划20 mm、执行5 mm，模型/障碍/关节顺序保存到独立配置。
- 示教28次（24训练+4验证），8752样本，239536真实物理步；24训练中仅12独特轨迹。均保留完整tracking/control_signals/clearance/events、样本NPZ和参数配置，无被筛掉的失败回合。
- BC78/85各1个训练种子42；最佳验证归一化动作MSE分别0.0009228787/0.0005107658。完整训练曲线及initial/best/final checkpoint均保留。
- 验证20方法工况结果，其中原专家4次复用验证示教，新增16次执行；最终30次执行，全部预先冻结且未按结果排除工况。
- 录像新增5次真实闭环执行，不计为独立测试任务。去重后本研究共{len(unique)}次动力学执行，**{sum(unique.values())}个mj_step**（含初始化保持、轨迹、最终保持；不含自动测试产生的额外仿真）。不能把复用示教与评价重复计费。
- 197个历史测试+12个新增测试，**209项真实通过，0失败/跳过**。历史197用时212.81秒、新增12用时57.60秒。JUnit分别保留，合并索引 `results/phase3d_bc_all_tests.xml`。
- 本阶段PPO优化步为0，没有启动任何PPO训练。

## 公平闭环结果

同一行方法使用相同路径、物理参数和种子。主RMSE覆盖trajectory+settle，全记录（含预保持）RMSE另列CSV；未过滤失败回合。成功判据沿用原配置：最大最终关节误差≤0.02 rad、TCP≤10 mm、速度≤0.02 rad/s连续保持0.5秒，并全程满足5 mm和禁止接触要求。此成功门槛远宽于亚毫米精度差异，不能用同样100%到达率掩盖精度差距。

{table}

以下均为TCP轨迹RMSE，单位mm：

{cases}

BC85最终六工况均优于M3和BC78，均逊于原500Hz积分专家；平均比M3降低约53.6%，比专家仍高约25.0%。Integral20均值0.397613 mm，与500Hz专家0.403321 mm接近，不能把主要BC差距归因于20Hz接口本身。

验证并非处处改善：nominal下M3/专家/BC78/BC85分别0.122850/0.216808/0.799902/0.269173 mm；gain92下BC78(0.490140)略优于BC85(0.514398)。保留这些负例，未据此更改模型。最终六工况全安全不等于更广泛状态中的安全保证。

## 监督误差与状态分布

离线验证MSE低，并未保证学生闭环模仿准确。学生自己访问状态上的“动作—其自身因果积分目标”平均MSE，最终BC78为0.29899694、BC85为0.03049704（归一化动作单位）；它们不是隐藏教师在学生状态的反事实真实结果，但可诊断控制规则偏离。实际轨迹误差仍由真实tracking独立计算。

示教标签与已施加残差字段共线，78维可通过复制历史输出获得低监督误差，却不持续正确积累补偿。85维增加自身在线积分能改善闭环，但也使模仿部分成为对可计算积分规则的函数近似；不能宣称学到了超越传统积分的原创控制算法。训练几何只有两条、监督种子仅一个、物理种子重复无额外噪声，因此不作统计显著性/广泛鲁棒性结论。

## 动作、安全与视频验证

每物理步保留原安全检查、native接触和原力矩限制。独立审计所有50评价记录的实际ctrl限速≤0.35 rad/s、有效残差变化率≤0.01 rad/s、目标/实际残差≤0.002 rad及关节/力矩范围；未通过修改live qpos驱动机器人。全部原始误差/间隙可由CSV重算，原始控制日志中的rate_limited布尔字段与新统计effective_command_slew_steps区分保存，不把布尔字段当精确介入次数。

`assets/videos/phase3d_bc_vs_integral.mp4`：1000帧、10fps、100秒，五段每段200帧；每段从当前live MjData录制，保存完整integration state及RGB帧索引。逐帧状态与tracking一致，五次录制的tracking与对应正式执行逐项相同；机械臂关节实际位移约1.01 rad。保留原始各段MP4，视频SHA256 `{movie['sha256']}`。此视频是BC/传统控制闭环执行，不是随机PPO训练。

已实际通过旧观看系统Mode C的headless-export，输出到 `results/runs/phase3d_viewer_34736_1792744481979136900`。该导出是完整记录状态重建，与训练现场RGB严格区分。GUI按键本轮未重新人工逐键验收。

## 哈希、归档及已知运行问题

历史{len(history)}文件SHA256再次核验无变化；最终开启时冻结的BC源码、配置及模型哈希均一致。协议SHA256 `{opening['config_sha256']}`。模型保存在 `{training.relative_to(ROOT)}`。

{hashes}

独立目录包含示教/验证/最终/录像的所有原始记录。首次监督入口因torch导入先于依赖激活失败（未发生训练），保留 `results/phase3d_bc_training.log`；修正导入顺序后真实成功，日志 `results/phase3d_bc_training_fixed.log`。没有删除失败日志、安装包或修改历史模型。未出现动态控制失败，未人为制造失败视频。

## 六个研究问题及未完成事项

1. **能否学会补偿？** 在示教分布上能预测动作，两个BC也均真实闭环到达；但78维存在明显分布偏移，85维仍有专家精度差距。
2. **哪种观测有效？** 本单种子pilot中85维的独立验证MSE、最终六工况闭环误差更好；不能预设任意任务均如此。
3. **离线与闭环一致吗？** 排序整体相符，但离线极低误差明显低估闭环规则偏离；共线性及自身状态分布必须同时检查。
4. **未见模型失配能否控制？** 在两个未见布局及三种预先冻结失配中均到达、无违规，85维六工况优于M3；范围仅限本实验。
5. **距专家多少？** 最终均值0.504066对0.403321 mm，85维约差25%；78维0.716125 mm。并未超过原积分专家。
6. **是否开始PPO？** 暂不开始。先增加不重复的几何/状态覆盖、多监督种子，预先设定精度保持门槛，必要时单独申请有界学生状态补充示教；验证85维Gym观测、归一化及Actor输出头兼容性、零残差/M3等价，再决定BC初始化PPO。上述后续实验均**未执行**。

未执行：DAgger、更多训练种子、大量噪声/延迟随机化、PPO初始化或微调、实机验证、连续时间安全证明。本轮新GUI未做人工键盘验收，已有播放器只完成真实数据的命令行导出验收。规划离散复核/仿真无碰撞不构成实机安全认证。

## 直接观看

{command}

详细设计、数据划分、操作键与复现预算见 `docs/27_Phase3D_B专家示教与BC设计.md`；CSV见results同名前缀文件；发布验收 `results/phase3d_bc_acceptance.json`。完成后暂停，等待审核，不自动进入Phase3D-C。
'''
    (ROOT/'docs/28_Phase3D_B验收报告.md').write_text(report,encoding='utf-8')
    artifacts=[ROOT/'docs/27_Phase3D_B专家示教与BC设计.md',ROOT/'docs/28_Phase3D_B验收报告.md',ROOT/'configs/phase3d_bc.json']
    artifacts+=list((ROOT/'results').glob('phase3d_bc*.csv'))+list((ROOT/'results').glob('phase3d_bc*.json'))
    artifacts+=list((ROOT/'assets/plots').glob('phase3d_bc*.png'))+[ROOT/'assets/videos/phase3d_bc_vs_integral.mp4']+list(training.glob('*.pt'))
    artifacts=[p for p in artifacts if p.name!='phase3d_bc_release_manifest.json']
    write_json(ROOT/'results/phase3d_bc_release_manifest.json',dict(artifacts=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in artifacts],
        protocol=opening,acceptance=acceptance))
    print('FINALIZED',testcount,'tests',len(history),'historical files',len(unique),'runs',sum(unique.values()),'physics steps',flush=True)


def plot(agg,evaluations):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods=list(agg['final']);fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
    means=[agg['final'][m]['tcp_rmse_mm'] for m in methods]
    ax.bar(methods,means,alpha=.65)
    for i,m in enumerate(methods):
        y=[r['tcp_rmse_mm'] for r in evaluations if r['split']=='final' and r['method']==m]
        ax.scatter(i+np.linspace(-.12,.12,len(y)),y,s=25,c='black')
    ax.set(ylabel='TCP trajectory RMSE (mm)',title='Frozen final cases: 2 new geometries x 3 mismatches; no PPO')
    ax.grid(axis='y',alpha=.2);fig.savefig(ROOT/'assets/plots/phase3d_bc_closed_loop_comparison.png',dpi=150);plt.close(fig)


if __name__=='__main__':main_guard(main)
