"""Generate Phase2B report and figures only from completed run artifacts."""
from _bootstrap import main_guard
import csv,json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.common import ROOT,write_json,sha


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    run=Path(read(ROOT/'results/phase2b_batch_latest.json')['run']);summary=read(run/'phase2b_performance.json')
    pilot=Path(read(ROOT/'results/phase2b_pilot_latest.json')['run'])
    demo=Path(read(ROOT/'results/phase2b_demo_latest.json')['run']);video=read(demo/'complete.json')
    regression=Path(read(ROOT/'results/phase2b_regression_latest.json')['run'])
    with (run/'phase2b_planning_results.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    protected=read(ROOT/'results/phase2b_prior_snapshot.json')
    check={p:sha(ROOT/p)==h for p,h in protected.items()}
    assert all(check.values()),check
    tests=ET.parse(ROOT/'results/phase2b_full_tests.xml').getroot()
    testcases=list(tests.iter('testcase'));failures=list(tests.iter('failure'));errors=list(tests.iter('error'))
    assert not failures and not errors and len(testcases)==89
    representative=[s for s in summary['scene_summaries'] if not s['scene'].startswith('F_')]
    successful=[r for r in rows if r['success']=='True']
    assert summary['trials']==80 and len(successful)==49
    failed=[r for r in rows if r['failure_reason']=='smoothed_fine_validation_failed']
    assert len(failed)==1
    rejected=read(failed[0]['record_path'])
    evidence=read(demo/'source_and_validation.json');selected=read(evidence['source'])['result']
    assert not evidence['direct']['valid'] and selected['success'] and selected['final_validation_passed']
    assert video['native_viewer_launched'] and video['viewer_closed'] and not video['interrupted']
    assert video['decoded_frames']==video['expected_frames'] and video['raw_unique_frames']>1 and video['physics_steps']==0
    plotdir=ROOT/'assets/plots';plotdir.mkdir(exist_ok=True,parents=True)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    labels=[s['scene'].split('_',1)[0] for s in representative];x=np.arange(len(labels))
    axs[0].bar(x-.17,[s['direct_fine_time_s'] for s in representative],.34,label='Direct .005 rad check')
    axs[0].bar(x+.17,[s['successful_planning_time_median_s'] for s in representative],.34,label='RRT + shortcut + fine audits')
    axs[0].set(ylabel='Wall time (s)',xticks=x,xticklabels=labels,title='Actual desktop timings (not hard real time)');axs[0].legend(fontsize=8)
    axs[1].bar(x-.17,[s['raw_length_median_rad'] for s in representative],.34,label='Raw path median')
    axs[1].bar(x+.17,[s['shortcut_length_median_rad'] for s in representative],.34,label='Shortcut path median')
    axs[1].set(ylabel='Joint-space L2 length (rad)',xticks=x,xticklabels=labels,title='Successful plans only');axs[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(plotdir/'phase2b_performance.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4))
    for name,key in [('Raw RRT','raw_path'),('Shortcut','shortcut_path')]:
        checks=evidence['independent_replay_audit']['raw' if name=='Raw RRT' else 'shortcut']
        qpath=evidence[key];dist=[0.];gap=[];base=0.
        xs=[]
        for edge,a,b in zip(checks,qpath,qpath[1:]):
            length=float(np.linalg.norm(np.array(b)-a))
            for sample in edge['samples']:
                xs.append(base+sample['alpha']*length);gap.append(sample['minimum_clearance']*1000)
            base+=length
        ax.plot(xs,gap,label=name)
    ax.axhline(5,color='red',ls='--',label='5 mm configured margin')
    ax.set(xlabel='Joint path arc length (rad)',ylabel='Sampled minimum gap (mm)',title='B sphere / seed 42 / independent .005 rad audit')
    ax.legend();fig.tight_layout();fig.savefig(plotdir/'phase2b_path_clearance.png',dpi=160);plt.close(fig)
    collision_fraction=np.median([float(r['collision_check_time_s'])/float(r['planning_time_s']) for r in successful])
    improvement=np.median([float(r['length_reduction_fraction'])*100 for r in successful])
    first=rejected['result']['search_fine_discrepancy']['first_invalid']
    pairs='；'.join(p['geom1']+' ↔ '+p['geom2']+f" ({p['distance_m']*1000:.6f} mm)" for p in first['offending_geom_pairs'])
    table=[]
    for s in summary['scene_summaries']:
        def fmt(k):return '—' if s[k] is None else f'{s[k]:.4f}'
        table.append(f"| {s['scene']} | {s['successes']}/{s['trials']} | {'有效' if s['direct_fine_valid'] else '无效'} | {fmt('successful_planning_time_median_s')} / {fmt('successful_planning_time_p95_s')} | {fmt('raw_length_median_rad')} → {fmt('shortcut_length_median_rad')} | {s['average_state_calls']:.1f} / {s['average_edge_calls']:.1f} |")
    protected_lines='\n'.join(f'- `{p}`：一致' for p in check)
    text=f'''# Panda Phase 2B 验收报告

验收日期：2026-10-21。项目：`D:\\mujoco_learning\\panda_safe_residual_rl`。解释器：`D:\\miniconda\\envs\\panda_safe_rl\\python.exe`。

本轮实际完成自行实现的七自由度双向 RRT-Connect、碰撞检查 Shortcut、独立细化复核、批量实验和 Windows 原生 MuJoCo 运动学可视化。没有修改 Phase 1.5 控制器，没有执行动力学轨迹，没有训练 PPO/SAC，也没有修改 Reacher 工程。

## 1. 验收结论及真实运行清单

- Pilot：8 个场景 × 3 种子 = 24 次。有效端点 A～E 共 15 次，14 次成功、1 次平滑后细化拒绝；非法端点 F 共 9 次，全部正确拒绝。
- 正式 batch：8 个场景 × 10 种子 = 80 次。有效端点 A～E 共 50 次，49 次成功（98%），1 次平滑后细化拒绝。F 类 30 次全部正确拒绝；不能把这 30 次预期拒绝混入有效问题的成功率分母。
- 四个直接边无效的场景 B～E 均已实际找到并独立细化验证绕行路径；39/40 次成功，其中 B 为 9/10。
- 自动测试：89 passed、0 failed、0 skipped，包括原有 71 项回归和新增 18 项。
- 原生窗口加录像：B_sphere，固定列表首个种子 42。{video['decoded_frames']} 帧、{video['duration_s']} s、{video['resolution'][0]}×{video['resolution'][1]}、{video['file_bytes']} 字节、{video['raw_unique_frames']} 个不同原始 RGB 帧，完整解码且正常退出。
- 单独保存原始路径细化拒绝、迭代预算、时间预算失败证据；没有删除失败或只保留成功种子。

Pilot 路径：`{pilot}`。

正式 batch 路径：`{run}`。

录像及逐帧数据：`{demo}`。

额外失败/分辨率证据：`{regression}`。

## 2. 配置、碰撞集合与算法

使用原 Phase 2A 检查器：23 个机器人 collision geoms、274 个审计配对，其中 183 自碰撞、91 环境配对。仍检查地面、球、圆柱、盒体及规定自碰撞；排除规则见文档 07。夹爪固定 0.04 m，5 mm 安全间隙不变。规划器不直接操作执行中的 Live MjData。

真实 MJCF 七关节限位采样，不使用手动 home±0.35 rad 包络。最近邻按量程归一化，扩展长度最大 0.35 rad（七维 L2），路径长度同单位。搜索/Shortcut 插值为最大单关节 0.02 rad，原始和最终路径分别由新建独立模型/数据以 0.005 rad 完整复核。失败不输出可执行路径。

统一预算：800 次外层迭代、6 s 搜索、12 s 总预算、CONNECT 最多 128 次、Shortcut 最多 40 次/2 s、goal bias 0.1。完整检查不能中途抢占，墙钟预算可能多出当前一次检查耗时。每次 plan 重置随机种子；Shortcut 使用独立种子流。

Pilot 种子：42、123、2026。正式种子：42、123、2026、7、11、19、31、73、101、314。配置在 pilot 前确定，批量实验沿用相同配置。

实际平台：{summary['platform']}，Python {summary['python']}，MuJoCo 3.16.0。所有正式场景使用同一个模型、相同碰撞配对与间隙，仅障碍位置变化。耗时为本机普通桌面负载下实测，batch 开始阶段与一次 pytest 回归短暂并行，不是隔离 CPU 基准；不外推其他机器或硬实时性能。

## 3. 各场景结果

耗时仅统计成功路径，含搜索、Shortcut 和两次细化；长度为成功路径中位数，调用数为该场景所有试验的平均。state 调用包含 edge 内部采样。

| 场景 | 成功/次数 | 直接 .005 边 | 耗时中位数 / P95 (s) | 原始→平滑长度中位数 (rad) | 平均 state / edge 调用 |
|---|---:|---|---:|---:|---:|
{chr(10).join(table)}

A 清空近处障碍；B 球；C 圆柱；D 球+圆柱；E 盒体。A～E 的起点、终点全部有效。F_invalid_start：起点外部碰撞；F_invalid_goal：目标外部碰撞；F_goal_limit：目标关节超限。F 各 10 次都以对应原因拒绝，未开始树搜索。

49 条成功路径全部通过原始及平滑后的独立 0.005 rad 复核；不是只复核展示种子。50 次有效端点试验的最终复核通过率为 49/50。B～E 的直接插值经细化确认无效，已保存首次违规位置和碰撞对象。

所有成功路径的 Shortcut 长度改善比例中位数为 {improvement:.2f}%。碰撞检查耗时占完整规划耗时的比例中位数为 {collision_fraction*100:.2f}%；本版使用全审计配对、多次端点及细化检查，几何距离查询是主要开销。没有关闭检查来追求速度。

![规划耗时与路径长度](../assets/plots/phase2b_performance.png)

A 无障碍直接边检查耗时 {representative[0]['direct_fine_time_s']:.4f} s，而 RRT 完整流程中位数 {representative[0]['successful_planning_time_median_s']:.4f} s。RRT 额外建立两树、采样、验证候选、重建、平滑并复核；对于已知可直接连接的简单问题，这些额外成本没有必要。此处仍保留真正双树搜索以验收算法。

## 4. 实际失败及窄擦碰回归

正式 B_sphere / seed=123：搜索及原始路径细化通过，但 Shortcut 的 0.02 rad 候选边通过后，0.005 rad 复核发现违规。本次整体失败，`path=[]`，未回退伪装成功。首次违规边 {first.get('edge_index')}，alpha={first.get('alpha')}，q={first['q']}。

违规几何体及真实间隙：{pairs}。失败 JSON：`{failed[0]['record_path']}`。

另外新增 5 mm 门槛窄间隙反例：小球半径 3 mm，由真实几何查询二分定位至约 4.99 mm 间隙，0.02 rad 粗网格漏检、0.005 rad 拒绝，并保存 0.01、0.0025 rad 对照。原 Phase 2A 零间隙窄擦碰案例、失败记录和测试均未修改。

首次新增测试执行为 17 passed / 1 failed：重复性测试原先错误要求每次都成功；实际同一 seed 两次均得到相同的细化失败。保留 `results/phase2b_initial_tests.log` 和 XML，并保存两次完整诊断至 `results/phase2b_repro_diagnostic.json`。修正测试为核对路径、树、计数和成功/失败判定一致后，全套 89 项通过。没有为通过测试放松安全门槛或删除真实失败。

## 5. 真实绕行视频与数值证据

展示选择预先列出的 B_sphere / seed=42。起终点有效；直接边 .005 无效，记录首次间隙违规及实际几何相交；原始 RRT 路径 {selected['raw_node_count']} 节点、长度 {selected['raw_path_length']:.6f} rad；Shortcut 后 {selected['smoothed_node_count']} 节点、长度 {selected['path_length']:.6f} rad；最终最小采样间隙 {selected['minimum_sampled_clearance']*1000:.6f} mm，两条路径均经独立细化。

![B场景路径采样间隙](../assets/plots/phase2b_path_clearance.png)

视频：`assets/videos/phase2b_rrt_planning_demo.mp4`。SHA256：`{video['mp4_sha256']}`。

核验：原生 MuJoCo 窗口已实际打开并观察到规划标签、机械臂、障碍和路径变化；完整播放后正常关闭。MP4 {video['decoded_frames']}/{video['expected_frames']} 帧匹配，图像变化与七关节位置变化记录一致，最大相对首帧关节 L2 位移 {video['max_displayed_joint_motion_rad']:.6f} rad。每帧 q、有效性、间隙、碰撞对象、RGB 哈希保存在逐帧 JSON；没有用静态图片替代视频。

窗口和录像始终标注 `KINEMATIC PLANNING VISUALIZATION` 与 `NOT DYNAMIC SERVO EXECUTION`。所有展示使用独立查询对象、仅 mj_forward，physics_steps=0，查询仿真时间保持 0。它是规划路径可视化，不是伺服动力学轨迹。

## 6. 已实测 PowerShell 命令

```powershell
Set-Location D:\\mujoco_learning\\panda_safe_residual_rl
& D:\\miniconda\\envs\\panda_safe_rl\\python.exe scripts/demo_rrt_connect.py --viewer --record --scene B_sphere --seed 42 --fps 10
```

不录制时去掉 --record。双障碍可用 `--scene D_two_obstacles --seed 42`；该路径已有 batch 数值复核，默认窗口流程相同。Esc、Q、关闭窗口或一遍播放结束均退出。每次录像写入独立运行目录，成功完整播放后更新 assets/videos 同名成品。

复现实验与测试：

```powershell
& D:\\miniconda\\envs\\panda_safe_rl\\python.exe scripts/run_phase2b_experiments.py --stage pilot
& D:\\miniconda\\envs\\panda_safe_rl\\python.exe scripts/run_phase2b_experiments.py --stage batch
& D:\\miniconda\\envs\\panda_safe_rl\\python.exe -m pytest -q
& D:\\miniconda\\envs\\panda_safe_rl\\python.exe scripts/phase2b_regression_evidence.py
& D:\\miniconda\\envs\\panda_safe_rl\\python.exe scripts/summarize_phase2b.py
```

## 7. 交付与保留内容

核心：src/planning/rrt_connect.py；配置：configs/rrt_connect.yaml、configs/phase2b_scenes.yaml；算法原理：docs/09_RRTConnect算法设计与原理.md；测试：tests/test_rrt_connect.py。

要求的结果均已生成：results/phase2b_planning_results.csv、results/phase2b_path_validation.csv、results/phase2b_performance.json、assets/videos/phase2b_rrt_planning_demo.mp4。完整逐试验 JSON、两条路径 CSV、父节点/扩展事件、失败 q 与碰撞配对见 batch 原始目录。CSV 和汇总 JSON 不代替原始证据。

以下历史文件 SHA256 与阶段开始快照逐一一致：

{protected_lines}

补充历史审计差异：旧 `results/phase2a_acceptance.json` 登记的 Phase 2A 碰撞视频哈希为 `0f95f6b4aa399b84f270e5f55916f35e953be7b921b8b6fd24380111f077565d`，当前成品为 `1391debfe2b0feb77d57c76ce256f2ef9b95669f65f4176a72db58ec150ea3c4`。当前文件修改时间 16:10:58，早于本轮开始快照 16:21:27；对应既有运行 `phase2a_collision_demo_1792570235842638700`。上午旧哈希版本仍保存在 `phase2a_collision_demo_1792552519491151300` 运行目录。两版本均保留，本轮未覆盖它们，也没有为消除差异改写旧验收报告。完整比较结果见 `results/phase2b_final_file_check.json`。

## 8. 限制及后续接口

离散 .005 rad 复核不能提供严格连续安全保证，窄擦碰漏检风险仍存在。实验仅代表保存的有限场景和十个种子；98% 是这 50 次有效端点问题的观测比例，不是普遍成功概率。RRT 非最优、随机且可能超时；没有证明一般不可解问题的不可解性。全配对查询与细化开销较大，本轮没有实现快速旁路、复杂最近邻结构或严格连续碰撞检测。

Phase 2C 可接收通过验收的 N×7 `result.path`、场景快照、真实关节顺序、配置/代码哈希和复核信息。当前路径未时间参数化，不保证速度、加速度、力矩、跟踪误差、动态障碍或接触执行安全。轨迹伺服执行、PPO/SAC 和残差强化学习均未实现/未执行。本阶段完成后暂停，等待用户审核。
'''
    (ROOT/'docs/10_Phase2B验收报告.md').write_text(text,encoding='utf-8')
    write_json(ROOT/'results/phase2b_acceptance.json',dict(phase='2B',date_local='2026-10-21',
       pytest=dict(passed=len(testcases),failed=0,skipped=0),pilot_run=pilot,batch_run=run,native_demo_run=demo,
       regression_run=regression,representative_trials=50,representative_successes=49,expected_invalid_rejections=30,
       fine_rejections=1,previous_stage_hashes_unchanged=check,native_video=video,
       dynamic_execution=False,reinforcement_learning_executed=False,continuous_safety_guarantee=False,
       audited_after_runs_sha256={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'src/planning/rrt_connect.py',ROOT/'tests/test_rrt_connect.py',
       ROOT/'configs/phase2b_scenes.yaml',ROOT/'configs/rrt_connect.yaml',ROOT/'scripts/demo_rrt_connect.py',
       ROOT/'scripts/phase2b_visualization.py',ROOT/'docs/09_RRTConnect算法设计与原理.md',ROOT/'docs/10_Phase2B验收报告.md',
       ROOT/'results/phase2b_planning_results.csv',ROOT/'results/phase2b_path_validation.csv',ROOT/'results/phase2b_performance.json']}))
    print('Report generated; protected hashes unchanged; 89 tests; 49/50 valid-endpoint plans.',flush=True)


if __name__=='__main__':main_guard(main)
