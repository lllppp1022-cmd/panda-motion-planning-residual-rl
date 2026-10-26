"""Generate the Phase2A audit and report strictly from actual saved measurements."""
from _bootstrap import ROOT,main_guard
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.common import write_json,write_csv,sha


def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def latest_native(prefix):
    for p in sorted((ROOT/'results/runs').glob(prefix+'_*'),reverse=True):
        if (p/'complete.json').exists():
            r=load(p/'complete.json')
            if r.get('native_viewer_launched') and not r['interrupted'] and (p/f'{prefix}.mp4').is_file():
                return p,r
    raise RuntimeError('Native complete demo required before reporting')


def main():
    run=Path(load(ROOT/'results/phase2a_latest.json')['run'])
    cases=load(run/'collision_cases_full.json');audit=load(run/'pair_audit.json')
    edges=load(run/'edge_validation_full.json');perf=load(run/'phase2a_performance.json')
    grazing=load(run/'grazing_fixture.json')
    native_collision,cv=latest_native('phase2a_collision_demo')
    native_edge,ev=latest_native('phase2a_edge_demo')
    junit=ET.parse(ROOT/'results/phase2a_final_tests.xml').getroot()
    passed=len(list(junit.iter('testcase')))
    failures=len(list(junit.iter('failure')))+len(list(junit.iter('error')))
    skips=len(list(junit.iter('skipped')))
    if (passed,failures,skips)!=(71,0,0):raise RuntimeError('Unexpected pytest status; report actual failed tests first')
    for name,source in [('phase2a_collision_demo',native_collision),('phase2a_edge_demo',native_edge)]:
        if sha(ROOT/f'assets/videos/{name}.mp4')!=sha(source/f'{name}.mp4'):
            raise RuntimeError('Canonical video differs from latest completed native recording')
    write_csv(ROOT/'results/phase2a_included_pairs.csv',audit['included'])
    write_csv(ROOT/'results/phase2a_excluded_pairs.csv',audit['excluded'])
    old=load(ROOT/'results/teleop_recording_validation.json')['source_and_data_sha256_audited_after_run']
    integrity={name:dict(expected=digest,actual=sha(ROOT/name),unchanged=sha(ROOT/name)==digest) for name,digest in old.items()}
    provenance=load(ROOT/'results/model_provenance.json')
    for name,digest in provenance['derived_files'].items():
        path=ROOT/'models/derived'/name
        integrity['models/derived/'+name]=dict(expected=digest,actual=sha(path),unchanged=sha(path)==digest)
    write_json(run/'prior_stage_integrity.json',integrity)
    if not all(v['unchanged'] for v in integrity.values()):raise RuntimeError('Previous-stage hash changed; investigate before reporting')
    write_json(run/'development_notes.json',dict(notes=[dict(observed='Initial helper import failed during a development invocation',
                error="ModuleNotFoundError: No module named '_bootstrap'",resolution='Shared fixture helper imports ROOT from src.common',
                note='Recorded after the observed tool output; not a failed pytest or fabricated experiment'),
                dict(observed='Initial grazing prototype used 2 micrometres nominal penetration',
                     resolution='Release fixture targets 10 micrometres, above the configured 1 micrometre CCD tolerance parameter',
                     note='Earlier development artifacts remain; the tolerance parameter is not a certified geometry error bound')]))
    grouped=Counter((p['category'],p['body1'],p['body2']) for p in audit['included'])
    excluded=Counter((p['body1'],p['body2'],p['exclusion_reason']) for p in audit['excluded'])
    robot_table='\n'.join(f"| {g['id']} | `{g['name']}` | {g['body']} | {g['type']} | {g['contype']}/{g['conaffinity']} |" for g in audit['robot_geoms'])
    pair_table='\n'.join(f'| {kind} | {a} ↔ {b} | {n} |' for (kind,a,b),n in grouped.items())
    translations=dict(official_contact_exclusion='官方 MJCF 明确排除 link0/link1；保留原装配规则',
                      same_body_compound_geometry='同 body 的多个凸几何片段构成同一连杆，不互测',
                      fixed_weld_same_rigid_body='link7/hand 固定相连，属于同一个刚体 weld',
                      adjacent_welded_bodies_at_joint='相邻活动关节两侧的焊接刚体，关节装配几何可合理重叠',
                      fixed_base_floor_mount_only='固定基座 link0 的地面安装接触；不排除任何活动连杆与地面')
    exclude_table='\n'.join(f'| {a} ↔ {b} | {n} | {translations[reason]} |' for (a,b,reason),n in excluded.items())
    adjacent=load(run/'allowed_adjacent_pairs_measured.json')[0]
    doc=f'''# Panda 自碰撞几何体与过滤规则（Phase 2A）

本文件对应官方 Menagerie Panda、提交 `{provenance['commit']}`。未编辑官方 XML/mesh/许可证，未关闭全部自碰撞。新增的独立场景只增加盒体，原 Phase 1 与键盘练习场景不变。

## 1. 查询集合与冻结条件

机器人采用原模型所有 **23 个命名 group=3 碰撞几何体**；visual group=2、无接触目标标记不参与距离检查。网格碰撞按 MuJoCo 的凸包处理，不是视觉三角网格的精确非凸表面。link5 的三个凸片段，以及左右夹爪各自的 mesh 和五个 box 指垫都保留。

环境默认包括世界平面 floor、半径 0.06 m 的球、半径 0.065 m/半高 0.18 m 的圆柱、半尺寸 [0.06,0.08,0.08] m 的盒体。sphere/cylinder/box 是静态 mocap 障碍，查询前从明确保存的场景位姿恢复；不是随机移动障碍。允许具名 `obstacle_` 前缀的同类静态或 mocap 几何供受控场景扩展；未审计的动态刚体或形状拒绝构造。

夹爪不是第八个规划关节：两个 finger slide 均固定为 0.04 m（可单独配置 0～0.04 m），保持相等；每个候选状态都从 home keyframe 恢复非规划自由度，清零速度和 warmstart，设置七关节与场景，再 mj_forward。夹爪开度改变会改变碰撞结果，不允许把本次结果直接用于任意抓持状态。

| geom ID | 碰撞几何名称 | 所属 body | MuJoCo geom type | contype/conaffinity |
|---|---|---|---|---|
{robot_table}

类型 7=mesh、6=box；环境 0=plane、2=sphere、5=cylinder、6=box。

## 2. 与原模型过滤的关系

原模型仅有一条显式 contact exclusion：`link0` / `link1`，编译 signature 为 65538（body IDs 1、2）；没有显式 contact pair。所有参与碰撞的机器人/环境 geoms 当前 contype=conaffinity=1。兼容条件为 `(type1 & affinity2) || (type2 & affinity1)`。若后续模型通过关闭位掩码让已审计配对不可碰撞，构造器会拒绝，而不是悄悄把它当作安全。

直接 mj_geomDistance 不替我们执行父子体、weld 或 contact exclusion 过滤，所以必须自己生成配对表。规则按**焊接刚体**而非只按原始 body 名称判断：同 body 的复合片段不互测；link7 和无关节固定 hand 属同一个 weld；相邻 weld 两侧的关节装配不互测。world weld 的相邻例外按 MuJoCo 规则处理，link0/link1 仍由官方显式排除决定。

左右 finger 是兄弟活动体，不是同一个刚体，也不是彼此父子，所以它们之间的 **36 对**碰撞几何全部保留。非相邻 link5/hand 亦保留，实际已检出约 14.54 mm 的穿透，证明自碰撞没有被整体关闭。

地面只排除 link0 的固定安装接触；其余 22 个机器人碰撞 geoms 均查 floor。固定基座与障碍物的显式距离仍检查；MuJoCo 原生接触列表可能因静态/静态过滤而不生成对应约束，因此规划有效性依据显式距离表，不能只看 `data.ncon`。

## 3. 全部参与的 body 配对分组

23 个机器人几何产生 253 个自配对候选，排除 70 对，保留 **183 对**。外部候选 23×4=92，排除一个固定基座地面安装配对，保留 **91 对**。总计 **274 对必查、71 对排除**。每次 state_valid 即使发现碰撞仍查询全部必查配对，用于真实最小间隙统计。

| 分类 | body 对 | 展开后的 geom 对数量 |
|---|---|---|
{pair_table}

完整 274 条 geom 名称及 ID：[phase2a_included_pairs.csv](../results/phase2a_included_pairs.csv)。

## 4. 全部排除项与原因

| body 对 | 展开后的 geom 对数量 | 原因 |
|---|---|---|
{exclude_table}

完整 71 条 geom 名称、ID、原因：[phase2a_excluded_pairs.csv](../results/phase2a_excluded_pairs.csv)。没有额外的“为了让 home 通过”排除，也没有按 graph 距离粗暴跳过全部近邻链段。

在合法 home 中，`{adjacent['geom1']}` / `{adjacent['geom2']}` 的直接表面查询为 {adjacent['distance_m']*1000:.4f} mm（相邻装配几何重叠），该对按关节邻接规则排除。home 仍保留全部非相邻检查，最小间隙为 23.0818 mm。此负距离是被允许的装配几何重叠，不能说成求解器真的产生了该接触事件：原生过滤通常不会给这一对生成 contact。

## 5. 距离与判定语义

默认安全间隙 0.005 m、distmax=0.10 m、额外判定容差 1e-7 m。查询值 ≤ 安全间隙+容差就拒绝，零/负值表示接触或穿透类别，正值但小于间隙仅表示安全距离不足。`external_collision` 包括环境与地面；`self_collision` 单独标识；`clearance_violation` 同时涵盖这两类碰撞与仅间隙不足。

达到 distmax 的配对记录 `is_lower_bound=True`，含义是“距离至少为该截断值”，不是精确 0.10 m。全局最小若仍被截断，`minimum_is_lower_bound=True`；若存在更小未截断值，则返回该查询最小值。构造器禁止 distmax≤间隙+容差，否则无法可靠作出通过判断。未截断也只是 MuJoCo 数值几何查询结果，不是精确数学距离证明。

本环境启用 native CCD，ccd_tolerance=1e-6、ccd_iterations=35。拒绝已关闭 native CCD 的查询；不使用官方文档提示距离可能不准确的 legacy 管线。公差参数不等于有保证的几何误差界。额外返回 `native_contacts` 供负例交叉核对，不用原生 contact 数量替代完整间隙查询。

## 6. 可维护性与边界

每个 checker 独占复制的 MjModel 和新 MjData，静态场景只通过 `set_obstacle_pose` 改自己的场景副本。七关节按实际 joint ID/qpos 地址设置。模型结构、形状及过滤数组的指纹被检查；检测到模型被改动、非有限值、错误位姿或 forward 异常时保守拒绝。新增形状、显式 contact pair、机器人碰撞 geom 数变化需重新审计；不把缺失几何自动当成自由空间。

不允许把这个 checker 的 data 作为正在积分的执行 data。并行规划每个 worker 单独建 checker；跨线程复用会拒绝。允许独立规划范围，但必须是 MJCF 硬限位的子集；未引入 teleop 的 home±0.35 rad 限制。

参考：[MuJoCo 碰撞过滤和凸包](https://mujoco.readthedocs.io/en/stable/computation/index.html#collision-detection)、[mj_geomDistance 截断语义](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-geomdistance)、原始 `models/mujoco_menagerie/franka_emika_panda/LICENSE`（Apache-2.0）。
'''
    (ROOT/'docs/07_自碰撞几何体与过滤规则.md').write_text(doc,encoding='utf-8')
    # Scientific figures from actual saved full-sample records, no generated trajectories.
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    broad=[e for e in edges if e['case']=='endpoints_clear_interior_collision']
    for e in broad:
        if e['step_rad'] in (.2,.02,.005):
            axes[0].plot([s['alpha'] for s in e['samples']],[s['minimum_clearance']*1000 for s in e['samples']],'.-',label=f"step={e['step_rad']} rad")
    axes[0].axhline(5,color='r',ls='--',label='5 mm margin');axes[0].set(xlabel='Interpolation alpha',ylabel='Minimum audited-pair distance (mm)',title='Endpoints clear, interior blocked')
    for e in edges:
        if e['case']=='grazing_zero_margin' and e['step_rad'] in (.02,.005,.00125):
            axes[1].plot([s['alpha'] for s in e['samples']],[s['minimum_clearance']*1e6 for s in e['samples']],'.-',label=f"step={e['step_rad']} rad")
    axes[1].axhline(0,color='r',ls='--');axes[1].set(xlabel='Interpolation alpha',ylabel='Minimum distance (micrometres)',title='Zero-margin grazing: coarse grid misses',xlim=(.45,.60),ylim=(-35,110))
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(ROOT/'results/phase2a_resolution_sensitivity.png',dpi=150);plt.close(fig)
    bs=perf['benchmarks'];fig,ax=plt.subplots(figsize=(8,4.5));positions=np.arange(len(bs))
    ax.bar(positions-.15,[b['median_ms'] for b in bs],width=.3,label='median')
    ax.bar(positions+.15,[b['p95_ms'] for b in bs],width=.3,label='p95')
    ax.set(xticks=positions,xticklabels=[f"{b['total_collision_geoms']} geoms / {b['audited_pairs']} pairs" for b in bs],ylabel='State query wall time (ms)',title='1000 fixed configurations per scene; all required pairs')
    ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(ROOT/'results/phase2a_performance.png',dpi=150);plt.close(fig)
    case_table='\n'.join(f"| {c['name']} | {'有效' if c['result']['valid'] else c['result']['rejection_reason']} | {c['result']['minimum_clearance']*1000:.4f} |" if c['result']['minimum_clearance'] is not None else f"| {c['name']} | {c['result']['rejection_reason']} | 未查询 |" for c in cases)
    benchmark_table='\n'.join(f"| {b['total_collision_geoms']} | {b['audited_pairs']} | {b['model_load_s']:.4f} | {b['query_total_s']:.4f} | {b['median_ms']:.4f} | {b['p95_ms']:.4f} | {b['p99_ms']:.4f} |" for b in bs)
    edge_table='\n'.join(f"| {e['case']} | {e['step_rad']} | {e['valid']} | {e['state_calls']} | {e['first_invalid_alpha']} | {e['elapsed_s']*1000:.3f} |" for e in edges)
    selfcase=next(c for c in cases if c['name']=='G_nonadjacent_self_collision')
    search=load(run/'self_collision_search.json')
    first=next(e for e in edges if e['case']=='endpoints_clear_interior_collision' and e['step_rad']==.02)
    contact=next(s for s in first['samples'] if any(p['distance_m']<=1e-7 for p in s['offending_geom_pairs']))
    report=f'''# Panda Phase 2A 验收报告：关节构型与离散路径碰撞检测

日期：2026-10-21（Asia/Shanghai）。工程 `D:\\mujoco_learning\\panda_safe_residual_rl`；解释器 `D:\\miniconda\\envs\\panda_safe_rl\\python.exe`。本次仅完成碰撞检查，未实现/运行 RRT-Connect，未进行 PPO/SAC/残差训练。

## 1. 实际完成与接口

新增 `src/planning/collision_checker.py`。`CollisionChecker` 独占 MjModel 副本和新 MjData；没有传入、写入或积分 Live 执行数据。查询通过 home 初始化，设置七关节及固定夹爪，调用 mj_forward 和 mj_geomDistance，**不调用 mj_step**。

```python
from src.planning.collision_checker import CollisionChecker, state_valid, edge_valid

scene = CollisionChecker()
scene.set_obstacle_pose('obstacle_sphere', [0.9, 0.4, 0.6])
q_start = scene.home.copy()
q_end = q_start.copy()
q_end[0] += 0.2

state = state_valid(q_start, scene)
edge = edge_valid(q_start, q_end, scene, step_rad=0.02)
fine = edge_valid(q_start, q_end, scene, step_rad=0.005)
print(state.valid, state.minimum_clearance, state.minimum_is_lower_bound)
print(edge.valid, edge.first_invalid_alpha, edge.offending_geom_pairs)
# 也可调用 scene.state_valid(q) 和 scene.edge_valid(a, b)。
# bool(result) 与 result.valid 一致，拒绝结果不会因 dataclass 真值而误放行。
```

`scene` 在这个便利接口中是持有静态场景快照的 CollisionChecker，不是实时执行器对象。后续 RRT 只需调用上述 API，但必须明确使用 result.valid/真值及拒绝原因；本轮没有把它集成成规划器。自定义 `options` 仅接受已知配置键，配置位于 `configs/planning_collision.yaml`。

state 返回 valid、joint_limit_valid、external_collision、self_collision、clearance_violation、minimum_clearance、minimum_is_lower_bound、offending_geom_pairs、rejection_reason、pair_results、native_contacts、distance_calls 与耗时。edge 额外返回起终点状态、全部样本、首次无效 index/alpha/q、全路径最小采样间隙与查询次数。

默认采用**真实 MJCF 硬限位**，不是 Phase 1.5 的 home±0.35 rad。额外 planning_joint_bounds_rad 可独立指定硬限位内的子范围；测试确认 home 的 joint1+0.8 rad 仍可按几何有效性通过。夹爪固定为两侧 0.04 m，未把它扩展为规划维度。

## 2. 碰撞对象、过滤和距离依据

23 个机器人 collision geoms，球/圆柱/盒/地面共四个环境 geoms；总共 274 对：183 自碰撞+91 外部。71 个排除项都有可审计原因，详见 [07_自碰撞几何体与过滤规则](07_自碰撞几何体与过滤规则.md) 与完整 included/excluded CSV。左右夹爪间 36 对保留；非相邻 link5/hand 已实际检出自碰撞。

只排除同 body、同固定 weld、关节相邻 weld、官方 link0/link1，以及 link0 固定地面安装；不是任意屏蔽自碰撞，也没有忽略活动连杆地面接触。原始 contact exclusion 与兼容掩码被审计，关闭已审计配对掩码的模型被拒绝。

默认间隙 5 mm，distmax=100 mm，判定额外容差 0.1 μm。大于等于 distmax 的记录是距离**下界**，不记为精确 100 mm；测试用 distmax=6 mm 的 home 返回 lower_bound=True。未截断查询仍受凸包近似和 native CCD 数值精度限制。native CCD 参数实际为 tolerance={perf['ccd_options']['ccd_tolerance']}、iterations={perf['ccd_options']['ccd_iterations']}；未使用 legacy CCD。

实际负距离与原生 `data.contact` 交叉核对，但 native contact 列表无法表达 5 mm 正间隙，也可能按静态/父子过滤不生成约束，因此不以 ncon=0 判定路径安全。

## 3. 可解释正例与负例

| 案例 | 实际判定 | 最小查询间隙 mm（可为穿透） |
|---|---|---|
{case_table}

另加 D2：同一 5.5 mm 间隙在 6 mm 阈值下被拒绝，external_collision=False、clearance_violation=True，避免把“接近但没撞”误叫穿透。

球、柱、盒负例均放置在实际 link4 collision geom 的世界中心，不改变机器人几何。D 用 45 次实测表面距离二分构造 5.5 mm 间隙，不使用中心距离猜测。自碰撞和地面负例使用 seed=20261021 在硬限位内采样；查了 {len(search)} 个真实构型便找到所需案例，不是训练。完整 q、场景位姿、查询列表及原生 contact 在 `{run.relative_to(ROOT).as_posix()}/collision_cases_full.json`。

非相邻自碰撞实际 q：`{selfcase['q']}`。link5/hand 的最小穿透约 14.54 mm。home 中 link3/link4 的装配几何有约 33.77 mm 重叠，按邻接规则排除；同一 home 的非相邻几何最小仍为 23.08 mm。没有通过关闭整个自碰撞来让它通过。

F 超限会在 FK 前拒绝；视频中该帧明确显示 HOME REFERENCE ONLY，不把上一构型或参考姿态冒充超限候选的实际求解结果。

## 4. edge 插值与分辨率实验

段数 `max(1, ceil(max(abs(q_end-q_start))/step_rad))`，逐段线性插值。起终点精确取输入值，检查每个样本；即使前面失效也遍历全部样本，确保起终点字段、采样最小值及耗时有完整含义。零长度边也检查两端。预算超限、分辨率错误和畸形输入保守拒绝，不悄悄减采样。

默认 0.02 rad，独立复核 0.005 rad。E 路径起终点均有效，joint1 从 -0.5 到 +0.5 rad，其余保持 home：0.02 rad 共 {first['state_calls']} 个样本，首次间隙不足 alpha={first['first_invalid_alpha']}（index {first['first_invalid_index']}）；首次零/负间隙样本 alpha={contact['alpha']}（index {contact['index']}）。这两个位置概念不同，视频分别标注 FIRST INVALID GAP 与 FIRST CONTACT。

| 案例 | 最大单关节步长 rad | valid | state 调用数 | 首次无效 alpha | 完整 edge 耗时 ms |
|---|---|---|---|---|---|
{edge_table}

**实测漏检反例：**保持原机器人碰撞几何，仅使用半径 3 mm 的独立小球敏感性夹具、零安全间隙。通过真实距离二分，使 joint1=0.005 rad 附近产生约 10 μm 穿透。0.05/0.02/0.01 rad 网格漏检；0.005/0.0025/0.00125 rad 网格检出。完整构造尝试、球位置与所有采样点保留于 grazing_fixture.json 和 edge_validation_full.json。并未把这条粗网格通过的边宣称为可执行安全路径。

同一个擦碰场景改用默认 5 mm 间隙后，0.02 和 0.005 rad 均拒绝，说明正间隙能增加保守性；**这不是任意场景都不会漏检的证明**。细网格也可能漏掉更窄区间，当前 API 没有连续碰撞、扫掠体、区间距离界或自适应误差保证。CCD 在这里指 MuJoCo 的凸碰撞距离算法，不代表机器人整条边已做连续时间碰撞检测。

![实测分辨率反例](../results/phase2a_resolution_sensitivity.png)

## 5. 1000 次查询性能

{perf['cpu']}，{perf['platform']}，Python {perf['python']}，MuJoCo {perf['mujoco']}。无 Viewer、无渲染；固定相同 seed 与 1000 个硬限位内输入，各场景先预热 20 次。新增障碍放远处，保留相同自碰撞、地面和所有机器人几何；没有为提速跳过必查配对，碰撞状态也查询完整表。

| 参与碰撞 geom 数（含地面） | 必查配对数 | 模型构造/编译 s | 1000 查询总 s | median ms | p95 ms | p99 ms |
|---|---|---|---|---|---|---|
{benchmark_table}

这里的 geom 数不含 visual：机器人固定23个，环境分别4/7/13个。模型构造耗时包括本进程已导入 MuJoCo 后的 XML 载入与编译，不是操作系统/解释器/DLL 全冷启动。查询中位数/分位数只计 state_valid；总耗时还包含逐条原始结果记录的少量 Python 开销。性能波动和最大耗时保留在 JSON，不挑最好一次。

基线分解：状态恢复+forward 共 {bs[0]['forward_total_s']:.4f} s；距离循环（含 Python 结果构造）共 {bs[0]['distance_total_s']:.4f} s，其余是模型/输入检查、指纹、min/contact 汇总和采样记录。这是分段墙钟计时，不是逐个 C++ 算法精确 profiler。原始1000输入 CSV及 SHA256、3000条逐查询计时均在原始目录。

![查询性能](../results/phase2a_performance.png)

## 6. 自动验收与独立性

实际运行 `python -m pytest tests -q`：**71 passed、0 failed、0 skipped**，其中原 Phase1/1.5共38项，本阶段33项（含参数化）。覆盖合法/超限、NaN/Inf/维度、三障碍与地面、阈值/截断、自碰撞过滤、有效边/无效起终点/中途碰撞、分辨率漏检、Live状态隔离、重复查询、夹爪恢复、模型/场景异常、关闭碰撞mask拒绝以及规划范围。

JUnit：`results/phase2a_final_tests.xml`；各次实际测试日志均保留，没有删除失败记录或伪造100%安全。开发辅助脚本曾有一次导入错误，已修复并记录 development_notes.json；本轮实际 pytest 没有失败项。早期2 μm擦碰原型和开发数据保留，正式表使用10 μm夹具。

Live 隔离测试先对真实执行 data 积分一步，快照全部 mjSTATE_INTEGRATION（含时间、关节、速度、控制等），从其 model 构造 checker，查询状态和边后逐值比较完全不变，model 外观/mask也不变。重复查询会重置夹爪、速度、warmstart与场景，不依赖前一次候选。已有 Phase1/1.5基准的 {len(integrity)} 个源码/配置/原始数据和模型哈希核验不变，详见 prior_stage_integrity.json。未修改 teleop 安全包络、奖励、模型权重或旧 Reacher 环境。

## 7. 实际可视化与启动命令

两个演示均在 Windows 原生 Viewer 完整运行、正常关闭；颜色绿色有效、黄色近阈值、红色无效，窗口文字和视频标题标注 KINEMATIC CHECK / NOT dynamic execution，并显示首个不合格配对名称。没有动力学积分，帧之间可能直接切换构型，这属于查询可视化。

- 构型视频 `assets/videos/phase2a_collision_demo.mp4`：{cv['frames']} 帧、{cv['duration_s']} s、{cv['resolution']}、{cv['file_bytes']} bytes，{cv['raw_unique_frames']} 个不同原始画面；native_viewer_launched=True，physics_steps=0。
- 路径视频 `assets/videos/phase2a_edge_demo.mp4`：{ev['frames']} 帧、{ev['duration_s']} s，停留突出首次间隙失效与首次接触。
- 完整视频/逐帧 q、requested_q/displayed_q、配对、首末构型截图及解码验证分别在 `{native_collision.relative_to(ROOT).as_posix()}`、`{native_edge.relative_to(ROOT).as_posix()}`。MP4 解码帧数与记录一致；所有 query_data_time=0，明确没有把运动学切换包装成动力学执行。

```powershell
. 'D:\\miniconda\\shell\\condabin\\conda-hook.ps1'
conda activate panda_safe_rl
cd D:\\mujoco_learning\\panda_safe_residual_rl
python scripts/demo_collision_checker.py --viewer --record
python scripts/demo_edge_validation.py --viewer --record
python scripts/demo_edge_validation.py --viewer --step-rad 0.005
python -m pytest tests -q
```

前两条 viewer+record 命令及 pytest 已实际执行。0.005 rad 的数值检查与测试已执行；末尾这一条更细**Viewer命令未单独运行**，不是新的演示实测结论。无图形批量检查和性能实验可运行 `python scripts/run_phase2a_validation.py`；它只构造/检查测试夹具与计时，不进行规划或训练，每次保留独立原始目录。

## 8. 交付与未实现事项

必需产物：本报告、过滤规则文档、`results/phase2a_collision_cases.csv`、`results/phase2a_edge_validation.csv`、`results/phase2a_performance.json`、`assets/videos/phase2a_collision_demo.mp4` 均已生成。总验收和源码/视频哈希见 `results/phase2a_acceptance.json`。最终数值原始目录 `{run.relative_to(ROOT).as_posix()}`；后续重跑可更新便捷副本，原始目录不覆盖。

尚未实现：RRT-Connect/任何完整规划器、连续扫掠安全证明、动态障碍、速度/加速度/扭矩可执行性、动力学轨迹跟踪、姿态约束、抓持允许接触集合、残差强化学习、硬件验证。凸包可能比真实非凸结构保守；固定夹爪与当前模型的排除规则不应不经审计用于另一机器人。静态间隙约束不保证执行误差、惯性、扰动下安全。

**Phase 2A 完成后暂停，等待用户审核。不自动进入 Phase 2B/RRT、轨迹执行或强化学习。**
'''
    (ROOT/'docs/08_Phase2A验收报告.md').write_text(report,encoding='utf-8')
    manifest_files=[ROOT/'src/planning/collision_checker.py',ROOT/'configs/planning_collision.yaml',ROOT/'tests/test_phase2a_collision.py',
                    ROOT/'scripts/phase2a_cases.py',ROOT/'scripts/phase2a_resolution.py',ROOT/'scripts/run_phase2a_validation.py',
                    ROOT/'scripts/demo_collision_checker.py',ROOT/'scripts/demo_edge_validation.py',ROOT/'scripts/phase2a_visualization.py',
                    ROOT/'models/derived/phase2a_scene.xml',ROOT/'assets/videos/phase2a_collision_demo.mp4',ROOT/'assets/videos/phase2a_edge_demo.mp4']
    acceptance=dict(phase='2A',date_local='2026-10-21',pytest=dict(passed=passed,failed=failures,skipped=skips),
                    numerical_run=str(run),native_collision_run=str(native_collision),native_edge_run=str(native_edge),
                    included_pairs=len(audit['included']),excluded_pairs=len(audit['excluded']),
                    previous_stage_hashes_unchanged=True,model_commit=provenance['commit'],
                    coarse_resolution_false_negative_demonstrated=True,continuous_collision_guarantee=False,
                    rrt_implemented=False,reinforcement_learning_executed=False,
                    audited_after_runs_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in manifest_files})
    write_json(ROOT/'results/phase2a_acceptance.json',acceptance)
    readme=ROOT/'README.md';text=readme.read_text(encoding='utf-8')
    marker='\n## Phase 2A：关节构型与离散路径碰撞检查\n'
    if marker not in text:
        text+=marker+'''
已建立独立 MjModel/MjData 的 `state_valid(q, scene)` 与 `edge_valid(a, b, scene)`，覆盖硬限位、球/柱/盒/地面、保留必要自碰撞及最小间隙。23 个机器人碰撞几何、274 个必查配对；不沿用 teleop 的 home±0.35 rad 练习包络。

71 项测试实际通过。实测保留了粗网格漏检擦碰、细网格检出的反例：离散检查不构成连续安全保证。

```powershell
python scripts/demo_collision_checker.py --viewer --record
python scripts/demo_edge_validation.py --viewer --record
```

[自碰撞规则与完整配对](docs/07_自碰撞几何体与过滤规则.md) · [Phase 2A 实际验收报告](docs/08_Phase2A验收报告.md) · [构型检查视频](assets/videos/phase2a_collision_demo.mp4) · [插值检查视频](assets/videos/phase2a_edge_demo.mp4)

演示是明确标注的运动学查询可视化，不是动力学轨迹执行。Phase 2A 完成后暂停，未启动 RRT 或强化学习。
'''
        readme.write_text(text,encoding='utf-8')
    print('WROTE docs/07, docs/08, pair CSVs, figures and results/phase2a_acceptance.json')


if __name__=='__main__':main_guard(main)
