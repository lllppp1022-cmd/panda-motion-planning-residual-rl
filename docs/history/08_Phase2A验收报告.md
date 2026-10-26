# Panda Phase 2A 验收报告：关节构型与离散路径碰撞检测

日期：2026-10-21（Asia/Shanghai）。工程 `D:\mujoco_learning\panda_safe_residual_rl`；解释器 `D:\miniconda\envs\panda_safe_rl\python.exe`。本次仅完成碰撞检查，未实现/运行 RRT-Connect，未进行 PPO/SAC/残差训练。

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

默认间隙 5 mm，distmax=100 mm，判定额外容差 0.1 μm。大于等于 distmax 的记录是距离**下界**，不记为精确 100 mm；测试用 distmax=6 mm 的 home 返回 lower_bound=True。未截断查询仍受凸包近似和 native CCD 数值精度限制。native CCD 参数实际为 tolerance=1e-06、iterations=35；未使用 legacy CCD。

实际负距离与原生 `data.contact` 交叉核对，但 native contact 列表无法表达 5 mm 正间隙，也可能按静态/父子过滤不生成约束，因此不以 ncon=0 判定路径安全。

## 3. 可解释正例与负例

| 案例 | 实际判定 | 最小查询间隙 mm（可为穿透） |
|---|---|---|
| A_clear_home | 有效 | 23.0818 |
| B_sphere_collision | external_collision | -35.9829 |
| C_cylinder_collision | external_collision | -48.1171 |
| B2_box_collision | external_collision | -75.5359 |
| D_clearance_boundary | 有效 | 5.5000 |
| F_joint_limit | joint_limit | 未查询 |
| H_floor_collision | external_collision | -13.0537 |
| G_nonadjacent_self_collision | self_collision | -14.5379 |
| G_allowed_adjacent_home | 有效 | 23.0818 |
| E_interior_collision | external_collision | -64.0000 |

另加 D2：同一 5.5 mm 间隙在 6 mm 阈值下被拒绝，external_collision=False、clearance_violation=True，避免把“接近但没撞”误叫穿透。

球、柱、盒负例均放置在实际 link4 collision geom 的世界中心，不改变机器人几何。D 用 45 次实测表面距离二分构造 5.5 mm 间隙，不使用中心距离猜测。自碰撞和地面负例使用 seed=20261021 在硬限位内采样；查了 11 个真实构型便找到所需案例，不是训练。完整 q、场景位姿、查询列表及原生 contact 在 `results/runs/phase2a_validation_1792552512460507700/collision_cases_full.json`。

非相邻自碰撞实际 q：`[-1.5078883874147178, -1.0003800216470333, -0.7468592171137103, -1.8273062372528468, -0.5297352484490072, 0.14600280364760831, 2.669910398844009]`。link5/hand 的最小穿透约 14.54 mm。home 中 link3/link4 的装配几何有约 33.77 mm 重叠，按邻接规则排除；同一 home 的非相邻几何最小仍为 23.08 mm。没有通过关闭整个自碰撞来让它通过。

F 超限会在 FK 前拒绝；视频中该帧明确显示 HOME REFERENCE ONLY，不把上一构型或参考姿态冒充超限候选的实际求解结果。

## 4. edge 插值与分辨率实验

段数 `max(1, ceil(max(abs(q_end-q_start))/step_rad))`，逐段线性插值。起终点精确取输入值，检查每个样本；即使前面失效也遍历全部样本，确保起终点字段、采样最小值及耗时有完整含义。零长度边也检查两端。预算超限、分辨率错误和畸形输入保守拒绝，不悄悄减采样。

默认 0.02 rad，独立复核 0.005 rad。E 路径起终点均有效，joint1 从 -0.5 到 +0.5 rad，其余保持 home：0.02 rad 共 51 个样本，首次间隙不足 alpha=0.38（index 19）；首次零/负间隙样本 alpha=0.4（index 20）。这两个位置概念不同，视频分别标注 FIRST INVALID GAP 与 FIRST CONTACT。

| 案例 | 最大单关节步长 rad | valid | state 调用数 | 首次无效 alpha | 完整 edge 耗时 ms |
|---|---|---|---|---|---|
| clear_edge | 0.2 | True | 2 | None | 1.943 |
| clear_edge | 0.05 | True | 5 | None | 3.539 |
| clear_edge | 0.02 | True | 11 | None | 7.609 |
| clear_edge | 0.01 | True | 21 | None | 20.243 |
| clear_edge | 0.005 | True | 41 | None | 32.127 |
| endpoints_clear_interior_collision | 0.2 | False | 6 | 0.4 | 5.254 |
| endpoints_clear_interior_collision | 0.05 | False | 21 | 0.4 | 15.510 |
| endpoints_clear_interior_collision | 0.02 | False | 51 | 0.38 | 45.612 |
| endpoints_clear_interior_collision | 0.01 | False | 101 | 0.38 | 74.874 |
| endpoints_clear_interior_collision | 0.005 | False | 201 | 0.38 | 144.482 |
| invalid_start | 0.02 | False | 26 | 0.0 | 17.649 |
| invalid_end | 0.02 | False | 26 | 0.76 | 17.253 |
| grazing_zero_margin | 0.05 | True | 5 | None | 4.480 |
| grazing_zero_margin | 0.02 | True | 11 | None | 8.888 |
| grazing_zero_margin | 0.01 | True | 21 | None | 14.523 |
| grazing_zero_margin | 0.005 | False | 41 | 0.525 | 26.233 |
| grazing_zero_margin | 0.0025 | False | 81 | 0.5125 | 53.319 |
| grazing_zero_margin | 0.00125 | False | 161 | 0.5125 | 110.138 |
| same_grazing_default_5mm_margin | 0.02 | False | 11 | 0.3 | 8.234 |
| same_grazing_default_5mm_margin | 0.005 | False | 41 | 0.25 | 28.919 |

**实测漏检反例：**保持原机器人碰撞几何，仅使用半径 3 mm 的独立小球敏感性夹具、零安全间隙。通过真实距离二分，使 joint1=0.005 rad 附近产生约 10 μm 穿透。0.05/0.02/0.01 rad 网格漏检；0.005/0.0025/0.00125 rad 网格检出。完整构造尝试、球位置与所有采样点保留于 grazing_fixture.json 和 edge_validation_full.json。并未把这条粗网格通过的边宣称为可执行安全路径。

同一个擦碰场景改用默认 5 mm 间隙后，0.02 和 0.005 rad 均拒绝，说明正间隙能增加保守性；**这不是任意场景都不会漏检的证明**。细网格也可能漏掉更窄区间，当前 API 没有连续碰撞、扫掠体、区间距离界或自适应误差保证。CCD 在这里指 MuJoCo 的凸碰撞距离算法，不代表机器人整条边已做连续时间碰撞检测。

![实测分辨率反例](../results/phase2a_resolution_sensitivity.png)

## 5. 1000 次查询性能

12th Gen Intel(R) Core(TM) i5-12400F，Windows-10-10.0.26200-SP0，Python 3.11.17，MuJoCo 3.16.0。无 Viewer、无渲染；固定相同 seed 与 1000 个硬限位内输入，各场景先预热 20 次。新增障碍放远处，保留相同自碰撞、地面和所有机器人几何；没有为提速跳过必查配对，碰撞状态也查询完整表。

| 参与碰撞 geom 数（含地面） | 必查配对数 | 模型构造/编译 s | 1000 查询总 s | median ms | p95 ms | p99 ms |
|---|---|---|---|---|---|---|
| 27 | 274 | 0.0546 | 0.7340 | 0.6701 | 1.0524 | 1.2661 |
| 30 | 343 | 0.0686 | 0.8741 | 0.7964 | 1.2279 | 1.6012 |
| 36 | 481 | 0.0709 | 1.1318 | 1.0416 | 1.6292 | 1.9711 |

这里的 geom 数不含 visual：机器人固定23个，环境分别4/7/13个。模型构造耗时包括本进程已导入 MuJoCo 后的 XML 载入与编译，不是操作系统/解释器/DLL 全冷启动。查询中位数/分位数只计 state_valid；总耗时还包含逐条原始结果记录的少量 Python 开销。性能波动和最大耗时保留在 JSON，不挑最好一次。

基线分解：状态恢复+forward 共 0.0499 s；距离循环（含 Python 结果构造）共 0.4967 s，其余是模型/输入检查、指纹、min/contact 汇总和采样记录。这是分段墙钟计时，不是逐个 C++ 算法精确 profiler。原始1000输入 CSV及 SHA256、3000条逐查询计时均在原始目录。

![查询性能](../results/phase2a_performance.png)

## 6. 自动验收与独立性

实际运行 `python -m pytest tests -q`：**71 passed、0 failed、0 skipped**，其中原 Phase1/1.5共38项，本阶段33项（含参数化）。覆盖合法/超限、NaN/Inf/维度、三障碍与地面、阈值/截断、自碰撞过滤、有效边/无效起终点/中途碰撞、分辨率漏检、Live状态隔离、重复查询、夹爪恢复、模型/场景异常、关闭碰撞mask拒绝以及规划范围。

JUnit：`results/phase2a_final_tests.xml`；各次实际测试日志均保留，没有删除失败记录或伪造100%安全。开发辅助脚本曾有一次导入错误，已修复并记录 development_notes.json；本轮实际 pytest 没有失败项。早期2 μm擦碰原型和开发数据保留，正式表使用10 μm夹具。

Live 隔离测试先对真实执行 data 积分一步，快照全部 mjSTATE_INTEGRATION（含时间、关节、速度、控制等），从其 model 构造 checker，查询状态和边后逐值比较完全不变，model 外观/mask也不变。重复查询会重置夹爪、速度、warmstart与场景，不依赖前一次候选。已有 Phase1/1.5基准的 9 个源码/配置/原始数据和模型哈希核验不变，详见 prior_stage_integrity.json。未修改 teleop 安全包络、奖励、模型权重或旧 Reacher 环境。

## 7. 实际可视化与启动命令

两个演示均在 Windows 原生 Viewer 完整运行、正常关闭；颜色绿色有效、黄色近阈值、红色无效，窗口文字和视频标题标注 KINEMATIC CHECK / NOT dynamic execution，并显示首个不合格配对名称。没有动力学积分，帧之间可能直接切换构型，这属于查询可视化。

- 构型视频 `assets/videos/phase2a_collision_demo.mp4`：150 帧、15.0 s、[960, 752]、248427 bytes，30 个不同原始画面；native_viewer_launched=True，physics_steps=0。
- 路径视频 `assets/videos/phase2a_edge_demo.mp4`：69 帧、6.9 s，停留突出首次间隙失效与首次接触。
- 完整视频/逐帧 q、requested_q/displayed_q、配对、首末构型截图及解码验证分别在 `results/runs/phase2a_collision_demo_1792552519491151300`、`results/runs/phase2a_edge_demo_1792552419164423000`。MP4 解码帧数与记录一致；所有 query_data_time=0，明确没有把运动学切换包装成动力学执行。

```powershell
. 'D:\miniconda\shell\condabin\conda-hook.ps1'
conda activate panda_safe_rl
cd D:\mujoco_learning\panda_safe_residual_rl
python scripts/demo_collision_checker.py --viewer --record
python scripts/demo_edge_validation.py --viewer --record
python scripts/demo_edge_validation.py --viewer --step-rad 0.005
python -m pytest tests -q
```

前两条 viewer+record 命令及 pytest 已实际执行。0.005 rad 的数值检查与测试已执行；末尾这一条更细**Viewer命令未单独运行**，不是新的演示实测结论。无图形批量检查和性能实验可运行 `python scripts/run_phase2a_validation.py`；它只构造/检查测试夹具与计时，不进行规划或训练，每次保留独立原始目录。

## 8. 交付与未实现事项

必需产物：本报告、过滤规则文档、`results/phase2a_collision_cases.csv`、`results/phase2a_edge_validation.csv`、`results/phase2a_performance.json`、`assets/videos/phase2a_collision_demo.mp4` 均已生成。总验收和源码/视频哈希见 `results/phase2a_acceptance.json`。最终数值原始目录 `results/runs/phase2a_validation_1792552512460507700`；后续重跑可更新便捷副本，原始目录不覆盖。

尚未实现：RRT-Connect/任何完整规划器、连续扫掠安全证明、动态障碍、速度/加速度/扭矩可执行性、动力学轨迹跟踪、姿态约束、抓持允许接触集合、残差强化学习、硬件验证。凸包可能比真实非凸结构保守；固定夹爪与当前模型的排除规则不应不经审计用于另一机器人。静态间隙约束不保证执行误差、惯性、扰动下安全。

**Phase 2A 完成后暂停，等待用户审核。不自动进入 Phase 2B/RRT、轨迹执行或强化学习。**
