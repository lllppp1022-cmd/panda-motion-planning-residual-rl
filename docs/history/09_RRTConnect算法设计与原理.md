# Panda Phase 2B：RRT-Connect 算法设计与原理

本阶段实现七关节空间双向 RRT-Connect、经过碰撞检查的 Shortcut 和独立细化复核。输出是关节折线路径，不包含时间参数化、速度/加速度约束或动力学执行。算法来自 Kuffner 与 LaValle 的 [RRT-Connect 原始论文（ICRA 2000）](https://www.kuffner.org/james/papers/kuffner_icra2000.pdf)，本项目自行实现树搜索，没有调用外部规划器。

## 1. 接口与状态

```python
from src.planning.collision_checker import CollisionChecker
from src.planning.rrt_connect import RRTConnectPlanner, PlannerConfig

checker = CollisionChecker()  # 独立复制模型及 MjData，不使用执行器的 live data
# 按需要 checker.set_obstacle_pose(...)；也可用 scripts.phase2b_common 加载保存场景。
planner = RRTConnectPlanner(checker, PlannerConfig.from_file(), seed=42)
result = planner.plan(q_start, q_goal)
if result.success and result.final_validation_passed:
    joint_path = result.path  # N×7，顺序始终从起点到终点，单位 rad
else:
    print(result.failure_reason, result.search_fine_discrepancy)
```

每个 q 必须是一维七元素有限数值。真实采样范围来自 `checker.robot.limits` 中的 MJCF 硬限位；如果配置了更严格的 planning_bounds 则采用其交集结果。Phase 1.5 的 home±0.35 rad 手动练习范围未被用于规划。夹爪由 Phase 2A 查询器固定在 0.04 m，并在每次查询时明确初始化。

`PlanResult` 包含 success、path、raw_path、smoothed_path、failure_reason、planning_time、search_time、smoothing_time、validation_time、iterations、两树节点数、state_valid_calls、explicit_state_valid_calls、edge_valid_calls、geom_distance_calls、collision_check_time、minimum_sampled_clearance、minimum_is_lower_bound、两种路径长度和节点数、两种细化复核、随机种子与完整配置。还保存父节点、扩展事件、Shortcut 接受/拒绝事件、首次违规构型和碰撞几何体。

失败时 `path=[]`，禁止把诊断用 raw_path/smoothed_path 当成通过复核的轨迹。失败记录中的长度 0 和最小间隙 null 表示没有交付有效最终路径；可从细化诊断中读取拒绝路径的真实间隙。若校验在预算内被中断，部分未完成的阶段时间字段可能保持 0；总耗时及已经发生的碰撞调用计数仍保存。

## 2. 双向树如何扩展

1. 使用 `state_valid` 检查两个根节点。任一根无效立即返回明确原因。
2. 建立以 start 和 goal 为根的两棵树，每个节点存储 q 和父节点索引。
3. 以 0.1 概率采样另一棵树的根，其余时间在实际七关节范围内均匀采样。
4. 在当前树中寻找最近节点。默认距离为 `sqrt(sum(((q-q_sample)/joint_range)**2))`，仅用于最近邻选择。
5. EXTEND 沿采样方向扩展至多 0.35 rad（七维欧氏范数），先检查新节点，再以 `edge_valid(..., step_rad=0.02)` 检查整条扩展边。
6. 对方树主动 CONNECT，重复同样的已检查扩展，最多 128 次，直到连接、受阻或达到预算。
7. 未连接时交换两棵树的工作引用，但每棵树的根角色 start/goal 不变。
8. 连接点必须完全一致。分别沿父节点追溯到根，根据根角色拼接为 start→goal，并删除连续完全相同的节点。

扩展长度与路径长度均采用未归一化的七维欧氏 rad；碰撞插值的 step_rad 则表示最大单关节相邻采样差（L∞）。归一化最近邻距离不会被错误地作为扩展 rad 长度。树节点均在硬限位内，插值是关节范围内线性插值，不对有硬限位的关节做角度环绕。

## 3. 严格复用 Phase 2A

所有节点调用原 `state_valid`；所有树边、连接边与 Shortcut 候选调用原 `edge_valid`。本版未新增快速碰撞旁路，也没有修改原检查器。

模型的 23 个机器人碰撞几何体、274 个审计配对（183 自碰撞、91 环境配对）、地面、球体、圆柱和盒体均保留；自碰撞规则见 `docs/07_自碰撞几何体与过滤规则.md`。5 mm 默认安全间隙、有限数值检查、实际关节限位及独立 MjData 均不变。相邻体的合理过滤由 Phase 2A 负责，不在规划器里临时删除配对。

距离查询的 distmax 截断仍按 Phase 2A 规则返回下界标记。最小采样间隙不能被解释为连续路径最小值；若 `minimum_is_lower_bound=True`，还不能把记录值解释为精确距离。

调用次数定义：state_valid_calls 包含显式节点查询，以及 edge_valid 内部每个采样点的查询；explicit_state_valid_calls 仅记录显式节点调用。edge_valid_calls 计整条边请求数。phases 中区分端点、搜索、粗路径审计、原始细化、Shortcut、平滑细化；geom_distance_calls 是底层几何配对距离调用总数。直接插值基线单独测量，不计入规划器调用数。

## 4. 两级复核与 Shortcut

流水线：**搜索 0.02 → 重建路径完整粗审计 → 新建模型/数据的原始路径 0.005 复核 → Shortcut 0.02 → 平滑路径完整 0.005 复核**。

每次完整路径审计检查方向、起终点、全部节点及所有相邻边。复核器从查询模型独立复制，使用同一固定障碍场景快照，不共享搜索 MjData。每条 edge 都含起点、终点及中间插值样本。

Shortcut 随机选择非相邻节点，仅在整条候选边通过检查且路径长度严格减少时替换中间子路径；最多 40 次尝试、2 s 时间预算。Shortcut 使用与搜索独立且可重复的随机数流。原始路径始终保留。

原始或平滑路径细化失败时，本次结果整体判失败；保留首次无效 q、所在边、插值 alpha、碰撞对象与搜索/细化差异。不会自动回退后仍把本次记录标为成功。本轮球障碍 seed=123 确实发生了这种拒绝，说明较粗网格可能漏掉窄间隙违规。

Phase 2A 原有窄擦碰反例及其失败记录保留。另外在 5 mm 门槛下新增 0.00499 m 窄间隙反例：0.02 rad 通过、0.005 rad 拒绝，作为规划器“粗网格通过但最终失败”的端到端回归。更细的 0.0025 rad 结果也保存。

**离散碰撞检查不等于严格连续碰撞安全保证。** 更细网格仍不能证明所有采样点之间绝无碰撞。

## 5. 有界预算与可重复性

配置文件 `configs/rrt_connect.yaml`：800 次外层迭代、6 s 搜索预算、12 s 总预算、每次 CONNECT 128 次扩展、Shortcut 40 次/2 s。时间使用单调时钟；在节点或整条边查询的前后检查预算，当前完整 Phase 2A edge 查询不可中途抢占，因此墙钟可能超出预算一个正在执行的查询耗时。不是硬实时系统。

相同种子每次 plan 都重置随机数流。预算未触发且输入不变时路径和计数应一致；不同机器负载下时间预算触发点可能不同。预算耗尽、非法输入、场景在规划中变化及复核失败均返回失败原因。不存在无限 CONNECT 或无终止预算循环。

## 6. 场景与复现实验

`configs/phase2b_scenes.yaml` 保存全部障碍参数、起终点、真实关节范围、三种子 pilot 与十种子正式列表。起终点由已知关节构型直接构造，避免把位置 IK、姿态和避障混为一个问题。障碍放在实际模型指尖几何体的 FK 位置，再以真实检查器核对端点及直接边；未验证绕行路径前不宣称场景已确认可解。

A 无近障碍；B 球体；C 圆柱；D 球+圆柱；E 盒体；F 三种非法端点。所有场景保留自碰撞和地面检查，远处停放的障碍也没有从配对集合删除。B～E 的已确认可解依据是实际保存并通过独立 0.005 rad 复核的路径。

```powershell
Set-Location D:\mujoco_learning\panda_safe_residual_rl
& D:\miniconda\envs\panda_safe_rl\python.exe scripts/run_phase2b_experiments.py --stage pilot
& D:\miniconda\envs\panda_safe_rl\python.exe scripts/run_phase2b_experiments.py --stage batch
& D:\miniconda\envs\panda_safe_rl\python.exe -m pytest -q
& D:\miniconda\envs\panda_safe_rl\python.exe scripts/phase2b_regression_evidence.py
```

每次运行创建独立 results/runs 子目录，保存全部成功和失败。latest 指针和汇总 CSV 指向最近完整运行，不删除旧记录。正式预算与种子列表在 pilot 前写入配置；没有选择性排除失败种子。

## 7. 可视化及 Phase 2C 接口边界

```powershell
Set-Location D:\mujoco_learning\panda_safe_residual_rl
& D:\miniconda\envs\panda_safe_rl\python.exe scripts/demo_rrt_connect.py --viewer --scene B_sphere --seed 42
& D:\miniconda\envs\panda_safe_rl\python.exe scripts/demo_rrt_connect.py --viewer --record --scene B_sphere --seed 42 --fps 10
& D:\miniconda\envs\panda_safe_rl\python.exe scripts/demo_rrt_connect.py --record --scene D_two_obstacles --seed 42
```

程序加载实际 batch 记录，再独立复核原始和最终路径。展示起点、目标、直接连接首次间隙违规与实际相交构型、原始绕行、Shortcut 路径。绿色有效、黄色接近间隙门槛、红色无效，窗口显示违规几何体。Esc、Q 或关闭窗口退出；默认一遍有界播放后自动退出。标题沿用查询模型的 Phase2A 内部名称，但窗口叠字和录像固定明确标注：

`KINEMATIC PLANNING VISUALIZATION`

`NOT DYNAMIC SERVO EXECUTION`

这里在独立 MjData 上写 qpos 后 mj_forward，仿真时间保持 0，从未调用 mj_step。录像是规划构型展示，不是实际伺服运动。原 Phase 1.5 键盘控制器完全保留。

后续 Phase 2C 可读取 `result.path` 或保存的 shortcut_path.csv，连同 scene/config/source hashes、复核分辨率、间隙、关节顺序和有效状态使用。当前路径没有时间戳、速度/加速度、平滑导数或动力学可跟踪性保证；不能直接作为已验收的控制器轨迹。本轮没有实现 Phase 2C，也没有训练强化学习。
