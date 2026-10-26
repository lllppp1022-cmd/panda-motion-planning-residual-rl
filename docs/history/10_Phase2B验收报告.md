# Panda Phase 2B 验收报告

验收日期：2026-10-21。项目：`D:\mujoco_learning\panda_safe_residual_rl`。解释器：`D:\miniconda\envs\panda_safe_rl\python.exe`。

本轮实际完成自行实现的七自由度双向 RRT-Connect、碰撞检查 Shortcut、独立细化复核、批量实验和 Windows 原生 MuJoCo 运动学可视化。没有修改 Phase 1.5 控制器，没有执行动力学轨迹，没有训练 PPO/SAC，也没有修改 Reacher 工程。

## 1. 验收结论及真实运行清单

- Pilot：8 个场景 × 3 种子 = 24 次。有效端点 A～E 共 15 次，14 次成功、1 次平滑后细化拒绝；非法端点 F 共 9 次，全部正确拒绝。
- 正式 batch：8 个场景 × 10 种子 = 80 次。有效端点 A～E 共 50 次，49 次成功（98%），1 次平滑后细化拒绝。F 类 30 次全部正确拒绝；不能把这 30 次预期拒绝混入有效问题的成功率分母。
- 四个直接边无效的场景 B～E 均已实际找到并独立细化验证绕行路径；39/40 次成功，其中 B 为 9/10。
- 自动测试：89 passed、0 failed、0 skipped，包括原有 71 项回归和新增 18 项。
- 原生窗口加录像：B_sphere，固定列表首个种子 42。356 帧、35.6 s、960×768、910932 字节、257 个不同原始 RGB 帧，完整解码且正常退出。
- 单独保存原始路径细化拒绝、迭代预算、时间预算失败证据；没有删除失败或只保留成功种子。

Pilot 路径：`D:\mujoco_learning\panda_safe_residual_rl\results\runs\phase2b_pilot_1792571046083953000`。

正式 batch 路径：`D:\mujoco_learning\panda_safe_residual_rl\results\runs\phase2b_batch_1792571440539901700`。

录像及逐帧数据：`D:\mujoco_learning\panda_safe_residual_rl\results\runs\phase2b_rrt_demo_1792571556789686400`。

额外失败/分辨率证据：`D:\mujoco_learning\panda_safe_residual_rl\results\runs\phase2b_regression_evidence_1792571537523801200`。

## 2. 配置、碰撞集合与算法

使用原 Phase 2A 检查器：23 个机器人 collision geoms、274 个审计配对，其中 183 自碰撞、91 环境配对。仍检查地面、球、圆柱、盒体及规定自碰撞；排除规则见文档 07。夹爪固定 0.04 m，5 mm 安全间隙不变。规划器不直接操作执行中的 Live MjData。

真实 MJCF 七关节限位采样，不使用手动 home±0.35 rad 包络。最近邻按量程归一化，扩展长度最大 0.35 rad（七维 L2），路径长度同单位。搜索/Shortcut 插值为最大单关节 0.02 rad，原始和最终路径分别由新建独立模型/数据以 0.005 rad 完整复核。失败不输出可执行路径。

统一预算：800 次外层迭代、6 s 搜索、12 s 总预算、CONNECT 最多 128 次、Shortcut 最多 40 次/2 s、goal bias 0.1。完整检查不能中途抢占，墙钟预算可能多出当前一次检查耗时。每次 plan 重置随机种子；Shortcut 使用独立种子流。

Pilot 种子：42、123、2026。正式种子：42、123、2026、7、11、19、31、73、101、314。配置在 pilot 前确定，批量实验沿用相同配置。

实际平台：Windows-10-10.0.26200-SP0，Python 3.11.17，MuJoCo 3.16.0。所有正式场景使用同一个模型、相同碰撞配对与间隙，仅障碍位置变化。耗时为本机普通桌面负载下实测，batch 开始阶段与一次 pytest 回归短暂并行，不是隔离 CPU 基准；不外推其他机器或硬实时性能。

## 3. 各场景结果

耗时仅统计成功路径，含搜索、Shortcut 和两次细化；长度为成功路径中位数，调用数为该场景所有试验的平均。state 调用包含 edge 内部采样。

| 场景 | 成功/次数 | 直接 .005 边 | 耗时中位数 / P95 (s) | 原始→平滑长度中位数 (rad) | 平均 state / edge 调用 |
|---|---:|---|---:|---:|---:|
| A_clear | 10/10 | 有效 | 0.6192 / 0.8066 | 1.6593 → 1.3000 | 948.1 / 21.1 |
| B_sphere | 9/10 | 无效 | 1.7536 / 1.8833 | 2.6118 → 1.8323 | 2203.5 / 61.3 |
| C_cylinder | 10/10 | 无效 | 2.4235 / 2.9142 | 3.5419 → 2.5147 | 2960.1 / 98.6 |
| D_two_obstacles | 10/10 | 无效 | 2.1356 / 2.8243 | 4.0932 → 2.3825 | 2995.4 / 97.2 |
| E_box_detour | 10/10 | 无效 | 1.9458 / 2.7349 | 3.3974 → 2.1436 | 2748.5 / 85.8 |
| F_invalid_start | 0/10 | 无效 | — / — | — → — | 1.0 / 0.0 |
| F_invalid_goal | 0/10 | 无效 | — / — | — → — | 2.0 / 0.0 |
| F_goal_limit | 0/10 | 无效 | — / — | — → — | 2.0 / 0.0 |

A 清空近处障碍；B 球；C 圆柱；D 球+圆柱；E 盒体。A～E 的起点、终点全部有效。F_invalid_start：起点外部碰撞；F_invalid_goal：目标外部碰撞；F_goal_limit：目标关节超限。F 各 10 次都以对应原因拒绝，未开始树搜索。

49 条成功路径全部通过原始及平滑后的独立 0.005 rad 复核；不是只复核展示种子。50 次有效端点试验的最终复核通过率为 49/50。B～E 的直接插值经细化确认无效，已保存首次违规位置和碰撞对象。

所有成功路径的 Shortcut 长度改善比例中位数为 31.17%。碰撞检查耗时占完整规划耗时的比例中位数为 96.90%；本版使用全审计配对、多次端点及细化检查，几何距离查询是主要开销。没有关闭检查来追求速度。

![规划耗时与路径长度](../assets/plots/phase2b_performance.png)

A 无障碍直接边检查耗时 0.1782 s，而 RRT 完整流程中位数 0.6192 s。RRT 额外建立两树、采样、验证候选、重建、平滑并复核；对于已知可直接连接的简单问题，这些额外成本没有必要。此处仍保留真正双树搜索以验收算法。

## 4. 实际失败及窄擦碰回归

正式 B_sphere / seed=123：搜索及原始路径细化通过，但 Shortcut 的 0.02 rad 候选边通过后，0.005 rad 复核发现违规。本次整体失败，`path=[]`，未回退伪装成功。首次违规边 1，alpha=0.22916666666666666，q=[-0.09020820387831385, -0.05484745928125327, -0.08662880537847967, -1.5300278552867304, 0.07279715268002324, 1.7003608358166489, -0.5208542923216353]。

违规几何体及真实间隙：panda_left_finger_fingertip_pad_collision_3_5 ↔ obstacle_sphere (4.937014 mm)。失败 JSON：`D:\mujoco_learning\panda_safe_residual_rl\results\runs\phase2b_batch_1792571440539901700\B_sphere_seed123.json`。

另外新增 5 mm 门槛窄间隙反例：小球半径 3 mm，由真实几何查询二分定位至约 4.99 mm 间隙，0.02 rad 粗网格漏检、0.005 rad 拒绝，并保存 0.01、0.0025 rad 对照。原 Phase 2A 零间隙窄擦碰案例、失败记录和测试均未修改。

首次新增测试执行为 17 passed / 1 failed：重复性测试原先错误要求每次都成功；实际同一 seed 两次均得到相同的细化失败。保留 `results/phase2b_initial_tests.log` 和 XML，并保存两次完整诊断至 `results/phase2b_repro_diagnostic.json`。修正测试为核对路径、树、计数和成功/失败判定一致后，全套 89 项通过。没有为通过测试放松安全门槛或删除真实失败。

## 5. 真实绕行视频与数值证据

展示选择预先列出的 B_sphere / seed=42。起终点有效；直接边 .005 无效，记录首次间隙违规及实际几何相交；原始 RRT 路径 13 节点、长度 4.009221 rad；Shortcut 后 3 节点、长度 2.108644 rad；最终最小采样间隙 22.806634 mm，两条路径均经独立细化。

![B场景路径采样间隙](../assets/plots/phase2b_path_clearance.png)

视频：`assets/videos/phase2b_rrt_planning_demo.mp4`。SHA256：`15384ceb63be8af11a2cd6cba6bde20680fff14a5662abcea035450b49c5b952`。

核验：原生 MuJoCo 窗口已实际打开并观察到规划标签、机械臂、障碍和路径变化；完整播放后正常关闭。MP4 356/356 帧匹配，图像变化与七关节位置变化记录一致，最大相对首帧关节 L2 位移 1.300000 rad。每帧 q、有效性、间隙、碰撞对象、RGB 哈希保存在逐帧 JSON；没有用静态图片替代视频。

窗口和录像始终标注 `KINEMATIC PLANNING VISUALIZATION` 与 `NOT DYNAMIC SERVO EXECUTION`。所有展示使用独立查询对象、仅 mj_forward，physics_steps=0，查询仿真时间保持 0。它是规划路径可视化，不是伺服动力学轨迹。

## 6. 已实测 PowerShell 命令

```powershell
Set-Location D:\mujoco_learning\panda_safe_residual_rl
& D:\miniconda\envs\panda_safe_rl\python.exe scripts/demo_rrt_connect.py --viewer --record --scene B_sphere --seed 42 --fps 10
```

不录制时去掉 --record。双障碍可用 `--scene D_two_obstacles --seed 42`；该路径已有 batch 数值复核，默认窗口流程相同。Esc、Q、关闭窗口或一遍播放结束均退出。每次录像写入独立运行目录，成功完整播放后更新 assets/videos 同名成品。

复现实验与测试：

```powershell
& D:\miniconda\envs\panda_safe_rl\python.exe scripts/run_phase2b_experiments.py --stage pilot
& D:\miniconda\envs\panda_safe_rl\python.exe scripts/run_phase2b_experiments.py --stage batch
& D:\miniconda\envs\panda_safe_rl\python.exe -m pytest -q
& D:\miniconda\envs\panda_safe_rl\python.exe scripts/phase2b_regression_evidence.py
& D:\miniconda\envs\panda_safe_rl\python.exe scripts/summarize_phase2b.py
```

## 7. 交付与保留内容

核心：src/planning/rrt_connect.py；配置：configs/rrt_connect.yaml、configs/phase2b_scenes.yaml；算法原理：docs/09_RRTConnect算法设计与原理.md；测试：tests/test_rrt_connect.py。

要求的结果均已生成：results/phase2b_planning_results.csv、results/phase2b_path_validation.csv、results/phase2b_performance.json、assets/videos/phase2b_rrt_planning_demo.mp4。完整逐试验 JSON、两条路径 CSV、父节点/扩展事件、失败 q 与碰撞配对见 batch 原始目录。CSV 和汇总 JSON 不代替原始证据。

以下历史文件 SHA256 与阶段开始快照逐一一致：

- `src/planning/collision_checker.py`：一致
- `configs/planning_collision.yaml`：一致
- `src/robot/teleop.py`：一致
- `scripts/teleop_panda.py`：一致
- `configs/teleop.yaml`：一致
- `src/robot/joint_controller.py`：一致
- `models/derived/panda_phase1.xml`：一致
- `models/derived/phase2a_scene.xml`：一致
- `results/phase2a_acceptance.json`：一致
- `results/phase2a_latest.json`：一致
- `scripts/phase2a_resolution.py`：一致
- `tests/test_phase2a_collision.py`：一致

补充历史审计差异：旧 `results/phase2a_acceptance.json` 登记的 Phase 2A 碰撞视频哈希为 `0f95f6b4aa399b84f270e5f55916f35e953be7b921b8b6fd24380111f077565d`，当前成品为 `1391debfe2b0feb77d57c76ce256f2ef9b95669f65f4176a72db58ec150ea3c4`。当前文件修改时间 16:10:58，早于本轮开始快照 16:21:27；对应既有运行 `phase2a_collision_demo_1792570235842638700`。上午旧哈希版本仍保存在 `phase2a_collision_demo_1792552519491151300` 运行目录。两版本均保留，本轮未覆盖它们，也没有为消除差异改写旧验收报告。完整比较结果见 `results/phase2b_final_file_check.json`。

## 8. 限制及后续接口

离散 .005 rad 复核不能提供严格连续安全保证，窄擦碰漏检风险仍存在。实验仅代表保存的有限场景和十个种子；98% 是这 50 次有效端点问题的观测比例，不是普遍成功概率。RRT 非最优、随机且可能超时；没有证明一般不可解问题的不可解性。全配对查询与细化开销较大，本轮没有实现快速旁路、复杂最近邻结构或严格连续碰撞检测。

Phase 2C 可接收通过验收的 N×7 `result.path`、场景快照、真实关节顺序、配置/代码哈希和复核信息。当前路径未时间参数化，不保证速度、加速度、力矩、跟踪误差、动态障碍或接触执行安全。轨迹伺服执行、PPO/SAC 和残差强化学习均未实现/未执行。本阶段完成后暂停，等待用户审核。
