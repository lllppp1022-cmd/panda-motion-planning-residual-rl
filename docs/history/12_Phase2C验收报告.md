# Panda Phase 2C：真实 MuJoCo 动力学执行验收报告

日期：2026-10-21。项目：`D:\mujoco_learning\panda_safe_residual_rl`。Python：`D:\miniconda\envs\panda_safe_rl\python.exe`。MuJoCo 3.16.0，物理步长 0.002 s。

本阶段已完成冻结路径加载、五次时间参数化、原位置伺服真实积分、逐步实际碰撞监控、分级与速度对照实验、原生窗口、动力学录像和自动验收。**双障碍路径动态执行失败，未通过其安全执行验收**；不能将 Phase 2B 规划成功解释为本阶段执行成功。本报告保留该失败，不修改模型、间隙或历史成功率来制造通过。

## 1. 历史成果冻结

读取 Phase 2B 实际成功记录及 Shortcut CSV，选取 A_clear、B_sphere、C_cylinder、D_two_obstacles，种子均为 42。检查 success、final_validation_passed、N×7 有限性、CSV/JSON 一致、起终点、障碍位置/四元数/类型/尺寸、joint1～joint7 顺序，并在当前模型重新逐节点逐边作 0.005 rad 复核。

历史 derived XML 和过滤配置哈希与 Phase 2B 开始前快照匹配；80 个上游文件与 Phase 1 历史哈希一致。完整证据：`D:\mujoco_learning\panda_safe_residual_rl\results\runs\phase2c_freeze_1792572838675347100`。未重新规划、未虚构缺失路径。本轮结束再次检查冻结清单全部一致，原 Phase 2B CSV、报告 JSON、模型和原运动学视频保留。

## 2. 冻结配置、初始化及执行方法

`src/control/trajectory_generator.py` 对每个关节线段使用 s=10τ³−15τ⁴+6τ⁵，段端速度/加速度为零；以 1.875 与精确 10/√3 系数计算时长。两档参考最大速度 0.10、0.15 rad/s，参考最大加速度 0.30 rad/s²，最小段时长 0.10 s。不会越出线段；没有采用未经验证的过冲样条。

`src/control/trajectory_executor.py` 使用新的 live MjData，ctrl 是位置目标，不是力矩。保留原控制器、执行器增益/阻尼/力限、重力及原约束；指令变化限速 0.35 rad/s。500 Hz 控制、500 Hz 原检查器构型审计，加上每物理步在 live 数据上对相同 274 个配对的实际距离查询与原生接触记录，考虑真实夹爪微小偏移。没有通过关闭碰撞检查、减小 5 mm 门槛或移动障碍改善结果。

四路径起点 joint1=-0.65 rad，不是 home。因此只在 t=0、正式物理积分前明确初始化一次，日志注明不是 home 到起点的实际运动。之后 1.5 s 静态保持并检查最后至少 0.5 s 稳定；正式执行不再写 live qpos/qvel。初态不一致时拒绝运行。q_path、q_ref、实际 ctrl 和 q_actual 分开保存。

首次试验前冻结成功标准：全程无禁止接触或间隙违规；最终最大关节误差≤0.02 rad、TCP 误差≤10 mm、最大实际关节速度≤0.02 rad/s，并连续保持至少 0.5 s；终点最多等待 2 s。保护条件另含 0.10 rad 跟踪误差、1.0 rad/s 实际速度和有限执行预算。失败后停止参考推进，受限保持并继续记录约 0.5 s 惯性响应。

## 3. 实际运行清单与总体结论

- 分级 pilot：4 次，A/B/C 成功，D 因实际间隙不足失败。运行目录共 2 个，A 与 B/C/D 分开顺序运行。
- 正式实验：4 场景 × 2 档速度 × 2 次重复 = 16 次，12/16 稳定成功；A/B/C 共 12/12，D 共 0/4。重复为固定初态、固定路径、确定性物理的一致性检查，不是 16 个随机场景，更不能外推统计成功概率。
- 正式总积分 188030 步，累计 376.060 s 仿真；共 1004 个低间隙违规步（包括失败后的保持）；原生接触观察数 0，禁止接触数 0。
- 原生 B_sphere / 0.10 rad/s 录像单独实际成功；另实际点击 Viewer 关闭按钮验证中止归档。未将 GUI 运行、pytest 和故意碰撞的负例夹入正式完成率。
- 104 项 pytest 实际通过，包括原有 89 项全部回归、新增轨迹/真实执行/录像/实际窗口关闭验收。初次新增测试 13 项也通过。

正式原始目录：`D:\mujoco_learning\panda_safe_residual_rl\results\runs\phase2c_batch_1792573101827506800`。每次运行配置、初始化、tracking、clearance、events 和 summary 均独立保存；全部成功与失败均保留。

## 4. 正式结果表

时间含启动保持、实际运动和成功稳定/失败保护保持。每格汇总同速度的两次确定性重复。最终 TCP 误差始终相对真正目标；D 失败时仍远离目标，不用其局部小跟踪误差冒充到达。

| 场景 | vmax rad/s | 成功 | 规划→实际最小间隙 mm | 实际仿真时间 s | 最大关节跟踪误差 rad | TCP跟踪RMSE mm | 最终TCP误差 mm |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_clear | 0.10 | 2/2 | 23.082 → 23.078 | 26.376 | 0.009821 | 7.579 | 6.724 |
| A_clear | 0.15 | 2/2 | 23.082 → 23.078 | 18.250 | 0.014729 | 8.494 | 6.724 |
| B_sphere | 0.10 | 2/2 | 22.807 → 22.816 | 33.152 | 0.009839 | 7.376 | 6.724 |
| B_sphere | 0.15 | 2/2 | 22.807 → 22.816 | 22.768 | 0.014744 | 8.198 | 6.724 |
| C_cylinder | 0.10 | 2/2 | 21.682 → 21.684 | 36.188 | 0.013025 | 5.824 | 6.724 |
| C_cylinder | 0.15 | 2/2 | 21.682 → 21.684 | 24.792 | 0.016416 | 6.579 | 6.724 |
| D_two_obstacles | 0.10 | 0/2 | 5.756 → 4.671 | 15.482 | 0.009844 | 8.639 | 381.051 |
| D_two_obstacles | 0.15 | 0/2 | 5.756 → 4.605 | 11.022 | 0.014763 | 10.055 | 381.047 |

关节跟踪 RMSE（rad，轨迹及最终稳定阶段；D 为失败前已执行部分，不代表整条路径）：

| 场景 / 速度 | J1 | J2 | J3 | J4 | J5 | J6 | J7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| A_clear / 0.10 | 0.006197 | 0.006583 | 0.000004 | 0.006313 | 0.000334 | 0.001141 | 0.000000 |
| A_clear / 0.15 | 0.009249 | 0.006584 | 0.000008 | 0.006313 | 0.000334 | 0.001141 | 0.000000 |
| B_sphere / 0.10 | 0.005266 | 0.006051 | 0.002596 | 0.006256 | 0.002885 | 0.001793 | 0.004146 |
| B_sphere / 0.15 | 0.007867 | 0.006109 | 0.003873 | 0.006349 | 0.004282 | 0.002296 | 0.006191 |
| C_cylinder / 0.10 | 0.005544 | 0.005928 | 0.002959 | 0.006692 | 0.003136 | 0.002085 | 0.002783 |
| C_cylinder / 0.15 | 0.008280 | 0.007079 | 0.004329 | 0.007282 | 0.004667 | 0.002874 | 0.004153 |
| D_two_obstacles / 0.10 | 0.000850 | 0.007057 | 0.003419 | 0.007169 | 0.003259 | 0.003109 | 0.007043 |
| D_two_obstacles / 0.15 | 0.001273 | 0.007698 | 0.005211 | 0.007690 | 0.004697 | 0.004049 | 0.010544 |

![七关节参考与实际](../assets/plots/phase2c_joint_tracking.png)

所有场景目标构型相同，A/B/C 的最终约 6.724 mm TCP 静差也因此相近。这是保留原伺服和重力下的真实结果，未做瞬移、增益修改或后补静差抵消。更高参考速度缩短执行时间，同时改变跟踪误差；两档速度都不能保证 D 的实际间隙。

## 5. D 双障碍失败诊断

保存路径的规划最小采样间隙仅 5.756322 mm，距 5 mm 门槛只有约 0.756 mm 裕度。0.10 rad/s 第一次违规发生在 t=14.982000 s，轨迹时间 13.482000 s，右夹指 `panda_right_finger_collision_2` 对 `obstacle_cylinder` 实际间隙约 4.959684 mm。

该时刻参考构型本身有效，完整几何查询最小间隙 5.944584 mm；实际七关节构型固定夹爪查询为 4.959684 mm，与直接 live 查询几乎一致。逐关节实际减参考偏差为 [-0.00067084, 0.00525642, -0.00222749, -0.00665243, -0.00272421, -0.00289388, -0.0055806] rad。可见实际伺服偏差已耗尽很小的几何裕度，不能因参考路径有效而继续放行。

停止后继续记录 0.500000 s，最小实际间隙进一步到 4.670673 mm，证明保护保持不能瞬间消除运动。0.15 rad/s 同样失败；完整首次 q、碰撞配对、参考/实际差、停止后响应见 `results/phase2c_failure_diagnosis.json`。本轮不改障碍、不降阈值、不伪造执行完成；更大规划裕度或经独立验收的跟踪改进留待用户决定。

![动态间隙与失败后的保持](../assets/plots/phase2c_clearance.png)

低间隙违规步数不是独立碰撞次数。本轮正式实验均没有原生实际接触；故意把障碍移动进机器人的测试负例会触发接触/违规，只用于检查保护逻辑，不计入上述场景结果。

## 6. 速度、加速度与控制限速

实际速度峰值、全段有限差分加速度峰值及限速介入步数逐运行记录在 execution_summary 和正式 summary CSV。全段加速度约 8.941548 rad/s²，峰值主要来自预保持刚开始的重力瞬态；不能拿参考 0.30 rad/s²声称实际也受此硬限制。单独抽取运动阶段加速度至 `results/phase2c_acceleration_audit.csv`，不掩盖全段峰值。

成功运行未因正常参考运动触发 0.35 rad/s 指令限速；D 失败后的受限保持会发生限速介入，仍沿用同一控制器。每个物理步均记录实际 ctrl，能与 q_ref、q_actual 对照。

| 场景 / 参考速度 | 实际速度峰值 rad/s | 全段实际加速度峰值 rad/s² | 运动/稳定阶段加速度峰值 rad/s² | 两次合计指令限速步数 | 墙钟耗时中位数 s |
|---|---:|---:|---:|---:|---:|
| A_clear / 0.10 | 0.099988 | 8.941548 | 0.012629 | 0 | 19.857 |
| A_clear / 0.15 | 0.149958 | 8.941548 | 0.028406 | 0 | 14.279 |
| B_sphere / 0.10 | 0.099984 | 8.941548 | 0.032116 | 0 | 25.265 |
| B_sphere / 0.15 | 0.149946 | 8.941548 | 0.072095 | 0 | 16.565 |
| C_cylinder / 0.10 | 0.100721 | 8.941548 | 0.079162 | 0 | 29.038 |
| C_cylinder / 0.15 | 0.150844 | 8.941548 | 0.175900 | 0 | 18.871 |
| D_two_obstacles / 0.10 | 0.099977 | 8.941548 | 0.017209 | 18 | 11.221 |
| D_two_obstacles / 0.15 | 0.149919 | 8.941548 | 0.038692 | 22 | 8.253 |

墙钟耗时包含完整 500 Hz 检查及日志 I/O，期间有其他本地测试/GUI 进程运行；它不是隔离硬件性能基准，也不影响固定物理步长下的仿真时间。

## 7. 原生 Viewer 与 live 录像验收

单独录像路径：`D:\mujoco_learning\panda_safe_residual_rl\results\runs\phase2c_dynamic_demo_1792573114343141400`。窗口实际观察到 DYNAMIC SERVO EXECUTION、递增仿真时间、运动段、TCP 参考/实际和间隙。窗口内部模型名沿用原查询场景名称，明确的动力学标签及数据来源覆盖了这一历史名称差异。

视频 `assets/videos/phase2c_dynamic_execution.mp4`：333 帧全部解码，33.300 s、960×768、927050 字节，333 个不同原始 RGB 帧；最大关节 L2 位移 1.300032 rad。所有帧直接来自正在积分的同一个 live MjData，时间单调，各帧保存 q_actual/q_ref/ctrl 与 physics_step，未通过 scratch 切换 qpos 生成执行录像。SHA256：`c36cd8b80f817929e42b81220d553b94938f890cd91a31ec6f83c5736fb3d49c`。

本视频 10 fps 按仿真时间采样，播放时长与真实仿真时间接近但存在首尾帧取整；墙钟运行速度不是视频时间尺度。图中原模型绿色球为旧场景标记，本次目标和参考以明确的 TCP 数值为准。

实际窗口关闭测试归档：`D:\mujoco_learning\panda_safe_residual_rl\results\runs\phase2c_dynamic_demo_1792573202340571200`；中止原因 viewer_closed，窗口正常销毁，真实积分和保护保持日志全部保存，没有覆盖成功录像。显示/录制异常与动力学日志独立处理。

## 8. 自动测试与文件交付

104 passed / 0 failed / 0 skipped；日志 `results/phase2c_full_tests.log`，JUnit `results/phase2c_full_tests.xml`。测试覆盖端点/段端静止、线段内参考、速度加速度、真实 A/B 执行、实际 mj_step 次数及仿真时间、未复核/失败路径拒绝、初态不一致拒绝、实际 live 障碍侵入触发保护并保留惯性段、关闭意图和实际窗口关闭归档、原生动态视频完整解码、原有全部 89 项回归。

主要交付：

- src/control/trajectory_generator.py、trajectory_executor.py、validated_path.py。
- scripts/demo_rrt_execution.py、freeze_phase2c_inputs.py、run_phase2c_experiments.py、summarize_phase2c.py。
- results/phase2c_execution_summary.csv、phase2c_tracking.csv、phase2c_clearance.csv。
- results/phase2c_failure_diagnosis.json、phase2c_acceleration_audit.csv、phase2c_performance.json。
- assets/videos/phase2c_dynamic_execution.mp4、assets/plots/phase2c_joint_tracking.png、phase2c_clearance.png。
- docs/11_轨迹时间参数化与动力学执行.md 及本报告。

最初 pilot 与修正后的 batch 均保留：中间仅把保护保持计时改为从首次违规时刻累计，以保证后续完整记录至少约 0.5 s，并加入原生窗口积分锁；没有改变成功标准、控制增益、规划路径或障碍参数。

## 9. 已实际运行的 PowerShell 命令

```powershell
Set-Location D:\mujoco_learning\panda_safe_residual_rl
& D:\miniconda\envs\panda_safe_rl\python.exe scripts/demo_rrt_execution.py --viewer --record --scene B_sphere --speed 0.10
```

只观看去掉 --record。A_clear 用于无障碍对照；C_cylinder 可查看圆柱执行。D_two_obstacles 会如实展示本轮已发现的动态间隙失败，不保证到达。Esc/Q/关闭窗口触发中止与归档。

```powershell
& D:\miniconda\envs\panda_safe_rl\python.exe scripts/run_phase2c_experiments.py --stage batch
& D:\miniconda\envs\panda_safe_rl\python.exe -m pytest -q
& D:\miniconda\envs\panda_safe_rl\python.exe scripts/summarize_phase2c.py
```

原 Phase 2B 对比命令：

```powershell
& D:\miniconda\envs\panda_safe_rl\python.exe scripts/demo_rrt_connect.py --viewer --scene B_sphere --seed 42
```

## 10. 限制、未完成事项与后续接口

本轮实现与验收工作已完成，但 D 场景安全执行未通过，不能作为可放心执行的案例。仅用一个保存的规划种子、四场景、两档保守速度和固定初态；不构成随机扰动鲁棒性或实机安全认证。没有严格连续碰撞保证、动态障碍规划、实时刹车距离证明、连续 jerk 约束、姿态控制、残差策略或 RL 训练。初始化构型是仿真前配置，不是已完成的物理 home→start 路径。

后续可使用 `QuinticTrajectory.sample(t)` 提供的 q_ref/dq_ref/ddq_ref、段号/进度，以及执行器实际 q/qvel、TCP、误差、ctrl、间隙与失败状态作为控制/残差学习接口。本轮没有自动进入 PPO/SAC。规划细化通过不等于实际执行绝对安全，仿真无碰撞也不等于实机安全认证。阶段到此暂停，等待用户审核。
