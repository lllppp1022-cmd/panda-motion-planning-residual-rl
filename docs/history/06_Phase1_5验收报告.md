# Panda Phase 1.5 实际验收报告

验收日期：2026-10-21（Asia/Shanghai）。工程：`D:\mujoco_learning\panda_safe_residual_rl`。本阶段只增加键盘控制、日志、真实录像及 Windows 退出处理；没有训练网络或实现 RRT。

## 实现与验收范围

- 新入口 `scripts/teleop_panda.py`，默认无参数即打开原生 MuJoCo Viewer。
- `src/robot/teleop.py` 复用原位置控制器、DLS IK；`configs/teleop.yaml` 管理步长与限速。
- callbacks 仅入队；命令处理线程与回调线程不同；IK 只修改 scratch 状态。
- 七关节键盘控制、XYZ 六方向、soft stop、home、限位/最小步长/非有限拒绝、接触及意外运动联锁、独立 CSV/JSON/视频输出已实现。
- 原 Phase 1 的 18 项测试加本阶段 20 项，**38 passed / 0 failed / 0 skipped**。JUnit：`results/phase1_5_tests.xml`。

## 原生窗口真实交互与录像

最终合格录制目录：`results/runs/teleop_panda_1792550069788228500`。

通过 computer-use 在原生 Viewer 中发送真实键盘事件。没有把直接调用 `TeleopController.command` 的单元测试当作 GUI 验收，也没有用预置 qpos 或动画替代运动。最终录像包含七次 K（joint1～7）、joint7 的 J、M、W/S/A/D/R/F、空格、H 和 × 关闭。

- 实际命令 25 条（含关闭转成的退出命令）。
- 物理积分 **29603 步 / 59.206 仿真秒**；步数乘 0.002 s 与仿真时钟一致。
- 正常 `viewer_closed` 退出，原生关闭请求 1，窗口已销毁，进程退出码 0。
- 最大 ctrl 单步变化 **0.0004000 rad**，符合 0.20 rad/s × 0.002 s。
- 最大实际关节单步变化 **0.00032671 rad**；最大实际角速度 **0.163357 rad/s**。这两项是观测值，不是未来轨迹保证。
- 外部接触最大 0，自接触最大 0，异常告警步数 0。
- 每条命令执行前后 live qpos 完全相同；命令只改目标，后续积分产生位移。
- 593 帧视频与同时间物理日志逐一匹配，最大关节差 **0.0 rad**；原始机器人 RGB 有 221 个不同帧，排除了仅时钟文字变化的假动画。
- GIF 解码 593 帧、59.3 s；MP4 解码 593 帧；模型关节位移范数最大 0.063027 rad。

展示文件：

- `assets/videos/panda_keyboard_teleop_20261021.mp4`
- `assets/videos/panda_keyboard_teleop_20261021.gif`
- 原目录 `teleop_joint_tracking.png`、`teleop_xyz_tracking.png`、首末 PNG 和全部 CSV/逐帧 JSON。

### 单次按键的实际位移

从命令前状态，到约 0.65 s 后真实日志，分别检查所选关节或对应世界轴。没有把理想 FK 位移当作实际位移。

| 键 | 控制对象 | 实测位移绝对值或沿要求方向的有符号位移 |
|---|---|---|
| K | joint1 | 0.019962 rad |
| K | joint2 | 0.020123 rad |
| K | joint3 | 0.019957 rad |
| K | joint4 | 0.019981 rad |
| K | joint5 | 0.019936 rad |
| K | joint6 | 0.019933 rad |
| K | joint7 | 0.019952 rad |
| J | joint7 | 0.019950 rad |
| W | 世界X | 4.8984 mm，方向正确 |
| S | 世界X | 4.8997 mm，方向正确 |
| A | 世界Y | 4.9874 mm，方向正确 |
| D | 世界Y | 4.9837 mm，方向正确 |
| R | 世界Z | 5.0206 mm，方向正确 |
| F | 世界Z | 5.0198 mm，方向正确 |

末端 5 mm 指令的对应轴实际位移约 4.90～5.02 mm。实际定位误差不等于零：本次回 home 后三维误差 6.7245 mm，与 Phase 1 的原伺服重力静差一致。没有为改善数字修改控制增益、奖励或阈值。

## 安全与失败路径证据

1. 每个目标先经真实 MJCF 七关节限位，再经 home±0.35 rad 练习包络。过小剩余步长拒绝；手动目标超前量与单次 IK 关节变化限制为 0.12 rad。
2. 原生窗口限幅试验的真实日志在 `results/runs/teleop_panda_1792549639781183200/interaction_commands.csv`：约 t=89.534 s 命中包络边缘，t=90.004/90.578/91.532 s 的 K 被拒绝；随后 H、空格、H 均接受。该开发运行之后关闭窗口崩溃，因此不把它列为完整成功运行；其已落盘操作是边界检查的部分证据。限幅与从边界回 home 也由最终 38 项测试中的实动力学测试复核。
3. IK 不可达、失败、非有限解和过大关节变化在隔离单元测试中注入/求解，验证目标、XYZ 和 live qpos 保持不变。没有为了演示失败，给最终 GUI 添加危险“跳到不可达点”的快捷键。**最终合格 GUI 录像中未发生 IK 失败；失败拒绝路径的证据来自自动测试。**
4. 接触联锁测试在独立测试环境移动障碍球到实际机器人碰撞几何，检测真实接触后拒绝移动；没有在人工练习中故意撞击机器人。外部/自接触仅是当前碰撞模型生成的实际接触，不是完整碰撞风险预测。
5. Windows 中文输入法会拦截字母键，已实际观察并验证切英文后的按键。Viewer 数字键显示组冲突以及 W/R/F 等渲染快捷键冲突均已处理，最终窗口保持正常机器人显示。
6. 开发中两次 native × 关闭触发 access violation，日志保留于 `results/teleop_close_test_20261021.log` 和相应不完整运行。根据实测栈推断为 Viewer 销毁锁的竞态；未修改 MuJoCo。加入本进程 WM_CLOSE 排队后，`teleop_panda_1792549993249892400` 与最终录制均实际点击 × 正常退出。默认无参数启动另在 `teleop_panda_1792550259629003800` 通过 Esc 正常退出，进程均为 0。
7. 一次开发运行收到 Viewer 原生重置造成状态修改，检测后以 error 正常清理并非合格控制运行；错误与配置均保留。不要使用 Backspace、滑块或 Ctrl 拖动操作这套 teleop。此限制详见教程。

## 所有开发/验收运行（未隐藏失败）

原始目录不覆盖、不删除。首个隔离桌面运行未获得原生桌面键盘交互；早期运行用于冲突排查和输入法排查，不替代最终合格录像。`wall_timeout` 是预先设置的墙钟超时，不能等同于用户关闭验收。没有 complete.json 的运行不填写伪造步数。

| 运行 | 实际退出/完整性状态 | 完整保存的步数 | 仿真秒 |
|---|---|---|---|
| teleop_panda_1792548955402479000 | wall_timeout | 140962 | 281.92400000061804 |
| teleop_panda_1792548999420367600 | keyboard_exit | 45338 | 90.67599999997007 |
| teleop_panda_1792549166582667300 | keyboard_exit | 50728 | 101.45599999994495 |
| teleop_panda_1792549322436213600 | wall_timeout | 70219 | 140.43799999994246 |
| teleop_panda_1792549577940202800 | error | 3968 | 0.0 |
| teleop_panda_1792549639781183200 | incomplete_native_close_crash | 未完整保存 | 未完整保存 |
| teleop_panda_1792549882151590600 | incomplete_native_close_crash | 未完整保存 | 未完整保存 |
| teleop_panda_1792549993249892400 | viewer_closed | 9605 | 19.210000000000232 |
| teleop_panda_1792550069788228500 | viewer_closed | 29603 | 59.20600000002639 |
| teleop_panda_1792550259629003800 | keyboard_exit | 7948 | 15.896000000001978 |

每个运行的视频异常、最终状态与原生关闭字段详见其 complete.json；总清单 `results/phase1_5_acceptance.json`。无动作的早期录制可能未通过视频运动校验；不作为演示发布。

## 复现与文件校验

```powershell
. 'D:\miniconda\shell\condabin\conda-hook.ps1'
conda activate panda_safe_rl
cd D:\mujoco_learning\panda_safe_residual_rl
python scripts/teleop_panda.py
# 退出后，如需录制自己下一次操作：
python scripts/teleop_panda.py --record
# 完成操作并退出后，再运行测试：
python -m pytest tests -q
python scripts/validate_teleop_recording.py --run results/runs/teleop_panda_1792550069788228500
```

`results/teleop_recording_validation.json` 保存逐命令位移与在运行后审计的源码/数据 SHA256。模型哈希在 run_config.json。视频哈希在 `results/phase1_5_acceptance.json`，逐帧原始 RGB 哈希在原始目录。安装环境沿用 `panda_safe_rl`，未升级或更改任何依赖，未改旧 Reacher 源码、模型、数据与环境；原 Phase 1 实现仍被原有测试覆盖。

## 尚未实现与暂停边界

- 未实现全关节范围自由移动、姿态控制、路径碰撞检查、连续安全保证、自动避障或规划。
- 不支持障碍场景交互开关；默认只是无额外障碍物的小范围练习。
- 未消除原位置伺服的重力静差，软停止也不是硬件急停。
- 原生生命周期保护只在本 Windows 环境验证，未执行 Linux/macOS 或其他 MuJoCo 版本测试。
- 没有 RRT-Connect、残差 RL、PPO/SAC 训练、Sim-to-Real 或算法性能提升实验。

Phase 1.5 完成，等待用户审核，不自动开展 Phase 2。
