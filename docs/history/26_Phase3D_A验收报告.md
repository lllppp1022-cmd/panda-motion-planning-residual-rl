# Phase3D-A训练观察验收报告

日期2026-10-23。任务是观察和诊断，不以提高PPO指标为目标。完成后暂停。

## 实际执行范围

先审计4组历史继续训练run、32个episode；冻结3355个历史文件SHA256，结束时全部一致。历史仅可重建七关节可视状态，没有历史现场RGB。

| 运行 | PPO决策 | 物理积分 | reset | cleanup | 更新/Adam | 录制问题 |
| --- | --- | --- | --- | --- | --- | --- |
| 原关闭观察器 | 1024 | 28791 | 3000 | 251 | 4/80 | 0 |
| 初版开启记录 | 1024 | 28791 | 3000 | 251 | 4/80 | 35条预览共享冲突 |
| 修复后开启记录 | 1024 | 28791 | 3000 | 251 | 4/80 | 0 |

合计3072个PPO决策、86373个训练相关真实积分；每次上限1024。接口探针/自动测试与7次独立确定性评价另存，不混入训练决策数。并非100k训练。最终修复版：`D:\mujoco_learning\panda_safe_residual_rl\results\runs\phase3d_observation_1792741615845772700\observer_on`。关闭组复用：`D:\mujoco_learning\panda_safe_residual_rl\results\runs\phase3d_observation_1792740617259955400\observer_off`。初版及失败诊断：`D:\mujoco_learning\panda_safe_residual_rl\results\runs\phase3d_observation_1792740617259955400`。

## 核心等价性与视频证据

采样、观察、奖励、terminated/truncated、tracking、实际控制信号（剔除algorithm_s墙钟耗时）、全部网络参数及Adam状态逐位一致；256/512/768/1024更新点一致。Actor/Critic每次均真实更新。ZIP路径元数据不同，验收采用每个参数及优化器张量一致，而非强迫ZIP哈希一致。

| Episode | 视频帧 | 时长(s) | 真实关节位移(rad) | 策略版本 | 结局 |
| --- | --- | --- | --- | --- | --- |
| 0 | 151 | 15.1 | 1.030019 | [0, 1] | success |
| 1 | 151 | 15.1 | 1.030055 | [1, 2] | success |
| 2 | 151 | 15.1 | 1.030085 | [2, 3] | success |
| 3 | 126 | 12.6 | 0.885107 | [3] | rl_external_truncation |

最终影片579帧、57.9秒；MP4已解码验收，状态/动作/ctrl/TCP/间隙逐帧对应原始CSV和完整状态NPZ，RGB在物理owner线程从同一个live MjData取得。新状态保存全部9维qpos/qvel和8维ctrl；旧历史夹爪缺失不伪造精确重建。影片只是对现场RGB加已记录奖励字幕，没有重新调用策略生成替代轨迹。

本轮3个真实完整回合均成功，1个因预算结束而未完成；未出现真正任务失败、安全违规或禁止接触。预算退出的保护保持已保存，不制作虚假失败样例。

## 五个checkpoint与传统基线的独立评价

同一gain90任务、同起点/终点、物理seed9001。它们是Checkpoint Evaluation / Baseline Comparison，绝不混作Actual Training。初版和修复版各更新权重哈希相同，复用真实评价，未重复优化。

| 方法/版本 | TCP轨迹RMSE(mm) | 完成 | 5mm违规步 |
| --- | --- | --- | --- |
| Integral | 0.471059 | True | 0 |
| M3 | 0.881533 | True | 0 |
| update00 | 0.881533 | True | 0 |
| update01 | 0.876711 | True | 0 |
| update02 | 0.874847 | True | 0 |
| update03 | 0.871686 | True | 0 |
| update04 | 0.869630 | True | 0 |

这些短实验不宣称统计显著或PPO优于积分补偿，也不修改Phase3C结论。

## 工程问题、修复与真实验收边界

- 初始2步接口探针发现重复episode显示字段，已修复，原目录保留。
- 初次1024录制有35条Windows预览PNG共享冲突。原始MP4/完整物理状态仍完整，训练等价性通过；并未把预览错误改写成成功日志。修复为不可变帧文件和有界JSON替换重试；32步非训练接口及修复后的1024对照均通过。
- 并行B/C回放曾因旧timestamp目录名冲突启动失败；改用带PID的独立观察目录，失败日志`results/phase3d_mode_c.log`和成功重跑日志`results/phase3d_mode_c_fixed.log`均保留。
- Windows现场RGB窗口已实际显示并读取截图；历史窗口日志记录了真实暂停、调速、关闭事件。用户将窗口最小化后，自动输入工具报告`user input was detected in this window`，刷新仍受保护；未强行覆盖用户操作。N/B的决策索引和帧对齐有自动测试，但没有宣称完成全部桌面按键注入验收；红色主动中止按钮未在正式对照中点击。
- 直接绑定live MjData的原生MuJoCo Viewer未实现；采用任务允许的物理线程RGB采集+独立同步窗口。GUI只读文件副本，不访问MjData，不调用step/reset。默认没有高频GPU渲染。

原186项和新增11项实际通过，合计197/197，0跳过0失败。合并JUnit/日志位于results/phase3d_tests.xml及results/phase3d_tests.log。新增实际覆盖跨rollout策略版本、完整状态、逐帧动作对应、录制开关等价性、GPU录制无状态副作用、故障隔离、预览修复、历史文件只读及独立评价。

## 交付及暂停

教程：docs/25_Phase3D_A训练可视化使用教程.md。主程序：scripts/watch_ppo_training.py；现场观察训练：scripts/train_phase3d_observed.py。视频：assets/videos/phase3d_actual_training.mp4。动作/更新/TCP/奖励/独立评价五图：assets/plots/phase3d_*.png。完整动作/状态/帧证据索引：results/phase3d_recording_validation.json。具体运行命令和三种来源的区别详见教程。

未执行：100k或更大训练、BC训练、实机实验、严格连续安全认证；没有完整任务失败可供本轮录制。原生live Viewer及全部桌面按键自动注入验收仍是限制。已停止PPO更新，现存显示窗口可Q/Esc/X关闭，仅查看不会启动训练。
