# Phase 3D-B 专家示教与BC设计

本阶段只训练监督学习BC，未进行PPO更新。冻结协议见 `configs/phase3d_bc.json`，原控制器、模型、历史路径和日志保持原样；发布时重新核对了 8661 个历史文件的SHA256。

## 动作接口与因果时序

教师为原版 `IntegralM3Controller`（M3重力+速度前馈+受限积分），500 Hz。积分增益0.3/s、积分/残差范围±0.002 rad、残差速率0.01 rad/s、总位置指令速率0.35 rad/s，保留原生执行器力矩与关节限位。实际安全门槛仍为5 mm，规划裕度独立为20 mm。

在线积分I每个0.002 s物理步按当前参考与最新可观测关节位置之差更新；上一控制步总指令限速或原生力矩饱和时阻断相应积分分量；裁剪±0.002；回合开始与保护保持时清零。名义补偿模型只读取名义参数和可观测状态。额外的ObservableIntegral不访问教师对象，生成数据时逐物理步与原教师积分比对，全部差值为0。学生运行时使用自己的观测/控制状态执行同一规则。

物理第750步（1.5秒预保持后）开始，每25物理步构造一次观测并选择七维动作。标签是**当前因果期望积分I/0.002，下一段执行之前**，不是后续25步的未来积分，也不是经过残差限速后的执行残差。学生输出[-1,1]^7，映射为±0.002 rad目标，再经过原限速器、总指令限速器、位置伺服和mj_step。分别保存内部可观测积分、教师积分（仅教师记录）、期望残差、实际残差、最终ctrl和原生力矩。

原专家继续500 Hz更新，学生20 Hz持有目标。另设Integral20因果适配对照，避免把专家/学生频率差异误认为学习差距。所有闭环运动均来自data.ctrl与mj_step，动态阶段live qpos写入计数为0。

## 78维和85维输入

原78维保留关节位置/速度、当前参考位置/速度、误差、名义M3基础目标、三个历史误差、进度、上一动作和已施加残差。85维在末尾增加当前自己计算的七维I/0.002。两者均无真实质量、实际Kp或未来参考。对传感器延迟的处理遵循原控制器last observed状态；本pilot未额外加入噪声/延迟实验。

重要限制：这批教师示教中积分变化缓慢，期望残差与已施加残差恰好相同；经逐样本核对，标签也与78维中的上一实际残差字段相等。85维额外积分同样直接对应标签。这是可在线获得的反馈记忆，不是隐藏教师状态；但造成共线性和复制捷径，离线低MSE不能证明学会在自身状态上积累补偿。报告保留这一问题，不在看过测试后重新选标签或调模型。

## 隔离场景及预算

新建两条训练路径、一条验证路径、两条最终测试路径，障碍形状/位置、起终关节构型实际不同；完整参数、模型签名、规划种子和路径在 `results\runs\phase3d_bc_prepare_1792743213646170700`。RRT种子预算211、307；五条路径均首次211成功且通过0.005 rad独立复核，未启动第二种子搜索。

训练：nominal、Kp×0.95/0.90、质量及惯量×1.05、阻尼×1.10、组合(0.95/1.05/1.10)，两个物理种子4101/4102。验证：nominal、Kp×0.92、质量惯量×1.08、组合(0.94/1.07/1.08)，种子5101。最终：两个新布局×Kp×0.88、质量惯量×1.12、组合(0.90/1.10/1.15)，种子6101。未复用3C已开启的heldout_box作为新测试。

先4条pilot，再完成24训练+4验证示教。质量筛选冻结为：有效有限模型、稳定到达、无模型警告/控制异常/禁止接触/5 mm违规、残差范围正确、独立积分逐步一致。28条全部合格，0拒绝，原始记录全保留。8752样本：7716训练、1036验证；真实示教239536物理步。24训练记录仅12条独特观测—标签序列，另外12条为不同种子下确定性重复，不能视为24独立任务。最终集无示教标签参与优化。

## 监督训练及冻结

两层64单元Tanh，七维Tanh输出，输出层初始零；Adam 0.001，batch256，seed42，最大150 epoch，验证不改善25轮早停。输入均值/标准差仅用训练数据，标准差下限0.01，模型保存归一化参数。分别按独立验证动作MSE选择best，保存initial/best/final。BC78实际31轮961步优化，best第6轮；BC85实际37轮1147步优化，best第12轮。两次监督训练约3.54秒墙钟，不含采集和评价。

先冻结模型，再验证闭环；最终开启文件记录协议、BC源码和checkpoint哈希，此后未调参/重训/挑选。两模型实际各训练1个种子，不能声称跨训练种子稳定性。BC85不是现有78维Gym PPO的直接可加载权重，后续需单独验证85维适配、归一化及Tanh动作头兼容性。

## 运行及观看

```powershell
Set-Location D:\mujoco_learning\panda_safe_residual_rl
$py = 'D:\miniconda\envs\panda_safe_rl\python.exe'
# 播放真实动力学现场录像，不启动训练
Start-Process (Resolve-Path 'assets/videos/phase3d_bc_vs_integral.mp4')
# 复用3D-A播放器，完整记录状态重建，逐个窗口比较
& $py scripts/watch_ppo_training.py --mode comparison --replay-only --run results/runs/phase3d_bc_video_1792744038891360000/Integral500 --other-run results/runs/phase3d_bc_video_1792744038891360000/BC78 --other-run results/runs/phase3d_bc_video_1792744038891360000/BC85 --paused
```

播放器P/空格暂停，N/B逐决策前后，左右键逐记录，+/-调速，Esc/Q退出当前窗口；关闭一个比较窗口后打开下一个。该入口为Mode C Baseline Comparison，明确显示FULL RECORDED STATE RECONSTRUCTION；它恢复记录状态用于显示，不重新采样动作、不训练，也不冒充新的动力学执行。真正现场RGB来自上述MP4，五段依次M3、Integral500、Integral20、BC78、BC85。

实际执行过的无窗口复用验证：

```powershell
& $py scripts/watch_ppo_training.py --mode comparison --replay-only --run results/runs/phase3d_bc_video_1792744038891360000/Integral500 --other-run results/runs/phase3d_bc_video_1792744038891360000/BC78 --other-run results/runs/phase3d_bc_video_1792744038891360000/BC85 --headless-export
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
