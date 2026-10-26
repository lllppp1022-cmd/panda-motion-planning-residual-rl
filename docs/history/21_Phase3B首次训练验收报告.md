# Phase 3B 首次在线残差PPO训练验收报告

日期：2026-10-22。工程：`D:\mujoco_learning\panda_safe_residual_rl`。本轮短训练已完成并暂停，没有启动100k或其他长训练。

## 结论

**确实完成了真实PPO交互与Actor/Critic优化。基础环境与烟雾训练验收通过；尚不能证明PPO优于M3，不建议未经下一轮验证直接开展大规模正式训练。** 当前只有一个训练种子、1024个决策步、3个完整采样回合，主要结论是工程链路可用。小幅误差变化不是统计显著性或鲁棒性证据。

## 历史与新场景

历史冻结清单：`D:\mujoco_learning\panda_safe_residual_rl\results\runs\phase3b_prepare_1792657739286720700\history_sha256.json`；2132个文件在交付时逐一SHA256一致，包括Phase3A成果、旧模型、原控制器、157项测试和历史RRT路径，未覆盖。新train/validation/final三布局以新起终点及障碍位置实际规划，规划种子73；20mm规划门槛、0.005rad独立复核、5mm实际失败门槛。没有把旧heldout_box冒充新测试。

各路径真实配置、树、Shortcut CSV、场景及模型指纹在`D:\mujoco_learning\panda_safe_residual_rl\results\runs\phase3b_prepare_1792657739286720700`。最终盒形布局只在冻结最终模型、完成25次验证评价后打开，执行6次最终评价。该测试集现在已经查看，未来若调参应另建新的最终未见集。

## 首轮非零动作失败与修复

原候选±0.002rad没有独立残差变化率约束。12次全幅探针中4次交替动作未能稳定到达，原因均为`goal_not_stably_reached`，不是5mm或接触违规；原始失败、全部动作、物理和奖励日志完整保留。未放宽0.02rad/10mm/0.02rad/s保持0.5秒的原成功标准，也未在失败门禁下训练。

随后在M3之后、原总指令限速之前加入**0.01rad/s独立残差变化率约束**，把实际残差状态加入Actor观测，防止隐藏滤波状态。重新运行Gym检查、两个完整零残差等价对照和12次有界全幅测试。修复后12/12稳定到达、无5mm违规/禁止接触；另一次运动中退出保留了真实受限保持及额外关节运动。安全作用仅限这些状态和输入，不能推广为±0.002rad全局安全保证。

原候选目录：D:\mujoco_learning\panda_safe_residual_rl\results\runs\phase3b_safety_1792658056441764900。

## 环境、观测与奖励

物理0.002s、500Hz原位置伺服；PPO20Hz、通常每动作25次真实积分。环境用条件变量授权原TrajectoryExecutor状态机积分，没有重新编造完成判据。每个reset的750步保持、失败/退出后的保持、最后不足25步的决策均实际计数。live qpos仅允许reset初始化，运行中通过data.ctrl及mj_step运动。原274个几何配对和真实夹爪/原生接触每步检查保持不变。

78维固定归一化观察只包括可观测q/dq、参考q/dq、跟踪误差、M3基底目标、3次历史误差、进度、上一归一化动作及实际残差状态。Actor没有真实质量、惯量、Kp、阻尼、工况名或精确障碍距离。真实仿真间隙用于奖励/安全日志，属于评价信息而非Actor特征。噪声/观测延迟仍是上层模型，不冒充底层编码器或网络化原生伺服链。

奖励分量逐决策记录：关节/速度误差、残差幅度/变化、时间、参考进度；稳定完成+200、失败或Gym超时−200、禁止接触另−400。800步内最差完成的未折扣回报仍高于提前失败。未完轨迹、保护停止不发完成奖励。真正失败/到达terminated，Gym决策预算truncated；训练总预算结束引发的清理保持独立归档，不回填为成功或虚构PPO终止样本。

详见[环境设计](20_Phase3B残差RL环境设计.md)。

## 零残差等价与依赖

名义、Kp×0.95两组，各完整8919个物理步，原Phase3A M3与新环境零残差在q_ref/dq_ref/q_actual/dq_actual/ctrl、时间、TCP误差及间隙逐项字符串/浮点记录完全一致。初始策略确定性评价的四个共同任务同样与M3逐步一致。

实际解释器为`D:\miniconda\envs\panda_safe_rl\python.exe`。本机指定环境原来缺少RL包，通过追加已有`mujoco_learn` site-packages只读复用torch 2.14.1+cpu、Gymnasium1.4.0、SB3 2.9.0、TensorBoard2.21.0；仍优先使用Panda numpy2.4.6及MuJoCo3.16.0。不修改任何Reacher工程/历史环境，依赖来源完整写入训练配置。独立安装流程没有执行或验证，后备路径是本机复现的一个限制。

## 真实训练量与参数更新

训练目录：`D:\mujoco_learning\panda_safe_residual_rl\results\runs\phase3b_training_1792658731055848900`。种子42，64×64 Actor/Critic，初始Actor均值层权重/偏置0、高斯标准差0.1；训练使用随机动作，评价用deterministic=True。学习率1e-4，rollout256、batch64、5epoch、gamma0.995、GAE0.95、clip0.1；残差目标±0.002rad并受0.01rad/s限制。

- 实际PPO决策：**1024**。
- 决策交互中的实际MuJoCo积分：**25582**，包含正常完整回合最后不足25步的决策。
- 4次reset初始保持积分：**3000**。
- 训练结束清理保持：**251**；其中1步已进入原控制器的门控等待，随后执行原250步保持，未隐藏继续运动。
- 总实际积分：**28833 = 25582 + 3000 + 251**。
- 完整真实回合：3个，均成功，结束于决策327/654/981；第4回合在43个决策后因总预算结束，记未完成，不计成功率分母或失败回合冒充学习结果。
- reset实际工况次数：`{'nominal': 3, 'mild_gain': 1}`。这只是本轮真实抽样，不能声称每种工况都得到充分学习。
- 训练及退出期间5mm违规回合0、禁止接触回合0。
- PPO train调用4次，累计20epoch；每epoch4个minibatch，合计80次Adam优化步。
- Actor总参数L2变化0.05299441，Critic 0.51908947；均值输出层最终权重范数0.01955486，不是仅修改了日志。
- 随机动作非零比例100.0%，最大已执行归一化动作0.399157。保存原始高斯动作和裁剪动作，可对应到真实位置残差与ctrl。

|更新|决策步|Actor L2增量|Critic L2增量|策略梯度loss|value loss|approx KL|
|---|---:|---:|---:|---:|---:|---:|
|1|256|0.024854|0.084373|-0.014929|0.0081|0.0041045|
|2|512|0.021806|0.158827|-0.005437|1460.0822|0.0010603|
|3|768|0.014941|0.192470|-0.005329|1456.0025|0.0007670|
|4|1024|0.013090|0.181728|-0.002699|1451.7702|0.0005631|

最终模型：`D:\mujoco_learning\panda_safe_residual_rl\results\runs\phase3b_training_1792658731055848900\checkpoints\ppo_smoke_final.zip`，SHA256 `32ae03f299209d96056b1f044b2a290224ec52997daf9fa410ca8f6b354f163e`。初始、每次更新后、最终checkpoint均保留；最终模型保存/重载后确定性预测完全一致。TensorBoard事件：`D:\mujoco_learning\panda_safe_residual_rl\results\runs\phase3b_training_1792658731055848900\tensorboard`。value loss及短样本表现只能说明发生优化，不能据此声称Critic已收敛。

## 初始与短训策略的配对评价

共31次实际评价：M1/M3/最终PPO各9个同任务条件，初始PPO另4个共同任务。主对比不混用不同任务集。所有物理种子8001，训练/验证/最终布局分别保留标识，禁止接触和失败都计入，没有筛掉较差结果。

|方法|稳定到达|5mm违规回合|禁止接触回合|TCP轨迹RMSE中位mm|最大最终TCP误差mm|
|---|---:|---:|---:|---:|---:|
|M1|9/9|0|0|4.692961|0.363468|
|M3|9/9|0|0|0.118436|0.363997|
|M3_PPO|9/9|0|0|0.116753|0.378902|

初始确定性PPO的4个共同任务与M3逐步一致，不能把初始确定性回放称为随机训练。最终PPO相对M3的9个配对条件TCP RMSE变化中位数为-1.393%（负号改善，正号退化）：

|布局|条件|M3 TCP RMSE mm|短训PPO TCP RMSE mm|变化|
|---|---|---:|---:|---:|
|b_train_sphere|nominal|0.085507|0.084316|-1.393%|
|b_train_sphere|mild_gain|0.357639|0.361920|+1.197%|
|b_validation_cylinder|nominal|0.087099|0.085627|-1.690%|
|b_validation_cylinder|mild_gain|0.340969|0.344818|+1.129%|
|b_validation_cylinder|mild_mass|0.300256|0.302496|+0.746%|
|b_validation_cylinder|mild_damping|0.085979|0.084536|-1.677%|
|b_validation_cylinder|mild_observation|0.087994|0.086551|-1.640%|
|b_final_box|nominal|0.118436|0.116753|-1.421%|
|b_final_box|mild_gain|0.358058|0.361866|+1.064%|

这一个训练种子的微小差异不支持“PPO必然超过M3”。评价详情与七关节RMSE、实际速度/加速度、指令变化率、饱和和失败原因均在`results/phase3b_evaluation_comparison.csv`；原始CSV可复算。M1保留为辅助基线，M3才是当前主要传统对照。受限积分/自适应尚未实现，不将其说成已被PPO超过。

## 自动验收与真实日志

全部**173项测试通过、0失败/0跳过**，原157项保留；JUnit/日志：`results/phase3b_release_tests.xml/log`。新增覆盖接口、完整零残差、动作映射及残差/总指令限速、非法动作、真实25步、预算截断和受限保持、原交替失败terminated分类、PPO双网络更新、保存重载、最终测试冻结、无参数特权输入及历史文件一致性。

开发期一次pytest模块导入失败已经修复并保留原日志。初始无残差slew的4次真实失败没有删除，修复后重复门禁而非放松验收标准。独立CSV审计核验31次评价和4个训练物理回合，实际力矩映射、原生力矩限幅、总指令0.35rad/s、残差0.01rad/s及训练各奖励分量均通过，训练物理步总数从原始CSV重新计数一致。

回报图：`assets/plots/phase3b_reward_curve.png`。图中的稀疏完成奖励尖峰反映回合结束位置，不代表学习收敛。原始随机样本：`results/phase3b_smoke_training.csv`；每个episode保留完整tracking/control_signals/clearance/events/decisions及模型参数JSON，未用独立predict回放冒充采样。

## 可复现命令与剩余事项

以下短训练命令本轮实际执行过；再次执行会建立新的独立运行目录和checkpoint，但仍只训练1024个决策步：

```powershell
Set-Location D:\mujoco_learning\panda_safe_residual_rl
& D:\miniconda\envs\panda_safe_rl\python.exe scripts/train_phase3b.py --decisions 1024
```

验证入口（均已实际执行）：

```powershell
& D:\miniconda\envs\panda_safe_rl\python.exe scripts/validate_phase3b.py --stage interface
& D:\miniconda\envs\panda_safe_rl\python.exe scripts/validate_phase3b.py --stage equivalence
& D:\miniconda\envs\panda_safe_rl\python.exe scripts/validate_phase3b.py --stage safety
& D:\miniconda\envs\panda_safe_rl\python.exe -m pytest -q
```

没有执行2048决策、100k或任何长训练，没有进行多种子训练、大规模组合随机化、受限积分/自适应对照、实机或连续安全认证。历史D窄间隙路径未进入本轮PPO训练或评价，不能声称已经修复Phase3A已知失败。新测试集已打开，下一轮调参需再保留新最终测试布局。完整环境链路可以支撑下一轮明确授权的小规模研究，但**当前证据不支持直接启动大规模正式训练**；先补多种子有限pilot、传统强基线、奖励/观察分析及更丰富安全验证。阶段到此暂停。
