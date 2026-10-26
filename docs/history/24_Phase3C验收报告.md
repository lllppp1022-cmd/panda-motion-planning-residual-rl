# Phase3C有限Pilot验收报告

日期2026-10-22。完成后暂停；未启动100k PPO决策训练。以下所有数值来自真实动力学CSV及checkpoint，不修改Phase3B历史。

## 运行范围与共同门禁

四组R0/R1×42/123均先执行2048决策；中期两个失配条件分别确定性评价，完整记录见`results/phase3c_2048_gate.json`。只按预先共同门禁决定是否四组一起继续，最后checkpoint按共同预算固定，不按最终测试挑选。

| 奖励 | 种子 | PPO决策 | 真实积分步 | 其中reset/cleanup | 成功完整回合/预算中断 | 更新/Adam步 | 末Value Loss | 末EV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | 42 | 4096 | 114622 | 12000/502 | 14/2 | 16/320 | 1390.797673 | 0.049279 |
| R0 | 123 | 4096 | 114622 | 12000/502 | 14/2 | 16/320 | 1408.861246 | 0.025527 |
| R1 | 42 | 4096 | 114622 | 12000/502 | 14/2 | 16/320 | 0.015493 | 0.953830 |
| R1 | 123 | 4096 | 114622 | 12000/502 | 14/2 | 16/320 | 0.029863 | 0.897472 |

训练合计16384个PPO决策、458488个真实MuJoCo积分步；包含reset静态保持、决策执行、各阶段关闭时保护保持。每组阶段边界未结束的回合标记为预算中断，不当作到达成功，也不伪造失败的完整回报。继续阶段从保存的网络、Adam、Python/NumPy/Torch随机状态恢复，环境明确重置并推进工况序号，不伪称无缝跨episode继续。

网络两支每次均更新，所有rollout NPZ含旧值、GAE、价值目标、原始动作及logprob；checkpoint、TensorBoard、参数SHA、随机状态、完整CSV均保存。模型的确定性评价仍需读入机器人状态，不读取实际质量、增益或惯量。

## 新场景与隔离

新训练sphere、新验证cylinder、新最终box的障碍布局和起终点均不同，RRT路径均0.005rad复核，规划间隙20mm、执行门槛5mm；不是历史heldout_box更名。名义、gain90、mass110单因素共三工况，名义控制器不读真实扰动参数。最终测试开启前写入冻结方法、配置、四个checkpoint哈希及验证路径；没有根据最终结果再选奖励或模型。这里只覆盖每split一个几何任务与三工况，不代表广泛泛化。

## 真实确定性评价

各方法在每split三工况配对比较，物理seed9001；传统基线不重复当作两个独立随机种子。主TCP/关节RMSE与原执行器口径一致，包含trajectory和settle阶段；另列full_record指标纳入prehold和保护保持。安全、接触和实际积分数量统计全记录，所有失败也保留，完成进度与终止原因另列CSV。未用仅成功回合过滤误差。

### validation

| 方法 | nominal TCP RMSE(mm) | gain90 TCP RMSE(mm) | mass110 TCP RMSE(mm) | 稳定到达 | 5mm违规/禁止接触 |
| --- | --- | --- | --- | --- | --- |
| M1 | 4.569658 | 5.136335 | 4.612948 | 3/3 | 0/0 |
| M3 | 0.081872 | 0.853103 | 0.667550 | 3/3 | 0/0 |
| Integral | 0.169416 | 0.355354 | 0.222872 | 3/3 | 0/0 |
| R0_seed42 | 0.073413 | 0.852868 | 0.659926 | 3/3 | 0/0 |
| R0_seed123 | 0.092754 | 0.847147 | 0.667200 | 3/3 | 0/0 |
| R1_seed42 | 0.078753 | 0.846513 | 0.656918 | 3/3 | 0/0 |
| R1_seed123 | 0.088670 | 0.852747 | 0.670416 | 3/3 | 0/0 |

### final

| 方法 | nominal TCP RMSE(mm) | gain90 TCP RMSE(mm) | mass110 TCP RMSE(mm) | 稳定到达 | 5mm违规/禁止接触 |
| --- | --- | --- | --- | --- | --- |
| M1 | 3.498622 | 3.931436 | 3.531502 | 3/3 | 0/0 |
| M3 | 0.063458 | 0.765362 | 0.635973 | 3/3 | 0/0 |
| Integral | 0.126084 | 0.283645 | 0.153991 | 3/3 | 0/0 |
| R0_seed42 | 0.060658 | 0.762864 | 0.630175 | 3/3 | 0/0 |
| R0_seed123 | 0.070146 | 0.762719 | 0.635393 | 3/3 | 0/0 |
| R1_seed42 | 0.060632 | 0.759247 | 0.626799 | 3/3 | 0/0 |
| R1_seed123 | 0.070028 | 0.765954 | 0.638734 | 3/3 | 0/0 |

## 对研究问题的回答

- R0 seed42：验证/最终共4个失配条件相对M3的TCP RMSE变化（正值改善）为+0.028%, +1.142%, +0.326%, +0.912%；超过积分基线0/4。
- R0 seed123：验证/最终共4个失配条件相对M3的TCP RMSE变化（正值改善）为+0.698%, +0.052%, +0.345%, +0.091%；超过积分基线0/4。
- R1 seed42：验证/最终共4个失配条件相对M3的TCP RMSE变化（正值改善）为+0.772%, +1.593%, +0.799%, +1.443%；超过积分基线0/4。
- R1 seed123：验证/最终共4个失配条件相对M3的TCP RMSE变化（正值改善）为+0.042%, -0.429%, -0.077%, -0.434%；超过积分基线0/4。

严格按两个种子、两个几何任务及两个失配条件均改善的描述性条件：{'R0': True, 'R1': False}。这不是显著性检验，也未预先设定最小有用效应，微小变化不能等同工程收益。

R1的毫米敏感跟踪项和较小终止奖励更便于解释价值目标，旧+200主导的尺度问题已改善；更小的Value Loss不能作为R1控制优越的证据。受限积分基线保留M3前馈，ki=0.3/s、±0.002rad、0.01rad/s，抗饱和与延迟观测均独立记录；同一套参数先通过12次动作/传统控制门禁，最终测试后未调参。PPO必须同时超过M3和这个传统基线才有充分的额外收益证据，不能只挑某一工况或一粒种子。

训练5mm违规0物理步、禁止接触观察0次；42次评价中违规回合0，禁止接触回合0。全部实际残差最大0.00080474rad，未扩大探索边界；新动作探针中反馈降低误差、反向反馈增大误差，说明通道有效，但不构成全状态安全证明。

## 当前决策与限制

不自动扩大训练预算。只有在预先冻结的新失配任务上出现跨种子的、相对强传统控制器可重复且有实际意义的收益，同时维持安全条件，才值得进入下一阶段。当前有限Pilot只能检验这批安全较宽、轻度失配任务；未研究强组合扰动、时延/噪声泛化、更广场景，也不声称统计显著、连续碰撞安全或实机安全认证。

原173项与新增回归必须实际执行，测试详情另见`results/phase3c_tests.xml`和`results/phase3c_tests.log`；日志生成脚本不会把尚未运行的测试写成通过。`results/phase3c_raw_log_reaudit.json`独立复算训练奖励、内部GAE递推、动作映射、残差/总指令限速、积分步、评价RMSE、实际间隙及力矩约束。新NPZ末bootstrap未单独重建Critic预测，未虚称这一项独立精确复算；Phase3B四块bootstrap和优化器已完整逐参数复放。

## 运行与查看

以下为本轮实际使用的入口，重新运行会生成独立目录；训练命令会启动新实验，查看现有结果请先使用TensorBoard/报告。

```powershell
cd D:\mujoco_learning\panda_safe_residual_rl
& D:\miniconda\envs\panda_safe_rl\python.exe scripts/train_phase3c.py --stage initial
& D:\miniconda\envs\panda_safe_rl\python.exe scripts/evaluate_phase3c.py --stage gate
# 仅共同门禁全部通过才允许继续：
& D:\miniconda\envs\panda_safe_rl\python.exe scripts/train_phase3c.py --stage continue
& D:\miniconda\envs\panda_safe_rl\python.exe scripts/evaluate_phase3c.py --stage validation
& D:\miniconda\envs\panda_safe_rl\python.exe scripts/evaluate_phase3c.py --stage final
& D:\miniconda\envs\panda_safe_rl\python.exe scripts/summarize_phase3c.py
& D:\miniconda\envs\panda_safe_rl\python.exe -m pytest -q
```

### 最终模型与真实日志

- R0 seed42：`D:\mujoco_learning\panda_safe_residual_rl\results\runs\phase3c_training_continue_1792662132222765100\R0_seed42\checkpoints\ppo_4096.zip`；SHA256 `4d9b557c94eaa536cd57e245915d6ea44a214916be1f6b35be1e5d67305661b3`。同目录`tensorboard/`、`training_samples.csv`、`rollout_*.npz`及`actual_training/episode_*/`。
- R0 seed123：`D:\mujoco_learning\panda_safe_residual_rl\results\runs\phase3c_training_continue_1792662132222765100\R0_seed123\checkpoints\ppo_4096.zip`；SHA256 `d7c0903365eb98093ccd3f75bd28db4acc6bad88d449335daf37ce17e4fd3b12`。同目录`tensorboard/`、`training_samples.csv`、`rollout_*.npz`及`actual_training/episode_*/`。
- R1 seed42：`D:\mujoco_learning\panda_safe_residual_rl\results\runs\phase3c_training_continue_1792662132222765100\R1_seed42\checkpoints\ppo_4096.zip`；SHA256 `8ad1738acf72e1131dd57f14bf894d82abe35f41a5ebe4bdf8e4d866f582249f`。同目录`tensorboard/`、`training_samples.csv`、`rollout_*.npz`及`actual_training/episode_*/`。
- R1 seed123：`D:\mujoco_learning\panda_safe_residual_rl\results\runs\phase3c_training_continue_1792662132222765100\R1_seed123\checkpoints\ppo_4096.zip`；SHA256 `8ceb24791eb2331b72787db27371e87c699b83fc0fc395d5188d21af8f98eb43`。同目录`tensorboard/`、`training_samples.csv`、`rollout_*.npz`及`actual_training/episode_*/`。

结果索引：`results/phase3c_acceptance.json`；规划源：`results/phase3c_prepare_latest.json`；2048安全评价：`results/phase3c_eval_gate_latest.json`；验证/最终评价：`results/phase3c_eval_validation_latest.json`、`results/phase3c_eval_final_latest.json`。

未执行：100k决策训练、更多随机种子、强组合域随机化、观测噪声/时延专项、实机部署。本阶段没有视频交付要求；未用确定性回放冒充随机训练。Phase3B真实试错观察系统及历史视频原样保留。

## 实际工程验收记录

原173项+新增13项，合计186/186实际通过，0跳过、0失败。独立原始JUnit与终端日志合并到`results/phase3c_tests.xml`、`results/phase3c_tests.log`；未伪造单次测试时长。2652个历史文件SHA256均与开发前冻结清单一致；四组两阶段的训练源代码/配置哈希亦保持一致。两张PNG成功解码。

本轮曲线已在`http://127.0.0.1:6027/`实际验证HTTP200、8个训练run及标量标签。6026仍可能展示旧3B日志；训练已完成，曲线停止追加是预期行为。查看当前已保存曲线（不启动训练）：

```powershell
cd D:\mujoco_learning\panda_safe_residual_rl
& D:\miniconda\envs\panda_safe_rl\python.exe scripts/view_phase3c.py --port 6027
```

选择`train/value_loss`、`train/explained_variance`、`rollout/ep_rew_mean`；`recomputed_diagnostics`标签由实际已保存CSV/NPZ离线导出，含奖励分量和GAE分布，没有额外环境采样。R0到达奖励+200占主导，回报图变化小不等于Actor完全不变；同时查看真实TCP误差、实际残差及更新事件。

## 六项研究结论

1. **是否真正改善**：R0相对M3在本轮两个几何任务×两个失配工况×两个训练种子的8次配对中均降低TCP轨迹RMSE，但只有很小的数值收益。未证明存在显著或具有实际用途的控制优势。
2. **是否跨两个种子**：R0满足本轮全部8次失配配对改善；R1未满足，seed123在部分条件下恶化。名义条件下两个奖励的seed123均比M3差，不能忽略这一取舍。
   - R0 seed42：四个失配条件平均改善+0.6019%，范围[+0.0276%, +1.1421%]。
   - R0 seed123：四个失配条件平均改善+0.2967%，范围[+0.0524%, +0.6981%]。
   - R1 seed42：四个失配条件平均改善+1.1517%，范围[+0.7725%, +1.5927%]。
   - R1 seed123：四个失配条件平均改善-0.2248%，范围[-0.4340%, +0.0417%]。
3. **哪种奖励更合理**：R1的毫米误差尺度、较小终止奖励及EV更便于诊断价值学习，价值拟合明显改善；但真实控制结果未显示R1比R0稳定更好，不能仅凭较小loss选为获胜方法。两者定义均在最终测试前冻结。
4. **安全代价**：四组训练以及42次正式验证/最终评价均无5mm违规和禁止接触，没有观察到安全退化；这仅限已经验证的宽裕度、轻度失配任务。
5. **额外收益**：四个PPO在4个失配条件下超过受限积分补偿的次数为0/16。积分在失配条件中明显更有效，但名义误差比M3大；保留全部结果，不宣称积分全工况最优。
6. **是否扩大训练**：当前不建议直接扩大PPO预算。优先保留M3与受限积分作为强基线，只有确认仍存在传统方法无法有效解决、可重复且有实际幅度的残余问题后，再冻结新任务开展有限实验。本轮已停止，不继续自动训练。
