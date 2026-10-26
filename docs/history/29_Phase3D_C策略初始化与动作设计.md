# Phase 3D-C 策略初始化与动作设计

本轮是A=BC85初始化PPO、B=Integral20+小幅PPO的有界对照。冻结配置 `configs/phase3dc.json`；原控制器和模型不改动。初始化/安全测试通过前未训练PPO。

## A：保留BC的确定性策略语义

历史最佳BC85为85→64 Tanh→64 Tanh→7 Tanh，输入均值/标准差来自3D-B训练集，标准差下限0.01。原checkpoint、输入顺序、归一化与哈希均冻结。新增 `BoundedMeanPolicy` 将两个隐藏层及输出线性层完整复制到SB3，并明确以Tanh(linear)作为高斯**均值**。分布为未作Tanh变量变换的七维高斯，初始std=0.03；随机样本到环境边界才裁剪[-1,1]，随后乘0.002 rad。不能将它称为tanh-squashed Gaussian，也不能漏记裁剪前动作。

SB3的get_distribution/predict_values会绕过普通extract_features覆写；第一轮审计正是发现这一点。现已分别保证采样forward、evaluate_actions、确定性predict和predict_values使用同一固定归一化。未修改奖励或通过加宽容差掩盖问题。

8752条历史观测批量动作最大差 `1.19209289551e-07`，保存重载预测完全一致；在两条新训练路径的增益/质量失配下，初始化A与原BC85的完整q_actual/ctrl轨迹逐步完全相同。容差预先设为2e-6动作/关节、2e-6 m TCP，但实际动力学差为0。

## B：Integral20加额外修正

原Integral20因果积分规则不变：0.3/s误差积分、±0.002 rad裁剪、原抗饱和、保护保持清零。每20Hz决策取当前可观测积分目标；RL七维动作映射±0.0002 rad，合成目标clip(integral20 + correction,±0.002)。随后仍经0.01 rad/s残差速率、0.35 rad/s总位置指令速率、实际关节/原native力矩范围及500Hz安全监控。名义补偿模型不读取真实质量/Kp。

原Integral20先将归一化请求转成float32，再映射到rad。首轮零动作比较发现约5.4e-11的差异后，恢复这一确切顺序，而不是把近似相同冒充逐步一致。修正后两种失配条件的零RL输出与原Integral20完整q/ctrl记录**bitwise相同**。正负满幅额外修正均安全完成，非零有效修正比例1.0；正式训练也另记合成裁剪比例，未扩大动作幅度。

## 环境与线程所有权

85维顺序与BC一致，追加积分由学生自己的观测误差在线算出；无真实物理参数或未来参考输入。3D-B已揭示85维记忆与教师标签直接对应，这一阶段不把可计算积分规则称为原创算法。

物理0.002秒/500Hz、RL20Hz，每动作25积分步。GatedShadow在物理线程写积分之前等待预算，避免主线程读取时积分已悄然前进一拍。观测/积分快照在同一锁内复制，渲染仍由物理所有者线程读取live状态；GUI不写MjData。所有真实执行通过ctrl和mj_step，qpos仅reset初始化。

B观测中的上一动作与已施加残差仍采用**总归一化补偿**语义，与A的原BC特征保持一致；Actor原始修正独立记录。两者奖励R1都作用于实际总补偿及其变化，而不是只惩罚B的小修正，保持控制代价定义一致。

## 新任务划分

两条新训练路径(dc_train_sphere/cylinder)、一条验证路径(dc_validation_box)、两条测试路径(dc_final_cylinder/sphere)。真实障碍位置、起终构型不同；新RRT预算种子521/719，均首次521成功并经0.005 rad复核。规划间隙20mm，执行门槛5mm；旧窄间隙案例未用于随机探索。全部配置、场景几何与模型签名见 `results\runs\phase3dc_prepare_1792746020372610600`。

训练覆盖nominal、Kp×0.93、质量惯量×1.07、组合(0.95/1.05/1.10)。按回合与训练seed轮换新条件和路径，同一seed下A/B匹配，物理seed=训练seed×10000+回合序号。验证：nominal、Kp×0.90、质量惯量×1.10，seed7201。最终：Kp×0.86、质量惯量×1.14、组合(0.89/1.11/1.12)，seed8201。测试仅在最终预算、checkpoint及协议冻结后开启，没有把旧已看过任务改名成新测试。

## 共同奖励、预算和门禁

沿用 `src/experiments/phase3c.py::reward_r1`：关节/TCP/速度误差、总补偿及平滑代价、时间/进度、成功2、失败-20、接触额外-40；误差项按真实物理区间/固定轨迹时长归一化。完整各分量与未折扣/GAE/returns保留。原稳定到达标准未改变，无法靠提前保护停止获得成功奖励。

两者Actor/Critic各64×64 Tanh、同一固定BC归一化、Adam学习率1e-4、n_steps256、batch64、n_epochs5、gamma0.995、GAE0.95、clip0.1、vf0.5、最大梯度范数0.5。各seed42/123，先各2048决策；没有最佳种子或最佳checkpoint筛选。

共同门禁在训练前冻结：训练无安全/接触违规，验证所有任务到达且安全，每组至少2个完整失配回合，所有更新Actor/Critic非零且损失有限，有效动作比例≥0.1，验证平均TCP相对各自初始BC/Integral20退化不超过5%。**四组全过才共同到4096，否则全部停在2048**。本轮共同门禁结果 `True`，最终每组4096决策。分阶段停止时保留未完成回合的保护保持/物理成本，继续时保存优化器/RNG、从新回合恢复；不是把预算截断计成任务失败或完整成功。

公平性限制：A复制BC初始权重，B初始零修正且隐藏层随机；两者网络容量和预算相同，但A总动作范围0.002 rad、B额外范围0.0002 rad，不能把归一化探索量直接比较为相同物理作用。B仍受总补偿饱和限制。Critic未BC预训练；两训练种子不足以做显著性结论。

## 观察方式与实测入口

```powershell
Set-Location D:\mujoco_learning\panda_safe_residual_rl
$py = 'D:\miniconda\envs\panda_safe_rl\python.exe'
Start-Process (Resolve-Path 'assets/videos/phase3dc_actual_training.mp4')
Start-Process (Resolve-Path 'assets/videos/phase3dc_policy_comparison.mp4')
# 回放实际随机训练记录，不训练、不重新predict
& $py scripts/watch_ppo_training.py --mode actual --run 'D:\mujoco_learning\panda_safe_residual_rl\results\runs\phase3dc_training_initial_1792746554788972800\A_seed42' --episode 0 --paused
# 本轮实际验收的无窗口导出命令
& $py scripts/watch_ppo_training.py --mode actual --run 'D:\mujoco_learning\panda_safe_residual_rl\results\runs\phase3dc_training_initial_1792746554788972800\A_seed42' --episode 0 --headless-export
```

P/空格暂停，N/B逐决策，+/-调速，Esc/Q退出。Actual Training视频只取各组首个真实随机训练回合（预先固定，未按成功挑选），更新版本可在同一回合中改变；奖励显示的是上一个已完成决策，当前奖励需物理区间结束才可得。独立策略对比另标Baseline Comparison/Checkpoint Evaluation。

兼容性边界：旧watch_ppo_training.py的 `--mode evaluation --checkpoint` 仍绑定旧78维环境，不能直接加载本轮85维PPO。上面的 `--mode actual` 记录回放已实测兼容；新的85维完整动力学评价使用本轮evaluate_phase3dc.py，或直接观看已经保存的策略对比视频。未修改旧播放器的历史评价实现。

实际运行的实验命令（会新建目录并消耗预算，本阶段已完成无需重复）：

```powershell
& $py scripts/prepare_phase3dc.py
& $py scripts/audit_phase3dc.py
& $py scripts/evaluate_phase3dc.py --stage baseline
& $py scripts/train_phase3dc.py --stage initial
& $py scripts/evaluate_phase3dc.py --stage gate
& $py scripts/train_phase3dc.py --stage continue
& $py scripts/evaluate_phase3dc.py --stage post
& $py scripts/evaluate_phase3dc.py --stage final
& $py scripts/analyze_phase3dc.py
& $py scripts/record_phase3dc.py
& $py -m pytest -q tests/test_phase3dc.py -k 'not frozen_final_opening' --basetemp=results/runs/phase3dc_pytest_acceptance_20261023_c --junitxml=results/phase3dc_new_tests_pre_final.xml
& $py -m pytest -q tests/test_phase3dc.py -k frozen_final_opening --basetemp=results/runs/phase3dc_pytest_acceptance_20261023_d --junitxml=results/phase3dc_new_tests_final_only.xml
```

原209项测试已在新增测试收集前完整重跑，日志/XML独立保留。首次本地多进程遭Windows沙箱管道限制，保留失败日志，随后在允许本地工作进程的执行方式下完成；未改仿真、未安装包。GUI按键本轮未重新人工逐键验收；视频解码、动作/状态一致性及无窗口回放命令分别有实际验收记录。
