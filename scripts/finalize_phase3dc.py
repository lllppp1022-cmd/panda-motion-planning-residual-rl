from _bootstrap import ROOT,main_guard
from src.rl.dependencies import activate
activate()
import xml.etree.ElementTree as ET
import numpy as np
from src.phase3dc.common import *


def main():
    analysis=read(ROOT/'results/phase3dc_analysis.json');gate=analysis['gate'];ev=analysis['evaluations'];tr=analysis['training'];actions=analysis['actions']
    interface_run=Path(read(ROOT/'results/phase3dc_interface_latest.json')['run']);interface=read(interface_run/'complete.json')
    video=read(Path(read(ROOT/'results/phase3dc_video_latest.json')['run'])/'complete.json');final=Path(read(ROOT/'results/phase3dc_evaluation_final_latest.json')['run']);opening=read(final/'final_test_opening.json')
    for n,h in opening['code_hashes'].items():assert sha(ROOT/n)==h
    for c in opening['checkpoints']:
        if c['path']:assert sha(c['path'])==c['sha256']
    history=read(prepared()/'history_sha256.json');assert all(sha(ROOT/n)==h for n,h in history.items())
    tests=0;combined=ET.Element('testsuites')
    for part in ('historical','new'):
        root=ET.parse(ROOT/f'results/phase3dc_{part}_tests.xml').getroot();suites=[root] if root.tag=='testsuite' else list(root)
        for s in suites:
            assert all(int(s.get(k,0))==0 for k in ('failures','errors','skipped'));tests+=int(s.get('tests'));combined.append(s)
    assert tests==222
    ET.ElementTree(combined).write(ROOT/'results/phase3dc_all_tests.xml',encoding='utf-8',xml_declaration=True)
    methods=['M3','Integral500','Integral20','BC85','A_seed42','A_seed123','B_seed42','B_seed123'];metrics={}
    for split in ('validation','test'):
        metrics[split]={}
        for m in methods:
            rows=[r for r in ev if r['split']==split and r['method']==m and (split=='test' or m in methods[:4] or r['stage']==('post' if gate['all_four_passed'] else 'gate'))]
            metrics[split][m]=dict(n=len(rows),success=sum(r['success'] for r in rows),tcp_mm=float(np.mean([r['tcp_rmse_mm'] for r in rows])),
                joint_rad=float(np.mean([r['joint_rmse_rad'] for r in rows])),final_tcp_mm=float(np.mean([r['final_tcp_mm'] for r in rows])),
                min_gap_mm=min(r['minimum_clearance_mm'] for r in rows),violations=sum(r['safety_violation_steps'] for r in rows),contacts=sum(r['prohibited_contacts'] for r in rows),
                residual_squared_rate=float(np.mean([r['residual_squared_rate'] for r in rows])),saturation=sum(r['torque_saturation_samples'] for r in rows))
    base=metrics['test']['Integral20']['tcp_mm'];improvements={m:(base-metrics['test'][m]['tcp_mm'])/base for m in methods[4:]}
    # Frozen before any final dynamics; NOT significance or tuning.
    both_B_measurable=all(improvements[f'B_seed{s}']>=.01 and metrics['test'][f'B_seed{s}']['violations']==metrics['test'][f'B_seed{s}']['contacts']==0 and metrics['test'][f'B_seed{s}']['success']==6 for s in (42,123))
    verdict=('两种子积分残差PPO均有至少1%的平均误差改善，有限Pilot观察到小幅额外收益；仍须结合各工况、名义精度与平滑代价判断，不能称为统计显著或普遍优势。' if both_B_measurable else '未获得跨两个训练种子、达到1%描述性效果门槛且保持安全的强积分基线额外收益证据。建议结束当前算法扩展，整理为求职作品集；本轮后不自动扩大训练。')
    acceptance=dict(passed=True,tests=tests,historical_files_verified=len(history),historical_changed=False,interface=interface,
        common_gate_passed=gate['all_four_passed'],training_decisions=analysis['total_training_decisions'],training_physics=analysis['training_physics'],metrics=metrics,
        relative_tcp_improvement_over_integral20=improvements,cross_seed_integral_ppo_effect_at_least_one_percent=both_B_measurable,conclusion=verdict,
        videos_validated=True,stage_paused=True,statistics='descriptive only; no significance claim')
    # Avoid duplicating large raw interface records in public summary.
    acceptance['interface']=dict(passed=True,action_max_abs_error=interface['action_max_abs_error'],all_four_trajectories_bitwise_equal=True,exact_reload=True)
    write_json(ROOT/'results/phase3dc_acceptance.json',acceptance)
    budget=4096 if gate['all_four_passed'] else 2048
    initial=Path(read(ROOT/'results/phase3dc_training_initial_latest.json')['run'])
    watcher=f'''```powershell
Set-Location D:\\mujoco_learning\\panda_safe_residual_rl
$py = 'D:\\miniconda\\envs\\panda_safe_rl\\python.exe'
Start-Process (Resolve-Path 'assets/videos/phase3dc_actual_training.mp4')
Start-Process (Resolve-Path 'assets/videos/phase3dc_policy_comparison.mp4')
# 回放实际随机训练记录，不训练、不重新predict
& $py scripts/watch_ppo_training.py --mode actual --run '{initial / 'A_seed42'}' --episode 0 --paused
# 本轮实际验收的无窗口导出命令
& $py scripts/watch_ppo_training.py --mode actual --run '{initial / 'A_seed42'}' --episode 0 --headless-export
```'''
    design_doc=f'''# Phase 3D-C 策略初始化与动作设计

本轮是A=BC85初始化PPO、B=Integral20+小幅PPO的有界对照。冻结配置 `configs/phase3dc.json`；原控制器和模型不改动。初始化/安全测试通过前未训练PPO。

## A：保留BC的确定性策略语义

历史最佳BC85为85→64 Tanh→64 Tanh→7 Tanh，输入均值/标准差来自3D-B训练集，标准差下限0.01。原checkpoint、输入顺序、归一化与哈希均冻结。新增 `BoundedMeanPolicy` 将两个隐藏层及输出线性层完整复制到SB3，并明确以Tanh(linear)作为高斯**均值**。分布为未作Tanh变量变换的七维高斯，初始std=0.03；随机样本到环境边界才裁剪[-1,1]，随后乘0.002 rad。不能将它称为tanh-squashed Gaussian，也不能漏记裁剪前动作。

SB3的get_distribution/predict_values会绕过普通extract_features覆写；第一轮审计正是发现这一点。现已分别保证采样forward、evaluate_actions、确定性predict和predict_values使用同一固定归一化。未修改奖励或通过加宽容差掩盖问题。

8752条历史观测批量动作最大差 `{interface['action_max_abs_error']:.12g}`，保存重载预测完全一致；在两条新训练路径的增益/质量失配下，初始化A与原BC85的完整q_actual/ctrl轨迹逐步完全相同。容差预先设为2e-6动作/关节、2e-6 m TCP，但实际动力学差为0。

## B：Integral20加额外修正

原Integral20因果积分规则不变：0.3/s误差积分、±0.002 rad裁剪、原抗饱和、保护保持清零。每20Hz决策取当前可观测积分目标；RL七维动作映射±0.0002 rad，合成目标clip(integral20 + correction,±0.002)。随后仍经0.01 rad/s残差速率、0.35 rad/s总位置指令速率、实际关节/原native力矩范围及500Hz安全监控。名义补偿模型不读取真实质量/Kp。

原Integral20先将归一化请求转成float32，再映射到rad。首轮零动作比较发现约5.4e-11的差异后，恢复这一确切顺序，而不是把近似相同冒充逐步一致。修正后两种失配条件的零RL输出与原Integral20完整q/ctrl记录**bitwise相同**。正负满幅额外修正均安全完成，非零有效修正比例1.0；正式训练也另记合成裁剪比例，未扩大动作幅度。

## 环境与线程所有权

85维顺序与BC一致，追加积分由学生自己的观测误差在线算出；无真实物理参数或未来参考输入。3D-B已揭示85维记忆与教师标签直接对应，这一阶段不把可计算积分规则称为原创算法。

物理0.002秒/500Hz、RL20Hz，每动作25积分步。GatedShadow在物理线程写积分之前等待预算，避免主线程读取时积分已悄然前进一拍。观测/积分快照在同一锁内复制，渲染仍由物理所有者线程读取live状态；GUI不写MjData。所有真实执行通过ctrl和mj_step，qpos仅reset初始化。

B观测中的上一动作与已施加残差仍采用**总归一化补偿**语义，与A的原BC特征保持一致；Actor原始修正独立记录。两者奖励R1都作用于实际总补偿及其变化，而不是只惩罚B的小修正，保持控制代价定义一致。

## 新任务划分

两条新训练路径(dc_train_sphere/cylinder)、一条验证路径(dc_validation_box)、两条测试路径(dc_final_cylinder/sphere)。真实障碍位置、起终构型不同；新RRT预算种子521/719，均首次521成功并经0.005 rad复核。规划间隙20mm，执行门槛5mm；旧窄间隙案例未用于随机探索。全部配置、场景几何与模型签名见 `{prepared().relative_to(ROOT)}`。

训练覆盖nominal、Kp×0.93、质量惯量×1.07、组合(0.95/1.05/1.10)。按回合与训练seed轮换新条件和路径，同一seed下A/B匹配，物理seed=训练seed×10000+回合序号。验证：nominal、Kp×0.90、质量惯量×1.10，seed7201。最终：Kp×0.86、质量惯量×1.14、组合(0.89/1.11/1.12)，seed8201。测试仅在最终预算、checkpoint及协议冻结后开启，没有把旧已看过任务改名成新测试。

## 共同奖励、预算和门禁

沿用 `src/experiments/phase3c.py::reward_r1`：关节/TCP/速度误差、总补偿及平滑代价、时间/进度、成功2、失败-20、接触额外-40；误差项按真实物理区间/固定轨迹时长归一化。完整各分量与未折扣/GAE/returns保留。原稳定到达标准未改变，无法靠提前保护停止获得成功奖励。

两者Actor/Critic各64×64 Tanh、同一固定BC归一化、Adam学习率1e-4、n_steps256、batch64、n_epochs5、gamma0.995、GAE0.95、clip0.1、vf0.5、最大梯度范数0.5。各seed42/123，先各2048决策；没有最佳种子或最佳checkpoint筛选。

共同门禁在训练前冻结：训练无安全/接触违规，验证所有任务到达且安全，每组至少2个完整失配回合，所有更新Actor/Critic非零且损失有限，有效动作比例≥0.1，验证平均TCP相对各自初始BC/Integral20退化不超过5%。**四组全过才共同到4096，否则全部停在2048**。本轮共同门禁结果 `{gate['all_four_passed']}`，最终每组{budget}决策。分阶段停止时保留未完成回合的保护保持/物理成本，继续时保存优化器/RNG、从新回合恢复；不是把预算截断计成任务失败或完整成功。

公平性限制：A复制BC初始权重，B初始零修正且隐藏层随机；两者网络容量和预算相同，但A总动作范围0.002 rad、B额外范围0.0002 rad，不能把归一化探索量直接比较为相同物理作用。B仍受总补偿饱和限制。Critic未BC预训练；两训练种子不足以做显著性结论。

## 观察方式与实测入口

{watcher}

P/空格暂停，N/B逐决策，+/-调速，Esc/Q退出。Actual Training视频只取各组首个真实随机训练回合（预先固定，未按成功挑选），更新版本可在同一回合中改变；奖励显示的是上一个已完成决策，当前奖励需物理区间结束才可得。独立策略对比另标Baseline Comparison/Checkpoint Evaluation。

兼容性边界：旧watch_ppo_training.py的 `--mode evaluation --checkpoint` 仍绑定旧78维环境，不能直接加载本轮85维PPO。上面的 `--mode actual` 记录回放已实测兼容；新的85维完整动力学评价使用本轮evaluate_phase3dc.py，或直接观看已经保存的策略对比视频。未修改旧播放器的历史评价实现。

实际运行的实验命令（会新建目录并消耗预算，本阶段已完成无需重复）：

```powershell
& $py scripts/prepare_phase3dc.py
& $py scripts/audit_phase3dc.py
& $py scripts/evaluate_phase3dc.py --stage baseline
& $py scripts/train_phase3dc.py --stage initial
& $py scripts/evaluate_phase3dc.py --stage gate
'''
    if gate['all_four_passed']:design_doc+='& $py scripts/train_phase3dc.py --stage continue\n& $py scripts/evaluate_phase3dc.py --stage post\n'
    design_doc+='''& $py scripts/evaluate_phase3dc.py --stage final
& $py scripts/analyze_phase3dc.py
& $py scripts/record_phase3dc.py
& $py -m pytest -q tests/test_phase3dc.py -k 'not frozen_final_opening' --basetemp=results/runs/phase3dc_pytest_acceptance_20261023_c --junitxml=results/phase3dc_new_tests_pre_final.xml
& $py -m pytest -q tests/test_phase3dc.py -k frozen_final_opening --basetemp=results/runs/phase3dc_pytest_acceptance_20261023_d --junitxml=results/phase3dc_new_tests_final_only.xml
```

原209项测试已在新增测试收集前完整重跑，日志/XML独立保留。首次本地多进程遭Windows沙箱管道限制，保留失败日志，随后在允许本地工作进程的执行方式下完成；未改仿真、未安装包。GUI按键本轮未重新人工逐键验收；视频解码、动作/状态一致性及无窗口回放命令分别有实际验收记录。
'''
    (ROOT/'docs/29_Phase3D_C策略初始化与动作设计.md').write_text(design_doc,encoding='utf-8')
    table='| 方法 | 到达 | TCP轨迹RMSE mm | 关节RMSE rad | 最终TCP mm | 最小间隙 mm | 违规/禁止接触 |\n|---|---:|---:|---:|---:|---:|---:|\n'
    for m in methods:
        r=metrics['test'][m];table+=f"| {m} | {r['success']}/{r['n']} | {r['tcp_mm']:.6f} | {r['joint_rad']:.8f} | {r['final_tcp_mm']:.6f} | {r['min_gap_mm']:.5f} | {r['violations']}/{r['contacts']} |\n"
    gatetable='| 方法/seed | 验证初始TCP mm | 2048后 mm | 门禁 | 未通过项 |\n|---|---:|---:|---|---|\n'
    for g in gate['groups']:gatetable+=f"| {g['method']}/{g['seed']} | {g['initial_tcp_mm']:.6f} | {g['validation_tcp_mm']:.6f} | {g['passed']} | {', '.join(k for k,v in g['checks'].items() if not v) or '无'} |\n"
    rows='|阶段|方法/seed|决策|真实物理步|完整回合/成功|预算尾段|Actor变化L2|Critic变化L2|\n|---|---|---:|---:|---:|---:|---:|---:|\n'
    for r in tr:rows+=f"| {r['stage']} | {r['method']}/{r['seed']} | {r['decisions']} | {r['physics']} | {r['full_episodes']}/{r['successful_full_episodes']} | {r['budget_truncated_episodes']} | {r['actor_delta']:.6f} | {r['critic_delta']:.6f} |\n"
    benefits='\n'.join(f'- {m}：相对Integral20平均TCP轨迹RMSE变化 `{improvements[m]*100:+.3f}%`（正数=改善）。' for m in methods[4:])
    bc_changes={m:(metrics['test']['BC85']['tcp_mm']-metrics['test'][m]['tcp_mm'])/metrics['test']['BC85']['tcp_mm']*100 for m in ('A_seed42','A_seed123')}
    nominal='\n'.join(f"- {m} 名义验证TCP RMSE：{next(r['tcp_rmse_mm'] for r in ev if r['method']==m and r['condition']=='nominal' and r['stage']==('baseline' if m in methods[:4] else 'post' if gate['all_four_passed'] else 'gate')):.6f} mm。" for m in methods)
    report=f'''# Phase 3D-C 有限训练验收报告

{verdict}

## 已执行与冻结检查

历史{len(history)}文件SHA256复核无变化，包含旧BC模型、归一化参数、209项原测试与历阶段结果。A/B接口均通过两种模型失配下逐步等价性验收后才训练；BC85批量输出最大误差1.1920929e-7，确定性实际动力学及B零修正实际动力学均逐步完全一致。两项实现问题（SB3预测归一化入口及Integral20的float32转换顺序）先修复再验收，失败日志保留。

完成两方法×两种子，最终每组**{budget} PPO决策**。累计训练{analysis['total_training_decisions']}决策、**{analysis['training_physics']}真实mj_step**，包含reset预保持、每个决策区间和预算清理保持；训练/优化与评价成本分开，不能把一个PPO决策当一个物理步。没有100k训练。

{rows}

完整回合不包含预算最后的保护截断；预算尾段的日志、物理步和失败原因rl_external_truncation保留，不计入完成率。真正安全违规和禁止接触另按物理记录统计。

## 2048共同门禁

{gatetable}

共同继续条件 `{gate['all_four_passed']}`。{'四组共同继续并仅到4096，没有给单一方法额外预算。' if gate['all_four_passed'] else '四组全部停在2048；4096继续训练未执行，未选择性追加某个方法或种子预算。'} 门禁定义在看训练/最终结果前已写入configs/phase3dc.json：安全、有效动作、实际Actor/Critic更新、完整失配经验与最多5%平均验证精度退化，不能看最终测试后放宽。

## 独立最终评价

两种新几何×三种未见失配×八种方法/策略，共48次完整执行；包含全部种子、成功及失败，不与旧测试集混算。下表对六工况等权取平均，主RMSE为trajectory+settle；任务成功沿用关节0.02rad/TCP10mm/速度0.02rad/s稳定0.5秒及全程5mm/禁止接触检查，不用成功率替代亚毫米精度比较。

{table}

{benefits}

最终checkpoint的名义验证（与2048门禁分开保存，未据此调参/挑选）：

{nominal}

完整每工况、执行时间、速度/加速度、力矩饱和、残差平滑性见 `results/phase3dc_controller_evaluation.csv`，没有只取成功回合。验证名义工况也保留，不能用失配平均改善遮盖名义精度退化。

## 研究问题

1. **BC初始化是否减少初始退化？** 初始化本身无退化，已用真实轨迹证明；训练后变化见门禁与各seed评价。本轮没有同预算冷启动PPO组，不能据此定量声称优于冷启动。
2. **是否只恢复BC—积分差距？** A_seed42相对BC85变化{bc_changes['A_seed42']:+.3f}%、A_seed123为{bc_changes['A_seed123']:+.3f}%（正为改善）。两个种子没有稳定缩小与强积分基线的差距，更没有超过它。不能把单一种子略优于BC称为RL优势。
3. **B是否有积分外修正？** 实际记录积分目标、Actor修正、合成目标、施加残差和ctrl。续训阶段B随机修正目标RMS约6.1微rad，目标有效/请求幅度比为1.0，未被总限幅吞掉；但非零并不等于有效收益。幅度/合成裁剪/有效比例见action_audit.csv。可能存在强基线下学习信号小、有限预算与小探索幅度的影响，但本轮未做原因消融，也没有看过最终结果后扩大幅度。
4. **是否超过传统积分？** {verdict} 这里1%是在首次最终动力学执行前单独冻结的描述性效果尺度（见configs/phase3dc_interpretation.json），不参与checkpoint选择/训练继续/调参，也不是显著性检验。
5. **成本是否值得？** 单次传统积分无需在线学习，BC/PPO有额外训练、归档及调试成本；本有界实验不证明在广泛工况值得部署，不能因原计划研究RL就编造传统方法无法解决的问题。
6. **安全、名义精度与平滑代价？** 全量CSV同时记录到达、5mm违规、禁止接触、平滑及饱和；R1惩罚总补偿不是只惩罚B修正，A/B动作幅度与初始化不同仍是公平性限制。两训练种子与六物理任务仅为有限Pilot，无统计显著性结论。

## 训练与动作审计

每256决策真实PPO更新，每次5epoch×4minibatch=20次Adam（实际事件与rollout_npz为准），Actor/Critic变化范数、策略/价值loss、KL/clipfraction、EV、GAE及价值目标逐次保存；全部checkpoint、TensorBoard、训练随机动作/均值/std、奖励分量、真实物理轨迹可追溯。初始std0.03，A的动作是总BC补偿，B是最多±0.0002rad附加修正；两者均受±0.002rad总补偿和原限制，不能把物理探索幅度称为完全相同。

训练平均奖励呈周期锯齿，不能视为精度周期性改善：R1一次成功终止奖励2，落入256决策区间时单独贡献2/256=0.0078125的区间平均奖励，另有冻结的路径/工况轮换。Critic的Value Loss与EV也受这些区间的目标分布影响；网络更新/价值拟合正常不等于控制效果超过积分基线。评价结论来自完整独立动力学回合，而不是奖励峰值。

原209项+新增13项，**{tests}项真实通过、无失败或跳过**。新增覆盖85维/归一化各入口、BC重载/动力学等价、Integral20零动作完全一致、非零通道、Gymnasium真实step、四组相同预算、更新边界、真实物理计数、ctrl/力矩限制、场景隔离、历史冻结、视频与live状态一致。

## 可视化与操作

真实随机训练视频 `assets/videos/phase3dc_actual_training.mp4`，{video['training_frames']}帧、10fps，仅使用四组预先固定首回合的现场RGB；同回合跨256更新时显示正确policy_version。每帧qpos/ctrl与完整integration_states及tracking对应，未重新调用策略生成所谓训练轨迹。

独立对比视频 `assets/videos/phase3dc_policy_comparison.mp4`，{video['comparison_frames']}帧、10fps，八方法在固定第一个最终场景/条件重新真实动力学执行，tracking与正式评价逐项一致；标为Baseline Comparison/Checkpoint Evaluation，不混同Actual Training。每个对比重跑都是额外可视化成本，不当作独立测试任务。

{watcher}

`phase3dc_learning_curves.png`包含实际奖励、Value Loss、EV、KL；`phase3dc_action_tcp_clearance.png`包含Actor修正/合成/施加残差、TCP、真实间隙及更新标记；`phase3dc_final_comparison.png`保留全部测试点。视频元数据/帧核验保存在phase3dc_video独立run中。

## 未完成与阶段边界

未执行：100k长训练、大规模噪声/延迟随机化、更多独立训练种子、实机测试、连续时间安全证明。{'4096阶段已执行，之后未增加预算。' if gate['all_four_passed'] else '共同门禁未过，4096阶段明确未执行。'} 本轮未人工逐键复验Windows GUI；沿用3D-A播放器，实际完成无窗口记录回放/解码及状态核验。失败命令、审计不一致、预算截断均保留，不生成虚构成功视频。

工具/导出问题另存日志：Windows沙箱阻止本地多进程管道及pytest临时目录，随后在允许本地进程的方式完成；首次回放参数误指actual_training而不是组目录，已改正；视频对比中PPO评价日志位于episode_0000而传统基线直接位于运行根目录，已修复读取路径，保留初次部分合成视频与失败日志，并复用已完成的真实执行录像继续导出。这些修复未改策略、训练轨迹或控制参数。

不降低5mm、不改native力矩/关节限位、不覆盖旧数据。仿真与离散碰撞检查不代表实机安全认证。完成后暂停，等待审核；是否整理作品集由上述证据支持，不自动进入下一阶段。
'''
    (ROOT/'docs/30_Phase3D_C有限训练验收报告.md').write_text(report,encoding='utf-8')
    files=[ROOT/'docs/29_Phase3D_C策略初始化与动作设计.md',ROOT/'docs/30_Phase3D_C有限训练验收报告.md',ROOT/'configs/phase3dc.json',ROOT/'configs/phase3dc_interpretation.json']+list((ROOT/'results').glob('phase3dc*.csv'))+list((ROOT/'assets/plots').glob('phase3dc*.png'))+list((ROOT/'assets/videos').glob('phase3dc*.mp4'))
    write_json(ROOT/'results/phase3dc_release_manifest.json',dict(files=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in files],checkpoint_hashes=opening['checkpoints'],acceptance=acceptance))
    print('FINALIZED',tests,analysis['training_physics'],verdict,flush=True)

if __name__=='__main__':main_guard(main)
