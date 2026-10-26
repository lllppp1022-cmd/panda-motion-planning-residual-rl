"""Publish evidence-based Chinese observer instructions and acceptance report."""
from _bootstrap import ROOT,main_guard
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scripts.phase3c_common import read,rows
from src.common import write_json,sha


def table(headers,data):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in data)


def main():
    v=read(ROOT/'results/phase3d_recording_validation.json');e=v['observer_equivalence'];assert v['passed'] and not e['on']['record_errors']
    training=Path(v['training_run']);on=Path(e['on']['run']);off=Path(e['off']['run']);history=Path(v['historical_audit_run'])
    original=ROOT/'results/runs/phase3d_observation_1792740617259955400';old=read(original/'equivalence.json')
    # Existing evaluations remain genuine: verify each policy version is identical to the fixed recorder run.
    evrun=Path(read(ROOT/'results/phase3d_evaluation_latest.json')['run']);ev=read(evrun/'complete.json')['results']
    for old_event,new_event in zip(old['on']['updates'],e['on']['updates']):
        assert old_event['actor_sha256']==new_event['actor_sha256'] and old_event['critic_sha256']==new_event['critic_sha256']
    assert old['on']['initial_actor_sha256']==e['on']['initial_actor_sha256'] and old['on']['initial_critic_sha256']==e['on']['initial_critic_sha256']
    # Current-policy independent evaluation plot, explicitly distinct from actual stochastic training.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True);points=[next(r for r in ev if r['method']==f'update{i:02d}') for i in range(5)]
    ax.plot(range(5),[r['tcp_rmse_m']*1000 for r in points],'o-',label='Deterministic PPO checkpoints')
    for method in ('M3','Integral'):ax.axhline(next(r['tcp_rmse_m']*1000 for r in ev if r['method']==method),ls='--',label=method)
    ax.set(title='CHECKPOINT EVALUATION / NOT TRAINING: same start, gain90, independent seed9001',xlabel='Actual policy update version',ylabel='TCP trajectory RMSE (mm)');ax.legend();ax.grid(alpha=.2);fig.savefig(ROOT/'assets/plots/phase3d_independent_evaluation.png',dpi=150);plt.close(fig)
    suites=ET.Element('testsuites');counts={}
    for name,expected in [('historical',186),('new',11)]:
        root=ET.parse(ROOT/f'results/phase3d_{name}_tests.xml').getroot();selected=[root] if root.tag=='testsuite' else list(root)
        count=sum(int(x.get('tests',0)) for x in selected);assert count==expected
        assert all(int(x.get('errors',0))==int(x.get('failures',0))==int(x.get('skipped',0))==0 for x in selected)
        for s in selected:suites.append(s)
        counts[name]=count
    ET.ElementTree(suites).write(ROOT/'results/phase3d_tests.xml',encoding='utf-8',xml_declaration=True)
    (ROOT/'results/phase3d_tests.log').write_text('\n\n'.join((ROOT/f'results/phase3d_{name}_tests.log').read_text(encoding='utf-8-sig') for name in ('historical','new')),encoding='utf-8')
    hist=read(history/'history_sha256.json');assert all(sha(ROOT/k)==digest for k,digest in hist.items())
    fixed_preview=read(Path(read(ROOT/'results/phase3d_preview_fix_latest.json')['run'])/'complete.json');assert fixed_preview['passed']
    base=ROOT/'results/runs/phase3c_training_continue_1792662132222765100'
    guide=f'''# Phase3D-A：真实PPO交互与训练回放教程

日期：2026-10-23。Python：`D:\\miniconda\\envs\\panda_safe_rl\\python.exe`。本教程的演示训练已经结束；播放已有文件不会训练。旧Phase3B/3C数据和代码均保持只读。

## 1. 窗口中的机械臂究竟在做什么

M3沿冻结并细化碰撞复核的RRT路径给出重力与速度前馈，PPO只学习七维小位置补偿。机械臂的大幅运动主要来自参考路径；随机探索通常只有约0.0002rad量级，不应期待机械臂随机乱甩。

reset先执行750个真实物理步稳定保持，然后Actor读取78维传感/参考/历史观察。Gaussian Actor产生7维均值μ和标准差σ，实际随机动作来自μ+σε；训练保留随机采样，独立评价才使用确定性动作。真实质量、惯量和执行器增益未输入Actor。

归一化动作先裁剪到[-1,1]，乘±0.002rad上限；再用0.01rad/s残差变化率限制，每个0.002s物理步最多改变0.00002rad。残差加到M3基础目标后，继续经过原关节限位、0.35rad/s总指令限速及原生执行器限制。ctrl是位置目标，实际q由mj_step积分得到，绝不是把目标写进live qpos。

## 2. 何时得到奖励、何时更新网络

一次PPO决策通常执行25个物理步，即0.05s；成功终止可能少于25步，保护停止可能在当前transition中继续记录保持。reset和预算退出保持也是真实积分，但不算新增PPO决策。准确计数看complete.json，而非简单地把决策数乘25。

R1沿用冻结的关节/TCP/速度误差、动作幅度/平滑、时间、进度与成功/失败/接触分量。决策奖励只有在该动作区间执行后才生成。实时窗口显示“Last completed decision reward”；回放视频显示已经执行过的区间真实奖励，不能把尚未执行的未来奖励说成实时已知。

每256个PPO决策收集完rollout，才用GAE与价值目标更新Actor/Critic。每次5个epoch×4个minibatch=20个Adam步。本轮1024决策有4次更新、80个Adam步；不是每一个动作或物理步都训练网络。episode约271个决策，因此会跨256步更新边界。回放按每个决策关联版本，不能给整个episode统一贴一个checkpoint。

## 3. 三种数据来源

| 模式 | 行为 | 标记 |
| --- | --- | --- |
| A Actual Training | 播放训练时真实随机动作、状态和奖励 | 历史3C为LOG RECONSTRUCTION；本轮有TRAINING-SITE RGB |
| B Checkpoint Evaluation | 独立重置后执行指定checkpoint的确定性策略 | Checkpoint Evaluation / deterministic |
| C Baseline Comparison | 相同条件比较M3、受限积分及PPO | Baseline Comparison |

Panda工程没有BC控制器，本轮不会虚构BC对照。B/C的轨迹和图不能当作Actual Training。B/C也支持`--replay-only`，仅播放已经真实执行过的独立评价日志。

## 4. 历史训练能恢复什么

3C保存了七关节q/qvel/ctrl、TCP、时间、参考、动作、残差、奖励、间隙、结束事件、rollout、checkpoint与TensorBoard，但没有完整夹爪状态、全部MuJoCo积分状态或现场RGB。程序用固定场景和七关节日志在scratch MjData重建可视机械臂，夹爪取明确的0.04m。TCP重建误差已核验小于1e-12m；这不等于完整仿真状态或接触历史精确恢复。

新录制在物理owner线程、实际积分后获取RGB，并保存9维qpos/qvel、8维ctrl、时间、mocap和官方mjSTATE_INTEGRATION数组。它足够回放这些可视状态，但不包含Python控制器缓存或所有随机数状态，不能宣称可以从任意帧无缝恢复整个训练进程。

## 5. 播放历史实际训练

下面的历史窗口命令已实际运行：

```powershell
cd D:\\mujoco_learning\\panda_safe_residual_rl
& D:\\miniconda\\envs\\panda_safe_rl\\python.exe scripts/watch_ppo_training.py --run {base / 'R1_seed42'} --episode 0 --paused
```

也可把目录换成R0_seed42、R0_seed123、R1_seed123，episode可选0～7。指定父run目录时用`--group R0_seed42`。想查看更新边界，可在这个episode用`--decision 2304`或`--decision 2305`（对应v8/v9）。CLI的决策选择和索引有自动测试，未声称全部桌面按键都由自动输入验收。

P/空格暂停或继续；N/B跳到下/上一个决策末状态；左右方向键逐物理步；+/-调整0.125～8倍速；V切换RGB与状态重建；Q/Esc或窗口X正常关闭。对新录制数据，逐决策操作自动使用完整状态重建，避免10fps的RGB冒充完整20Hz动作帧。RGB模式的时间/注释则严格吸附到对应录制帧。

## 6. 播放本轮实际现场RGB

```powershell
cd D:\\mujoco_learning\\panda_safe_residual_rl
& D:\\miniconda\\envs\\panda_safe_rl\\python.exe scripts/watch_ppo_training.py --run {on} --episode 0 --paused
```

早期/中期/后期为episode0/1/2，episode3是预算中断的未完成轨迹。本轮没有真正的安全违规、禁止接触或完整训练任务失败，不能人为制造“失败视频”。合并影片为`assets/videos/phase3d_actual_training.mp4`；579帧、10fps、57.9s，原始MP4与每帧CSV/JSON留在独立run目录。MP4有损压缩不允许声称像素逐位相等，物理状态、动作与日志关联已逐帧严格核对。

## 7. 新训练现场观察（这些命令会启动新短实验）

默认不启用GPU渲染。新入口复用现有DiagnosticEnv/PilotPPO及控制器，不修改冻结的3B/3C入口。无观察：不加下面两个开关；只录制：仅`--record-training-rollouts`；同步现场RGB：加`--render-training`。这是允许的物理线程内RGB录制+独立窗口方案，未实现GUI线程直接绑定live MjData。

```powershell
cd D:\\mujoco_learning\\panda_safe_residual_rl
& D:\\miniconda\\envs\\panda_safe_rl\\python.exe scripts/train_phase3d_observed.py --compare-observer --decisions 1024 --reward R1 --seed 42 --render-training --record-training-rollouts --record-every-n-episodes 1 --max-recorded-episodes 4
```

这是本轮实际运行过的开/关对照命令，分别1024决策。修复预览后仅额外重跑一次开启录制的1024决策，使用下面实际命令复用原关闭对照，不偷偷重训关闭组：

```powershell
& D:\\miniconda\\envs\\panda_safe_rl\\python.exe scripts/train_phase3d_observed.py --reference-off {off} --decisions 1024 --reward R1 --seed 42 --render-training --record-training-rollouts --record-every-n-episodes 1 --max-recorded-episodes 4
```

现场窗口的暂停只暂停显示，X/Q仅关闭显示，PPO和配置好的录制继续。红色ABORT TRAINING按钮才写入单独中止请求，并由训练回调在决策边界处理、受限保持及归档。没有GUI线程读取或修改live MjData，也没有额外step/reset。所有预算最高1024决策，不支持100k参数。

## 8. 查看每次更新与动作图

训练终端每256决策打印COLLECTING EXPERIENCE / UPDATING ACTOR / CRITIC。原始更新事件位于`{on / 'ppo_update_events.json'}`；每次checkpoint、Actor/Critic变化范数、策略/价值loss、KL、EV、GAE与价值目标分位数、Actor均值/标准差、当前episode累计奖励均可查看。原始采样观察、动作和真实施加残差分别位于observation_sequences.npz、training_samples.csv及actual_training/episode_*/control_signals.csv。

图位于assets/plots：phase3d_ppo_update_timeline.png、phase3d_action_vs_applied.png、phase3d_tcp_error.png、phase3d_reward_components.png、phase3d_independent_evaluation.png。最后一张明确是独立评价；不会拿它冒充训练轨迹。PPO更新很短，GUI可能来不及显示每一瞬间，完整事件和时间线才是每次更新的证据。

## 9. B/C模式与同起点比较

本轮已用7次真实独立动力学执行比较v0～v4和M3/积分，同一训练几何gain90、独立物理seed9001。旧预览录制与修复版各版本Actor/Critic哈希相同，复用这批真实评价，不再扩大训练。

以下回放/导出命令已实际执行（删去`--headless-export`即可打开窗口）：

```powershell
& D:\\miniconda\\envs\\panda_safe_rl\\python.exe scripts/watch_ppo_training.py --mode evaluation --replay-only --run {evrun / 'update02/episode_0000'} --headless-export
& D:\\miniconda\\envs\\panda_safe_rl\\python.exe scripts/watch_ppo_training.py --mode comparison --replay-only --run {evrun / 'M3'} --other-run {evrun / 'Integral'} --other-run {evrun / 'update04/episode_0000'} --headless-export
```

另支持`--mode evaluation --checkpoint <zip> --split train --condition gain90`执行新的独立确定性评价后观看；这会增加评价交互，不增加PPO训练。VS Code打开项目后，在集成PowerShell执行同样命令即可；无需通过Run Code使用其他Python解释器。

## 10. 如何核对观察器没有改变学习

查看results/phase3d_recording_validation.json：开/关实际采样动作、观测、奖励、结束类型、tracking、控制信号（排除墙钟计算耗时）、全部Actor/Critic参数和Adam状态逐位一致，四次更新时点完全相同。ZIP包含路径等元数据，可不同；不能把ZIP字节不同当作参数不同。修复后录制器再次1024决策通过同一严格比较。

初版现场预览有Windows文件共享冲突，原始RGB和状态未丢，错误与第一次对照完整保留。最终预览使用不可变PNG文件路径与有界JSON替换重试，32步零动作接口和最终1024决策均无录制错误。任何录制异常只禁用观察器并记错，父环境仍推进原物理流程；故障注入测试实际验证了这一点。

验证命令已实际运行：`python scripts/audit_phase3d_history.py`、`python scripts/run_phase3d_evaluations.py`、`python scripts/validate_phase3d_recordings.py`，以及原186项和新增11项pytest。全部操作结束后暂停，不会自动长训练。
'''
    (ROOT/'docs/25_Phase3D_A训练可视化使用教程.md').write_text(guide,encoding='utf-8')
    report='# Phase3D-A训练观察验收报告\n\n日期2026-10-23。任务是观察和诊断，不以提高PPO指标为目标。完成后暂停。\n\n'
    report+=f'## 实际执行范围\n\n先审计4组历史继续训练run、32个episode；冻结{len(hist)}个历史文件SHA256，结束时全部一致。历史仅可重建七关节可视状态，没有历史现场RGB。\n\n'
    report+=table(['运行','PPO决策','物理积分','reset','cleanup','更新/Adam','录制问题'],[
        ['原关闭观察器',e['off']['decisions'],e['off']['physics_total'],e['off']['reset_physics'],e['off']['cleanup_physics'],'4/80',0],
        ['初版开启记录',old['on']['decisions'],old['on']['physics_total'],old['on']['reset_physics'],old['on']['cleanup_physics'],'4/80',f"{len(old['on']['record_errors'])}条预览共享冲突"],
        ['修复后开启记录',e['on']['decisions'],e['on']['physics_total'],e['on']['reset_physics'],e['on']['cleanup_physics'],'4/80',len(e['on']['record_errors'])]])
    report+=f"\n合计3072个PPO决策、{old['off']['physics_total']+old['on']['physics_total']+e['on']['physics_total']}个训练相关真实积分；每次上限1024。接口探针/自动测试与7次独立确定性评价另存，不混入训练决策数。并非100k训练。最终修复版：`{on}`。关闭组复用：`{off}`。初版及失败诊断：`{original}`。\n\n"
    report+='## 核心等价性与视频证据\n\n采样、观察、奖励、terminated/truncated、tracking、实际控制信号（剔除algorithm_s墙钟耗时）、全部网络参数及Adam状态逐位一致；256/512/768/1024更新点一致。Actor/Critic每次均真实更新。ZIP路径元数据不同，验收采用每个参数及优化器张量一致，而非强迫ZIP哈希一致。\n\n'
    report+=table(['Episode','视频帧','时长(s)','真实关节位移(rad)','策略版本','结局'],[[r['episode'],r['frames'],r['duration_s'],f"{r['joint_displacement_rad']:.6f}",r['policy_versions'],r['outcome']] for r in v['video']['episodes']])
    report+=f"\n最终影片{v['video']['total_frames']}帧、{v['video']['duration_s']}秒；MP4已解码验收，状态/动作/ctrl/TCP/间隙逐帧对应原始CSV和完整状态NPZ，RGB在物理owner线程从同一个live MjData取得。新状态保存全部9维qpos/qvel和8维ctrl；旧历史夹爪缺失不伪造精确重建。影片只是对现场RGB加已记录奖励字幕，没有重新调用策略生成替代轨迹。\n\n"
    report+='本轮3个真实完整回合均成功，1个因预算结束而未完成；未出现真正任务失败、安全违规或禁止接触。预算退出的保护保持已保存，不制作虚假失败样例。\n\n'
    report+='## 五个checkpoint与传统基线的独立评价\n\n同一gain90任务、同起点/终点、物理seed9001。它们是Checkpoint Evaluation / Baseline Comparison，绝不混作Actual Training。初版和修复版各更新权重哈希相同，复用真实评价，未重复优化。\n\n'
    report+=table(['方法/版本','TCP轨迹RMSE(mm)','完成','5mm违规步'],[[r['method'],f"{r['tcp_rmse_m']*1000:.6f}",r['success'],r['safety_violation_steps']] for r in sorted(ev,key=lambda r:r['method'])])
    report+='\n这些短实验不宣称统计显著或PPO优于积分补偿，也不修改Phase3C结论。\n\n'
    report+='## 工程问题、修复与真实验收边界\n\n'
    report+=f'- 初始2步接口探针发现重复episode显示字段，已修复，原目录保留。\n- 初次1024录制有{len(old["on"]["record_errors"])}条Windows预览PNG共享冲突。原始MP4/完整物理状态仍完整，训练等价性通过；并未把预览错误改写成成功日志。修复为不可变帧文件和有界JSON替换重试；32步非训练接口及修复后的1024对照均通过。\n- 并行B/C回放曾因旧timestamp目录名冲突启动失败；改用带PID的独立观察目录，失败日志`results/phase3d_mode_c.log`和成功重跑日志`results/phase3d_mode_c_fixed.log`均保留。\n'
    report+='- Windows现场RGB窗口已实际显示并读取截图；历史窗口日志记录了真实暂停、调速、关闭事件。用户将窗口最小化后，自动输入工具报告`user input was detected in this window`，刷新仍受保护；未强行覆盖用户操作。N/B的决策索引和帧对齐有自动测试，但没有宣称完成全部桌面按键注入验收；红色主动中止按钮未在正式对照中点击。\n'
    report+='- 直接绑定live MjData的原生MuJoCo Viewer未实现；采用任务允许的物理线程RGB采集+独立同步窗口。GUI只读文件副本，不访问MjData，不调用step/reset。默认没有高频GPU渲染。\n\n'
    report+=f'原186项和新增11项实际通过，合计197/197，0跳过0失败。合并JUnit/日志位于results/phase3d_tests.xml及results/phase3d_tests.log。新增实际覆盖跨rollout策略版本、完整状态、逐帧动作对应、录制开关等价性、GPU录制无状态副作用、故障隔离、预览修复、历史文件只读及独立评价。\n\n'
    report+='## 交付及暂停\n\n教程：docs/25_Phase3D_A训练可视化使用教程.md。主程序：scripts/watch_ppo_training.py；现场观察训练：scripts/train_phase3d_observed.py。视频：assets/videos/phase3d_actual_training.mp4。动作/更新/TCP/奖励/独立评价五图：assets/plots/phase3d_*.png。完整动作/状态/帧证据索引：results/phase3d_recording_validation.json。具体运行命令和三种来源的区别详见教程。\n\n未执行：100k或更大训练、BC训练、实机实验、严格连续安全认证；没有完整任务失败可供本轮录制。原生live Viewer及全部桌面按键自动注入验收仍是限制。已停止PPO更新，现存显示窗口可Q/Esc/X关闭，仅查看不会启动训练。\n'
    (ROOT/'docs/26_Phase3D_A验收报告.md').write_text(report,encoding='utf-8')
    v.update(tests=dict(passed=197,historical=186,new=11,skipped=0,failures=0),original_observation_run=str(original),
        initial_preview_errors_retained=len(old['on']['record_errors']),fixed_preview_errors=0,
        independent_evaluation_run=str(evrun),all_three_training_runs_decisions=3072,total_training_integrations=old['off']['physics_total']+old['on']['physics_total']+e['on']['physics_total'],
        gui_limitations='Observed desktop RGB and user keyboard event logs; automated N/B desktop injection not fully verified due user-input protection; direct native live viewer not implemented.')
    write_json(ROOT/'results/phase3d_recording_validation.json',v)
    outputs=['docs/25_Phase3D_A训练可视化使用教程.md','docs/26_Phase3D_A验收报告.md','scripts/watch_ppo_training.py','scripts/train_phase3d_observed.py',
        'assets/videos/phase3d_actual_training.mp4','assets/plots/phase3d_action_vs_applied.png','assets/plots/phase3d_ppo_update_timeline.png','assets/plots/phase3d_tcp_error.png',
        'assets/plots/phase3d_reward_components.png','assets/plots/phase3d_independent_evaluation.png','results/phase3d_recording_validation.json']
    write_json(ROOT/'results/phase3d_release_manifest.json',dict(files={p:sha(ROOT/p) for p in outputs},historical_unchanged=True,training_stopped=True))
    print('PHASE3D COMPLETE',on,'tests197',flush=True)


if __name__=='__main__':main_guard(main)
