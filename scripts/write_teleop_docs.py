"""Summarize actual Phase 1.5 artifacts, including unsuccessful development runs."""
from _bootstrap import ROOT, main_guard
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from src.common import write_json, sha


def main():
    validation = json.loads((ROOT / 'results/teleop_recording_validation.json').read_text(encoding='utf-8'))
    run = Path(validation['run'])
    summary = json.loads((run / 'complete.json').read_text(encoding='utf-8'))
    media = summary['video']
    unit = ET.parse(ROOT / 'results/phase1_5_tests.xml').getroot()
    cases = list(unit.iter('testcase'))
    assert len(cases) == 38 and not list(unit.iter('failure')) and not list(unit.iter('error'))
    runs = []
    for p in sorted((ROOT / 'results/runs').glob('teleop_panda_*')):
        path = p / 'complete.json'
        if path.exists():
            s = json.loads(path.read_text(encoding='utf-8'))
            runs.append(dict(run=p.name, status=s['exit_status'], steps=s['physics_steps'],
                             simulation_seconds=s['simulation_seconds'], error=s['error'],
                             video_error=s['video_error'], viewer_closed=s['viewer_closed']))
        else:
            runs.append(dict(run=p.name, status='incomplete_native_close_crash', steps=None,
                             simulation_seconds=None, error='No complete.json; see native access violation logs',
                             video_error=None, viewer_closed=None))
    evidence = dict(acceptance_date_local='2026-10-21', phase='1.5',
                    pytest=dict(passed=len(cases), failed=0, skipped=0),
                    native_default_cli_run='teleop_panda_1792550259629003800',
                    native_close_fix_run='teleop_panda_1792549993249892400',
                    final_recorded_run=run.name, runs=runs,
                    video_sha256={f'panda_keyboard_teleop_20261021.{ext}': sha(ROOT / f'assets/videos/panda_keyboard_teleop_20261021.{ext}') for ext in ('gif','mp4')},
                    safety_scope='Local empty practice scene; no path or collision safety certification',
                    phase2_implemented=False, reinforcement_learning_executed=False)
    write_json(ROOT / 'results/phase1_5_acceptance.json', evidence)
    doc = '''# Panda 实时键盘控制使用教程（Phase 1.5）

本阶段是在同一个 MuJoCo 训练无关的物理环境里，实时用键盘改变目标，再通过原位置伺服执行器与 `mj_step` 积分运动。默认入口没有预置动作序列，也没有动画回放。未实现强化学习、RRT 或姿态控制。

## 1. PowerShell 一键启动

如果当前终端尚未识别 conda，先执行初始化行；已经能用 conda 的终端也可直接从激活行开始。独立环境沿用 Phase 1，不需要重新安装或修改旧 Reacher 环境。

```powershell
. 'D:\\miniconda\\shell\\condabin\\conda-hook.ps1'
conda activate panda_safe_rl
cd D:\\mujoco_learning\\panda_safe_residual_rl
python scripts/teleop_panda.py
```

不依赖 conda 激活的等价启动命令：

```powershell
cd D:\\mujoco_learning\\panda_safe_residual_rl
& 'D:\\miniconda\\envs\\panda_safe_rl\\python.exe' scripts/teleop_panda.py
```

**先单击机械臂窗口取得焦点，并切到英文输入状态。** 本机中文输入法会把字母送入候选框，导致 MuJoCo 收不到字母键。看见拼音候选框时先按 Shift 切到英文。切换方法依输入法设置而定；不要连续盲按，因为英文/中文可能被反复切回。只聚焦 PowerShell 或 VS Code 的终端无法控制 Viewer。

默认从 JOINT 模式、joint1 开始。状态显示在启动它的终端，每秒 4 次，并同步写入该次运行的 `live_status.json`。需要同时查看机器人和状态时，把 PowerShell 与 Viewer 并排。最终版不在窗口中叠加文字，以减少原生关闭期间的线程握手。

## 2. 准确键位

| 按键 | 生效模式 | 行为 |
|---|---|---|
| 数字主键盘 1～7 | 任意 | 选择 joint1～joint7；并不会自动切换控制模式 |
| J / K | JOINT | 当前关节目标减/增 0.02 rad |
| M | 任意 | JOINT 与 XYZ 切换；取消此前远处目标，以当前真实关节位置保持；进入 XYZ 时从当前实际 TCP 位置初始化目标 |
| W / S | XYZ | 世界 X 正/负方向，目标每次 0.005 m |
| A / D | XYZ | 世界 Y 正/负方向，目标每次 0.005 m |
| R / F | XYZ | 世界 Z 正/负方向，目标每次 0.005 m |
| 空格 | 任意 | 优先软停止，丢弃队列中尚待处理的移动指令；关节目标改为当前角度，仍经过限速和动力学伺服 |
| H | 任意 | 限速回到 home 关节目标；不瞬移，也不会重置仿真时间 |
| Esc / Q | 任意 | 请求退出，保存本次日志 |
| 窗口右上角 × | 任意 | 同样先排队退出，再安全关闭原生窗口并保存日志 |

字母键不要求大小写锁定；上表以大写标识。按一次产生一次小步命令，不承诺按住按键会连续运动。XYZ 模式下 J/K 会记录为 wrong_mode 拒绝；JOINT 模式下 W/S/A/D/R/F 也会拒绝。数字键与某些字母是 Viewer 内置显示快捷键，因此程序会恢复几何显示组、可视化选项及渲染 flags，避免切换关节后机器人消失或变成线框。相机的普通旋转、缩放仍可使用。

不要使用 Viewer 的关节滑块、Backspace/reset 或 Ctrl 拖动施力。这些属于另一套 Viewer 操作，不经过本控制器；程序检测到同步时 qpos 或时间被 UI 改动会报错退出，不会覆盖 qpos 去“修复”。请重新启动。正常 teleop 控制没有对 live qpos/qvel 的赋值。

## 3. 建议的首次练习

1. 确认终端显示 JOINT，按 1、K，等待约半秒；再按 J。
2. 分别选择 2～7 后各按一次 K，观察整条机械臂的变化。joint7 主要改变姿态，TCP 位置可能几乎不变，这是正常的。
3. 按 H 等待回 home，再按 M。确认终端显示 XYZ。
4. 依次单击 W、S、A、D、R、F，每次之间等待约半秒至一秒。方向是**世界坐标轴**，不是屏幕的左右上下。
5. 运动中按空格，观察目标变为当前关节角。软停止不是速度瞬间归零，也不是硬件急停；重力下位置伺服仍有静差。
6. 按 H 返回，最后按 Esc，或点击 ×。

## 4. 小范围限制与 IK

配置在 `configs/teleop.yaml`：关节步长默认 0.02 rad、笛卡尔步长 5 mm、最小步长分别 0.0001 rad / 0.0001 m、伺服命令速度 0.20 rad/s。每个 0.002 s 物理步，ctrl 最多改变 0.0004 rad。

本阶段刻意限制在 home 附近每关节 ±0.35 rad，并与真实 MJCF 关节范围取交集。手动关节与 XYZ 指令相对实际角度的目标超前量最多 0.12 rad；超出时拒绝并提示等待伺服。home 返回允许从练习包络边缘回到 home，但仍按 0.20 rad/s 逐步执行。配置不能突破脚本规定的速度、练习包络和步长上限；不提供关闭包络的 CLI。

到达边界会限幅；剩余可调量小于最小步长时拒绝。XYZ 以当前实际关节角作为 DLS 起点，成功后才提交目标关节角；失败、非有限结果、过大的 IK 关节变化都不会提交新目标，旧目标和旧 XYZ 目标保持不变。`ik_status` 表示最近一次求解结果；不代表实时定位误差为零。

已有 DLS 的位置终止阈值为 1 mm，但它是理想静态 FK 的求解误差。实际机器人通过带重力的位置伺服运动，本次通常仍有约 6～7 mm 静差。每次命令目标移动 5 mm，不意味着任意时刻都准确到达该目标。此阶段未调增益或增加重力补偿。

位置 IK 有冗余解，不约束姿态；末端方向正确不代表每个关节路径安全。默认新建 `teleop_practice_scene.xml`，没有额外球/柱障碍物；保留地面、原机器人接触参数和原伺服执行器。没有提供障碍场景 teleop 开关，也没有自动碰撞安全路径规划。原 Phase 1 的障碍接口只是实验性静态查询。

检测到实际外部/自接触、过大单步角变化、速度或跟踪误差时，记录事件、将目标改为当前位置并联锁后续移动。检查现场后可按空格确认；接触仍存在会再次联锁。这是响应式调试保护，不是碰撞预防或安全认证。

## 5. 日志与真实交互录像

每次运行自动创建独立 `results/runs/teleop_panda_<纳秒编号>/`，不会覆盖旧实验。

| 文件 | 内容 |
|---|---|
| interaction_commands.csv | 每条收到的命令，包括接受/拒绝、按键、仿真时间、回调/处理线程、修改前后关节与目标、TCP、IK、限制和接触 |
| physics_states.csv | 50 Hz 实际关节、速度、限速 ctrl、关节目标、TCP、目标、误差、接触及联锁 |
| events.csv | 启动、接触/意外运动告警变化、退出事件 |
| live_status.json | 4 Hz 最新状态，便于旁路观察 |
| run_config.json | 配置、模型哈希、CLI、MuJoCo 版本 |
| complete.json | 退出原因、物理步数、最大实际关节步幅/速度、ctrl 变化、原生关闭请求和录像结果 |

开启录像：

```powershell
cd D:\\mujoco_learning\\panda_safe_residual_rl
python scripts/teleop_panda.py --record
```

只录制同一 live MjData 在实际 `mj_step` 后的 RGB，默认 10 FPS、640×528（含 48 像素信息栏），最多前 90 **仿真秒**。可以改配置中的录制时长。输出 `teleop_actual_interaction.gif/.mp4`、首末 PNG、逐帧时间/关节/原始 RGB 哈希及视频校验 JSON。没有重新调用策略或重新执行一组动作。保存的是实际机器人场景画面，不是包含桌面/鼠标的屏幕录像。视频的时间轴按仿真时间，墙钟操作时间可能更长。

录像可显著增加开销；默认不录制。录制异常单独写入 `video_error`，不会因此终止控制。没有实际运动或过早关闭时不应把录像校验错误当成合格演示。正式数据以 CSV、`complete.json` 和逐帧记录为准。

已验收视频：`assets/videos/panda_keyboard_teleop_20261021.mp4`，也提供同名 GIF。查看本次原始目录和复核命令见验收报告。

## 6. VS Code 与命令行验收

在 VS Code 打开工程文件夹，Python 解释器选择 `D:\\miniconda\\envs\\panda_safe_rl\\python.exe`；打开 PowerShell 终端，执行第 1 节命令。窗口出现后将焦点移到 Viewer，并切英文。VS Code 自身的编辑区快捷键不会被程序接收。

```powershell
python scripts/teleop_panda.py --help
python -m pytest tests -q
python scripts/validate_teleop_recording.py --run results/runs/teleop_panda_1792550069788228500
```

这些入口均已实测；无参数入口通过 Esc 正常退出。`--max-seconds 180` 仅为可选的墙钟超时，不会注入动作。查看 CSV 可用 VS Code 或 Excel，数组字段保存为 JSON 字符串。

## 7. 线程与正常关闭的实现说明

MuJoCo `launch_passive(..., key_callback=...)` 的回调只向线程安全队列提交按键、接收时间和线程号。物理循环负责 IK、限幅、目标更新、ctrl 与唯一的 mj_step；Viewer lock 内操作数据，sync 在锁外。IK 仍使用原独立 scratch MjData，不写执行环境。

本机 MuJoCo 3.16.0 的原生 × 关闭在开发测试中重现了 `Handle.lock()` 附近 access violation。最终实现只对**本进程、唯一指定标题的 Viewer**增加 Windows 关闭消息处理：WM_CLOSE 排队为退出命令，其他消息转交原窗口过程；物理循环释放锁后调用 viewer.close 并等待该窗口销毁。未修改已安装的 MuJoCo、系统设置或其他应用窗口。这个 Windows 生命周期处理已通过真实 × 关闭和默认入口 Esc 复测。

参考：[MuJoCo passive viewer](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer)、[SetWindowLongPtrW](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-setwindowlongptrw)、[WM_CLOSE](https://learn.microsoft.com/en-us/windows/win32/winmsg/wm-close)。未声称这是全部 MuJoCo 版本的通用缺陷。

完成 Phase 1.5 后暂停，等待用户审核。没有自动开展 Phase 2 或强化学习训练。
'''
    (ROOT / 'docs/05_交互控制使用教程.md').write_text(doc, encoding='utf-8')
    rows = '\n'.join(f"| {r['key']} | {'joint'+str(r['selected_joint']) if r['command_type']=='joint_jog' else r['axis']} | {abs(r['actual_delta_q_rad'][r['selected_joint']-1]):.6f} rad |" if r['command_type']=='joint_jog' else f"| {r['key']} | 世界{r['axis']} | {r['signed_axis_movement_mm']:.4f} mm，方向正确 |" for r in validation['measurements'])
    run_table = '\n'.join(f"| {r['run']} | {r['status']} | {r['steps'] if r['steps'] is not None else '未完整保存'} | {r['simulation_seconds'] if r['simulation_seconds'] is not None else '未完整保存'} |" for r in runs)
    report = f'''# Panda Phase 1.5 实际验收报告

验收日期：2026-10-21（Asia/Shanghai）。工程：`D:\\mujoco_learning\\panda_safe_residual_rl`。本阶段只增加键盘控制、日志、真实录像及 Windows 退出处理；没有训练网络或实现 RRT。

## 实现与验收范围

- 新入口 `scripts/teleop_panda.py`，默认无参数即打开原生 MuJoCo Viewer。
- `src/robot/teleop.py` 复用原位置控制器、DLS IK；`configs/teleop.yaml` 管理步长与限速。
- callbacks 仅入队；命令处理线程与回调线程不同；IK 只修改 scratch 状态。
- 七关节键盘控制、XYZ 六方向、soft stop、home、限位/最小步长/非有限拒绝、接触及意外运动联锁、独立 CSV/JSON/视频输出已实现。
- 原 Phase 1 的 18 项测试加本阶段 20 项，**38 passed / 0 failed / 0 skipped**。JUnit：`results/phase1_5_tests.xml`。

## 原生窗口真实交互与录像

最终合格录制目录：`{run.relative_to(ROOT).as_posix()}`。

通过 computer-use 在原生 Viewer 中发送真实键盘事件。没有把直接调用 `TeleopController.command` 的单元测试当作 GUI 验收，也没有用预置 qpos 或动画替代运动。最终录像包含七次 K（joint1～7）、joint7 的 J、M、W/S/A/D/R/F、空格、H 和 × 关闭。

- 实际命令 {summary['commands']} 条（含关闭转成的退出命令）。
- 物理积分 **{summary['physics_steps']} 步 / {summary['simulation_seconds']:.3f} 仿真秒**；步数乘 0.002 s 与仿真时钟一致。
- 正常 `viewer_closed` 退出，原生关闭请求 1，窗口已销毁，进程退出码 0。
- 最大 ctrl 单步变化 **{summary['metrics']['max_ctrl_step_rad']:.7f} rad**，符合 0.20 rad/s × 0.002 s。
- 最大实际关节单步变化 **{summary['metrics']['max_actual_joint_step_rad']:.8f} rad**；最大实际角速度 **{summary['metrics']['max_actual_velocity_rad_s']:.6f} rad/s**。这两项是观测值，不是未来轨迹保证。
- 外部接触最大 {validation['external_contact_max']}，自接触最大 {validation['self_contact_max']}，异常告警步数 {summary['metrics']['alert_steps']}。
- 每条命令执行前后 live qpos 完全相同；命令只改目标，后续积分产生位移。
- {media['raw_frames']} 帧视频与同时间物理日志逐一匹配，最大关节差 **{validation['frame_state_max_q_error_rad']} rad**；原始机器人 RGB 有 {media['raw_unique_frames']} 个不同帧，排除了仅时钟文字变化的假动画。
- GIF 解码 {media['encoded_frames']} 帧、{media['duration_ms']/1000:.1f} s；MP4 解码 {media['mp4_decoded_frames']} 帧；模型关节位移范数最大 {media['joint_displacement_rad']:.6f} rad。

展示文件：

- `assets/videos/panda_keyboard_teleop_20261021.mp4`
- `assets/videos/panda_keyboard_teleop_20261021.gif`
- 原目录 `teleop_joint_tracking.png`、`teleop_xyz_tracking.png`、首末 PNG 和全部 CSV/逐帧 JSON。

### 单次按键的实际位移

从命令前状态，到约 0.65 s 后真实日志，分别检查所选关节或对应世界轴。没有把理想 FK 位移当作实际位移。

| 键 | 控制对象 | 实测位移绝对值或沿要求方向的有符号位移 |
|---|---|---|
{rows}

末端 5 mm 指令的对应轴实际位移约 4.90～5.02 mm。实际定位误差不等于零：本次回 home 后三维误差 {summary['final']['error_mm']:.4f} mm，与 Phase 1 的原伺服重力静差一致。没有为改善数字修改控制增益、奖励或阈值。

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
{run_table}

每个运行的视频异常、最终状态与原生关闭字段详见其 complete.json；总清单 `results/phase1_5_acceptance.json`。无动作的早期录制可能未通过视频运动校验；不作为演示发布。

## 复现与文件校验

```powershell
. 'D:\\miniconda\\shell\\condabin\\conda-hook.ps1'
conda activate panda_safe_rl
cd D:\\mujoco_learning\\panda_safe_residual_rl
python scripts/teleop_panda.py
# 退出后，如需录制自己下一次操作：
python scripts/teleop_panda.py --record
# 完成操作并退出后，再运行测试：
python -m pytest tests -q
python scripts/validate_teleop_recording.py --run results/runs/{run.name}
```

`results/teleop_recording_validation.json` 保存逐命令位移与在运行后审计的源码/数据 SHA256。模型哈希在 run_config.json。视频哈希在 `results/phase1_5_acceptance.json`，逐帧原始 RGB 哈希在原始目录。安装环境沿用 `panda_safe_rl`，未升级或更改任何依赖，未改旧 Reacher 源码、模型、数据与环境；原 Phase 1 实现仍被原有测试覆盖。

## 尚未实现与暂停边界

- 未实现全关节范围自由移动、姿态控制、路径碰撞检查、连续安全保证、自动避障或规划。
- 不支持障碍场景交互开关；默认只是无额外障碍物的小范围练习。
- 未消除原位置伺服的重力静差，软停止也不是硬件急停。
- 原生生命周期保护只在本 Windows 环境验证，未执行 Linux/macOS 或其他 MuJoCo 版本测试。
- 没有 RRT-Connect、残差 RL、PPO/SAC 训练、Sim-to-Real 或算法性能提升实验。

Phase 1.5 完成，等待用户审核，不自动开展 Phase 2。
'''
    (ROOT / 'docs/06_Phase1_5验收报告.md').write_text(report, encoding='utf-8')
    readme = ROOT / 'README.md'
    text = readme.read_text(encoding='utf-8')
    marker = '\n## Phase 1.5：实时键盘控制\n'
    if marker not in text:
        text += marker + '''
已完成真实键盘交互：七关节 J/K、世界 XYZ 六方向、软停止与限速 home、独立日志及实际物理运动录像。**先聚焦 Viewer 并切英文；按 M 进入 XYZ 后才能使用 W/S、A/D、R/F。** 状态显示在启动终端。

```powershell
conda activate panda_safe_rl
cd D:\\mujoco_learning\\panda_safe_residual_rl
python scripts/teleop_panda.py
# 下一次需要录像时：
python scripts/teleop_panda.py --record
```

[交互控制教程](docs/05_交互控制使用教程.md) · [Phase 1.5 实际验收报告](docs/06_Phase1_5验收报告.md) · [真实交互 MP4](assets/videos/panda_keyboard_teleop_20261021.mp4)

![实际键盘输入后的物理运动](assets/videos/panda_keyboard_teleop_20261021.gif)

38 项测试通过；最终原生交互 29,603 物理步，593 帧与实际状态日志一致。默认采用无额外障碍物的小范围练习包络，不构成碰撞安全保证。Phase 1.5 完成后暂停，等待审核，未进入 Phase 2。
'''
        readme.write_text(text, encoding='utf-8')
    print('Wrote docs/05, docs/06, results/phase1_5_acceptance.json and README entry')


if __name__ == '__main__':
    main_guard(main)
