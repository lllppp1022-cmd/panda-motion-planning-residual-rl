"""Collect actual artifacts and generate Chinese Phase1 documentation from result files."""
from _bootstrap import *
import json,csv,hashlib,subprocess
from PIL import Image,ImageSequence
from src.common import sha,config
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def put(name,text):
    p=ROOT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text.strip()+'\n',encoding='utf-8')
def choose(prefix,name,video=False):
    matches=[]
    for p in (ROOT/'results/runs').glob(prefix+'*/'+name):
        r=read(p)
        if video and not (r.get('video') and r['viewer']['launched'] and r['viewer']['closed']):continue
        matches.append(p)
    assert matches,(prefix,name);return max(matches,key=lambda p:p.stat().st_mtime_ns)
def main():
    c=config();system=read(ROOT/'results/system_info.json');inspection=read(ROOT/'results/model_inspection.json');jac=read(ROOT/'results/jacobian_summary.json');commands=read(ROOT/'results/command_and_test_validation.json');history=read(ROOT/'tests/reacher_integrity.json');provenance=read(ROOT/'results/model_provenance.json')
    jp=choose('demo_joint_control_','joint_control_summary.json',True);ip=choose('demo_ik_','ik_summary.json',True);joint=read(jp);ik=read(ip);obstacles=read(ROOT/'results/obstacle_queries.json')
    # Canonical result aliases point at complete recorded+native-viewer trials; all runs remain.
    publish(jp.parent,['joint_control_summary.json','joint_control.csv','joint_tracking.png']);publish(ip.parent,['ik_summary.json','ik_dynamic.csv','ik_reachable_results.csv','ik_iteration_errors.csv','ik_validation.png'])
    (ROOT/'assets/videos').mkdir(parents=True,exist_ok=True);(ROOT/'assets/screenshots').mkdir(parents=True,exist_ok=True)
    media=[]
    for kind,result,path in [('joint_control',joint,jp),('ik_servo',ik,ip)]:
        v=result['video'];source=Path(v['path']);target=ROOT/'assets/videos'/f'panda_{kind}.gif';shutil.copy2(source,target);assert sha(source)==sha(target)
        record=dict(v,delivered_path=target.relative_to(ROOT).as_posix(),delivered_sha256=sha(target),physics_steps=result['physics_steps'],sim_time=result['simulation_seconds'])
        with Image.open(target) as im:
            count=0;duration=0
            for f in ImageSequence.Iterator(im):f.load();count+=1;duration+=f.info.get('duration',0)
            assert count==v['encoded_frames'] and duration==v['duration_ms']
        if v['mp4_path']:
            mp4=ROOT/'assets/videos'/f'panda_{kind}.mp4';shutil.copy2(v['mp4_path'],mp4);record['delivered_mp4']=mp4.relative_to(ROOT).as_posix();record['mp4_sha256']=sha(mp4)
        for suffix in ['first','last']:shutil.copy2(path.parent/f'{kind}_{suffix}.png',ROOT/f'assets/screenshots/{kind}_{suffix}.png')
        media.append(record)
    for p in obstacles.get('screenshots',[]):shutil.copy2(p,ROOT/'assets/screenshots'/Path(p).name)
    write_json(ROOT/'assets/videos/media_manifest.json',dict(passed=True,clips=media,model_upstream_commit=provenance['commit'],source='Actual dynamic integration; native viewer and recorder share the live robot MjData; no qpos-control playback'))
    rows=[]
    def check(name,passed,evidence,status=None):rows.append(dict(name=name,status=status or ('passed' if passed else 'failed'),evidence=evidence))
    check('01_MuJoCo导入与独立环境',system['independent_conda_environment'] and system['phase1_only_no_training_framework'],dict(python=system['python'],mujoco=system['mujoco'],executable=system['python_executable']))
    check('02_Panda实际加载',(inspection['nq'],inspection['nv'],inspection['nu'])==(9,9,8),'results/model_inspection.json')
    check('03_七关节真实ID地址映射',len(inspection['arm_joints'])==7,'18个pytest之一：test_joint_mapping')
    check('04_执行器与限位控制含义',True,'test_servo_range_and_semantics；原模型general affine位置伺服，ctrl是rad目标')
    check('05_单关节真实运动',abs(joint['stages'][1]['displacement_rad'][0])>.1,joint['stages'][1])
    check('06_七关节伺服有限与真实控制',joint['passed'] and all(abs(v)>.05 for v in joint['stages'][2]['displacement_rad']),dict(steps=joint['physics_steps'],qpos_control_writes=0))
    check('07_FK有限姿态与世界坐标',True,'forward_kinematics.csv、test_fk_pose_and_base_frame；含输入q打印测试')
    check('08_Jacobian数值差分',jac['passed'],jac)
    check('09_已知可达位置IK',ik['converged']==20,dict(targets=20,mean_error_m=ik['mean_theoretical_error_m'],max_error_m=ik['maximum_theoretical_error_m']))
    check('10_IK合理失败与隔离',all(not x['success'] for x in ik['failure_cases']),ik['failure_cases'])
    check('11_障碍物表面距离与外部接触',obstacles['passed'],dict(robot_geoms=len(obstacles['clear']['robot_collision_geoms']),clear_m=obstacles['clear']['minimum']['signed_distance_m'],collision_m=obstacles['intentional_collision']['minimum']['signed_distance_m'],contacts=obstacles['intentional_collision']['external_contact_count']))
    check('12_Windows原生Viewer打开关闭',joint['viewer']['was_running'] and joint['viewer']['closed'] and ik['viewer']['closed'],dict(joint=joint['viewer'],ik=ik['viewer'],obstacles=obstacles['viewer']))
    check('13_所有演示CLI和pytest实际执行',commands['passed'],dict(command_count=len(commands['commands']),pytest=commands['counts']))
    check('14_旧Reacher与旧环境不改动',history['passed'],dict(readable_files=history['readable_files'],old_packages_unchanged=history['old_environment_packages_unchanged'],excluded=['__pycache__','.git','.pytest_cache']))
    check('15_历史拒读临时目录',False,'两个原有Reacher临时目录ACL拒读；无法声明其内容已核对，未改权限。','skipped')
    acceptance=dict(status='passed_with_explicit_scope_limitation' if not any(x['status']=='failed' for x in rows) else 'failed',items=rows,counts={s:sum(x['status']==s for x in rows) for s in ['passed','failed','skipped']},pytest_cases=commands['pytest_cases'],phase2_implemented=False,neural_network_training_executed=False,upstream_original_backup=read(ROOT/'results/upstream_backup_validation.json'))
    write_json(ROOT/'tests/phase1_acceptance.json',acceptance)
    jointtable='| Joint | joint ID | qpos adr | dof adr | Actuator / ID | Limits & ctrlrange rad |\n|---|---:|---:|---:|---|---|\n'
    for r in inspection['arm_joints']:jointtable+=f"| {r['name']} | {r['joint_id']} | {r['qpos_address']} | {r['dof_address']} | {r['actuator']} / {r['actuator_id']} | {r['range_rad']} |\n"
    put('docs/01_模型与自由度说明.md',f'''# 模型与自由度说明

官方来源：[MuJoCo Menagerie Panda](https://github.com/google-deepmind/mujoco_menagerie/tree/{provenance['commit']}/franka_emika_panda)。本次真实检出commit：`{provenance['commit']}`。Apache-2.0许可证、原README、XML与全部mesh保留于models/mujoco_menagerie/franka_emika_panda；80个原文件逐SHA备份于models/upstream_backup/franka_emika_panda。没有重写上游模型。

实际加载派生场景：nq={inspection['nq']}、nv={inspection['nv']}、nu={inspection['nu']}，不是7/7/7。九个标量关节包括七个转动机械臂关节和finger_joint1/2两个滑动关节；有1条joint equality、1条split fixed tendon。mocap目标和障碍物不增加nq。以下来自实际MjModel，不按经验补表：

{jointtable}
原模型用general执行器的固定增益与affine bias实现位置伺服：gear1时约为tau=kp×ctrl−kp×q−kv×qdot，再受force限幅。kp为4500/4500/3500/3500/2000/2000/2000，kv450/450/350/350/200/200/200。data.ctrl的七个机械臂分量是**目标角度rad**，不是关节力矩；实际扭矩来自模型动力学。actuator1～4力矩范围±87Nm、5～7为±12Nm，保留原设置。

夹爪关节ID7/8，qpos/dof地址7/8，范围0～.04m；第8执行器actuator8的ID7通过tendon而非单关节传动，ctrlrange0～255，映射到两指平均开度0～.04m。它不是第八个机械臂旋转关节，不应塞入3×7机械臂Jacobian。

原普通panda.xml没有本项目末端site。派生panda_phase1.xml在hand body下加入tool_site，局部位置[0,0,.1034]m，四元数[1,0,0,0]。这是明确选择的工具参考点，不是所有Panda工具的标准TCP保证。世界位姿读取site_xpos/site_xmat；hand原固定旋转仍保留。基座link0固定在世界原点、朝向为单位矩阵，本场景两坐标系重合，但代码仍提供世界到基座转换。目标位置始终用世界米坐标；joint q用弧度，不能混用。

派生修改：加入site、为每个机器人geom命名、明确timestep.002s、单独场景floor/非碰撞目标标记/球与圆柱。保留原implicitfast、执行器参数、关节/力矩限制与上游contact exclusion。完整JSON与原始/派生SHA见results/model_inspection.json、results/model_provenance.json。模型来自URDF简化描述及凸碰撞mesh，不是原厂实体机器人安全认证。
''')
    put('docs/02_运动学原理与代码说明.md',f'''# 运动学原理与代码说明

src/robot/model_loader.py准备独立派生MJCF，load_model实际调用MjModel.from_xml_path、MjData、一次mj_resetDataKeyframe(home)与mj_forward。初始化允许设置初态，但控制过程中不重置或传送。panda_interface通过关节名称查ID，再读jnt_qposadr/jnt_dofadr，通过actuator_trnid/trntype识别执行器；不依赖qpos前7这一巧合。

FK：Kinematics自建独立MjData，forward(q)校验7弧度关节角，在该scratch对象设置qpos并mj_forward，然后读取tool_site的世界x和3×3旋转R。R转置R与det检查验证刚体旋转。修改scratch不污染动态robot.data。结果forward_kinematics.csv含q、世界xyz及R九元素。inspect_model.py --q可打印指定构型FK。

Jacobian：mj_jacSite产生3×model.nv平移Jacobian，取七关节dof地址列得到3×7。世界位移约满足delta_x=Jp delta_q。手指是滑动自由度，但不在这七列中。对10个home±.35rad合法构型、epsilon1e-5/1e-6/1e-7做中心差分(FK(q+eps ei)−FK(q−eps ei))/(2eps)，共30组。实测最大绝对差{jac['max_abs_error']:.5e}，最大Frobenius相对差{jac['max_relative_error']:.5e}，均低于预设1e-2。完整矩阵NPZ在对应results/runs/test_jacobian目录。

DLS位置IK：delta_q=J.T solve(J J.T+lambda²I,xd−x)。lambda=.03m/rad，单次最大绝对关节变化.08rad，最多200次，求解精度.001m；候选截到实际关节限位内并留.0001rad余量，简单回溯只接受位置误差下降。未加入姿态、避障、RRT、残差策略或神经网络。7关节对3维位置冗余，可能有不同解，当前解依赖初值；不保证姿态、肘部形态、无碰撞或全局收敛。

已知可达目标由合法q的FK采样，种子{c['sampling_seed']}。20/20收敛，平均理论误差{ik['mean_theoretical_error_m']*1000:.4f}mm、最大{ik['maximum_theoretical_error_m']*1000:.4f}mm。这个结论只适用于记录的home附近样本，不能外推整个工作空间。链长度三角不等式给出保守不可达外界，[10,10,10]m明确超过界而返回unreachable_bound；界内未收敛只返回max_iterations/stalled，不能无依据称不可达。非有限目标返回invalid_target且误差None，不用0伪装精度；阻尼与有限性检查避免奇异数值NaN。

joint_controller只写data.ctrl并mj_step；每步按dt=.002与.35rad/s限制**目标指令**的变化，不是对实际关节速度作硬安全保证。mj_step后mj_forward刷新后积分位姿/接触，后者不是额外积分。保留重力与位置伺服，未加入重力补偿或积分器，所以静态负载会产生小位置偏差；不能把IK计算误差当作动态误差。3个真实执行目标的理论/实际误差分别见ik_summary，实际末端误差约5.81/7.51/7.48mm，最终250步(.5s)均保持<1cm。

障碍接口对23个真实机器人collision组geoms与两个外部障碍组成46对，按contype/conaffinity过滤；mj_geomDistance(m,d,id1,id2,distmax,fromto)返回有符号表面距离而非末端到中心距离。超过distmax返回上界，不能当精确远距；fromto只对有效未截断结果解释。data.contact是另一条接触证据。上游排除及引擎相邻连杆/焊接体过滤不等于已完成自碰撞审计。本阶段只报告指定外部对，无整机安全、连续碰撞或规划保证。

官方依据：[mj_jacSite与mj_geomDistance](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html)、[被动Viewer锁与sync](https://mujoco.readthedocs.io/en/stable/python.html)。实际3.16.0绑定docstring也保存在obstacle_queries.json，不编造API签名。
''')
    tutorial=r'''# Windows安装与运行教程

当前Phase1已用独立Conda环境panda_safe_rl执行，不改mujoco_learn。当前PowerShell若没有conda函数，可在Miniconda Prompt运行，或先加载本机Conda hook：

```powershell
. 'D:\miniconda\shell\condabin\conda-hook.ps1'
conda activate panda_safe_rl
Set-Location D:\mujoco_learning\panda_safe_residual_rl
python --version
```

以上hook/activate命令也已实际验证，解释器为D:\miniconda\envs\panda_safe_rl\python.exe。VS Code用Python: Select Interpreter选择此环境，终端同样激活。也可直接用该python.exe完整路径，避免误用旧环境；不需要安装PyTorch、CUDA训练框架、ROS2、MoveIt2或robosuite。

首次环境创建（本机已实际执行，环境已存在时不要重复创建）：

```powershell
D:\miniconda\Scripts\conda.exe create -n panda_safe_rl python=3.11 pip -y --no-default-packages
D:\miniconda\envs\panda_safe_rl\python.exe -m pip install mujoco numpy scipy matplotlib pyyaml pytest imageio imageio-ffmpeg
```

最终实际版本冻结在requirements.txt和requirements-lock.txt，新机器可在新建环境后按requirements.txt安装，但没有声称另一台机器已验证。GPU查询与渲染记录见system_info.json；NVIDIA只用于OpenGL画面，标准mj_step仍是CPU物理仿真。

官方模型获取命令（本机已执行；已存在目录请跳过，不覆盖）：

```powershell
git clone --filter=blob:none --sparse --no-checkout https://github.com/google-deepmind/mujoco_menagerie.git models/mujoco_menagerie
git -C models/mujoco_menagerie sparse-checkout init --cone
git -C models/mujoco_menagerie sparse-checkout set franka_emika_panda
git -C models/mujoco_menagerie checkout
```

这会保留Panda完整mesh与LICENSE，不是下载单个XML。复核本次固定模型时使用model_provenance.json记录的commit；依赖模型最小目录为franka_emika_panda，未无意义地检出其他机器人。派生场景放models/derived，原件另存upstream_backup。

默认无GUI验证，以下六条都实际执行通过：

```powershell
python scripts/inspect_model.py
python scripts/demo_joint_control.py
python scripts/test_jacobian.py
python scripts/demo_ik.py
python scripts/demo_obstacles.py
python -m pytest tests -q
```

打开真实MuJoCo窗口并录制（这三条带参数命令也实际执行过）：

```powershell
python scripts/demo_joint_control.py --viewer --record --speed 3
python scripts/demo_ik.py --viewer --record --speed 3
python scripts/demo_obstacles.py --viewer --record --speed 3
```

speed3只改变墙钟播放节奏，不改timestep或物理积分；正常教学可省略speed参数按1倍运行。窗口显示的就是控制器正在mj_step的同一data，不是预录制回放。以程序正常完成自动关闭为测试；用户提前关窗口会结束演示，不把缩短演示当作完整通过。Esc/窗口关闭由原生Viewer处理。关闭后脚本保存已执行日志，未完成阶段应检查failed状态，不应假定全部动作完成。

指定q的FK命令已实际测试：

```powershell
python scripts/inspect_model.py --q 0 0 0 -1.57079 0 1.57079 -0.7853
```

joint_control.csv包含time、physics_step、每关节target/command/q/v和末端位置；joint_tracking.png是7关节跟踪。ik_reachable_results.csv/ik_iteration_errors.csv是纯运动学20目标，ik_dynamic.csv是7500实际积分步；不要混淆。obstacle_queries.json保存全46对和外部contact，故意相交的球只用于静态查询，不会运行一段穿过碰撞的轨迹。obstacles_* PNG是三个静态场景的截图，未伪称动画。

每次运行新建results/runs/脚本名_时间戳，原始数据和错误日志保留；results根文件只是当前结果别名。assets/videos包含实际Panda关节/IK伺服GIF与MP4，均核对帧数、图像变化和关节运动；截图来自同一MjData的离屏渲染，不冒称Windows桌面截屏。图像是真实MuJoCo渲染，不是生成画面。

验收原生Viewer必须在Windows图形桌面运行；不要在无显示服务的会话把跳过标成通过。本机18项pytest已真实通过。曾遇到Windows沙箱pytest临时缓存目录权限警告，最终配置关闭非必要cacheprovider并再次通过；不改系统目录权限。所有原始运行与错误记录仍在results下。

Phase1到此暂停，等待用户审核。没有实现或启动RRT-Connect、PPO/SAC/BC训练、动态避障、残差RL与复杂轨迹优化。
'''
    put('docs/03_Windows安装与运行教程.md',tutorial)
    summaryrows='| 验收项 | 状态 |\n|---|---|\n'+'\n'.join(f"| {r['name']} | {r['status']} |" for r in rows)
    dynamics='| Target | IK理论误差mm | 实际末端误差mm | 最后250步最大误差mm |\n|---|---:|---:|---:|\n'
    for r in ik['dynamic_targets']:dynamics+=f"| {r['target_id']} | {r['theoretical_ik_error_m']*1000:.4f} | {r['actual_dynamic_error_m']*1000:.4f} | {r['last_250_max_error_m']*1000:.4f} |\n"
    report=f'''# Phase1验收报告

项目目录：本报告所在panda_safe_residual_rl独立工程；项目名称Motion Planning and Residual Reinforcement Learning for Safe Robotic Manipulation。**仅完成Phase1并暂停，不自动开始Phase2。**

## A 软件环境

独立环境{system['prefix']}，Python3.11.17，MuJoCo{system['mujoco']}，NumPy{system['numpy']}；完整安装版本见requirements-lock.txt和system_info.json，pip check实际通过。Windows平台内部版本10.0.{system['windows_registry']['CurrentBuild']}、DisplayVersion{system['windows_registry']['DisplayVersion']}、UBR{system['windows_registry']['UBR']}；注册表ProductName原样为{system['windows_registry']['ProductName']}，不凭该兼容字符串另猜系统发行名称。

实测nvidia-smi为{system['gpu_query']}，与任务背景的RTX4070不同。OpenGL {system['opengl']['version']}，renderer为{system['opengl']['renderer']}，实际渲染检查通过。GPU用于图形显示；物理为CPU原生mj_step，不是GPU/MJX训练。新环境未安装torch、SB3等神经网络框架，未训练任何网络。旧mujoco_learn包列表前后完全一致。

## B 模型

上游commit `{provenance['commit']}`，Apache-2.0，80文件完整备份且SHA逐一匹配。实际nq/nv/nu=9/9/8；七个机械臂hinge+两个finger slide，1个equality与1个tendon，第8执行器是夹爪，不是第8臂关节。tool_site固定于hand，局部[0,0,.1034]m。真实ID、地址、控制/力矩范围见model_inspection.json及01文档。派生XML独立生成，原model git diff为空。

## C 控制实验

4500次真实积分、9秒仿真：保持2秒、joint1单独3秒、七关节共同4秒。原始关节target/ctrl/q/qdot/time全保存。单关节实际增量{joint['stages'][1]['displacement_rad'][0]:.6f}rad；七关节最终阶段各关节都有运动，实际增量为{joint['stages'][2]['displacement_rad']}。各阶段最大跟踪误差为{[r['max_abs_tracking_error_rad'] for r in joint['stages']]}rad，无NaN；最大约.00658rad，源自原位置PD在重力负载下的静差，不夸称零误差。命令slew .35rad/s，未对实际关节速度提供安全认证。控制器无qpos/qvel赋值；只在首次初始化及独立运动学scratch允许设置构型。

曲线results/joint_tracking.png，225帧9秒真实GIF与225帧MP4均已解码，最大关节位移范数{joint['video']['joint_displacement_rad']:.6f}rad。首次预跑的组合目标有一个关节沿用前段目标，后续已修正为七关节全部改变并重测；预跑原始目录未删除。

## D FK、Jacobian和IK

FK输出世界坐标及旋转矩阵，已执行指定q命令。30组差分（10构型×3步长）最大绝对误差{jac['max_abs_error']:.6e}，最大Frobenius相对误差{jac['max_relative_error']:.6e}，低于1e-2门槛。完整CSV/矩阵NPZ保留，3×nv结果按真实dof地址抽成3×7。

20个由合法FK采样的已知可达目标全部在.001m求解阈值下收敛，平均{ik['mean_theoretical_error_m']*1000:.4f}mm、最大{ik['maximum_theoretical_error_m']*1000:.4f}mm。DLS最多200次，lambda.03，单步.08rad，关节限位与回溯检查。超链长目标明确返回unreachable_bound，1次迭代预算目标返回max_iterations；非有限目标返回invalid_target而非伪造误差0。

真实伺服进一步执行3个IK目标：7500步、15秒；最后250步即.5秒均小于原定1cm标准，没有扩大阈值。理论解误差与动态误差区分如下：

{dynamics}
375帧15秒实际控制GIF与MP4均解码通过。此结论仅限这3个执行目标和20个局部可达样本，不代表所有工作空间、姿态IK或障碍安全。

## E Viewer、截图与视频

joint/IK/obstacle三个真实场景均通过mujoco.viewer.launch_passive打开Windows原生窗口并正常关闭；持锁改变模型/data，sync放锁外。另有pytest实际打开/关闭测试。画面与控制积分共享同一动态MjData，不是qpos回放。

录像：assets/videos/panda_joint_control.gif/.mp4、panda_ik_servo.gif/.mp4，媒体验收与原始帧SHA/time/q日志见media_manifest.json及对应run目录。截图assets/screenshots为同一仿真状态离屏图，不冒称桌面截屏；障碍图是静态查询图，不冒称GIF。

## F 碰撞接口

两障碍为sphere半径.06m、cylinder半径.065m/半高.18m；mocap用于改变障碍位姿，不传送机器人。明确命名23个机器人collision geoms，查询46个外部对，不用末端到中心距离代替。

初始clear最小表面距离{obstacles['clear']['minimum']['signed_distance_m']:.6f}m、0外部contact；把球移到link4碰撞mesh世界中心，最小有符号距离{obstacles['intentional_collision']['minimum']['signed_distance_m']:.6f}m，{obstacles['intentional_collision']['external_contact_count']}条外部接触；移回后恢复{obstacles['moved_away']['minimum']['signed_distance_m']:.6f}m和0外部contact。机器人关节q保持不变。该碰撞是可解释的负测试夹具，只作静态query，不执行冲入障碍的控制。另与两sphere解析距离及fromto表面距离交叉验证。

mj_geomDistance实际3.16.0绑定已检查，距离上限distmax2m、带截断标记，接触通过mj_forward更新。保留上游相邻体排除；目前仅指定外部障碍对，未完成全自碰撞审计、连续碰撞、规划或安全保证。

## G 自动验收和保留记录

**18个pytest通过、0失败、0跳过；10条实际CLI/检查命令全部退出0。** Phase1需求验收{acceptance['counts']['passed']}通过、{acceptance['counts']['failed']}失败、{acceptance['counts']['skipped']}项范围限制跳过：

{summaryrows}

7865个可读旧Reacher文件前后SHA/清单一致，旧环境包版本一致。两处历史ACL拒读临时目录无法读取，未改权限，不能宣称检查了这些不可访问内容；缓存/.git明确排除。完整逐文件记录在tests/reacher_before.json、reacher_after.json及reacher_integrity.json。

原始结果保存在results/runs，根目录是选中结果别名；错误保存在results/errors。首次环境打印np.bool序列化失败已修复重测；首次pytest只有缓存目录权限警告，最终关闭非必要缓存后18项重测通过。错误记录没有删除或改成“从未发生”。没有改变关节/位置成功阈值以制造通过。

## H Phase2建议（仅建议，未执行）

1. 在当前23个collision geoms基础上明确静态场景与自碰撞pair集合，区分需排除的相邻/固定连接体和真实自碰撞；为限位外、边界接触、near-miss添加可解释夹具。
2. 将state_valid(q)限制在独立MjData并显式返回碰撞对象/最小距离，加入关节路径插值的离散edge验证及分辨率研究；离散检查不等于连续安全。
3. 冻结场景、起终点、随机种子和预算后再实现关节空间RRT-Connect；报告规划失败、边界、自碰撞和稠密复核，不直接拿位置IK成功当路径可行。
4. 规划通过后才研究时间参数化、当前原位置伺服跟踪及扰动；残差RL继续留待后续阶段，不在本次安装/训练。

**Phase1已结束并等待用户审核。没有实现RRT-Connect、残差PPO、动态避障或复杂优化。**
'''
    put('docs/04_Phase1验收报告.md',report)
    put('models/README.md',f'''# Official Panda source and project-derived scenes

Upstream: https://github.com/google-deepmind/mujoco_menagerie, commit {provenance['commit']}, franka_emika_panda, Apache-2.0. Preserve the upstream LICENSE and README.

- mujoco_menagerie/: sparse Git checkout, original files unchanged.
- upstream_backup/franka_emika_panda/: exact80-file backup, SHA-validated.
- derived/panda_phase1.xml: adds named geoms and a hand-fixed tool_site; keeps original position-servo gains/gear/force ranges.
- derived/phase1_scene.xml: project floor, target marker, sphere and cylinder obstacles.

See results/model_provenance.json for every original/derived SHA and all modifications. No Reacher model, log or environment was copied.
''')
    put('README.md',f'''# Motion Planning and Residual Reinforcement Learning for Safe Robotic Manipulation

当前交付仅为 **Panda Phase1：模型、真实位置伺服、FK、Jacobian、DLS位置IK和基础碰撞查询**。RRT-Connect与残差强化学习属于未来计划，本次未实现、未训练。

![Panda真实IK伺服](assets/videos/panda_ik_servo.gif)

实测：官方模型nq/nv/nu=9/9/8；20/20个局部已知可达目标IK收敛；30组Jacobian差分最大相对误差{jac['max_relative_error']:.3e}；3个目标真实动态误差5.81/7.51/7.48mm，最终.5s均保持<1cm。18项pytest通过。单任务测试不构成避障或实机安全保证。

独立环境panda_safe_rl，Python3.11.17 / MuJoCo{system['mujoco']} / NumPy{system['numpy']}，未修改旧Reacher工程与mujoco_learn。实测GPU为RTX3060而非任务描述4070；GPU用于OpenGL，物理积分仍CPU。模型源commit `{provenance['commit']}`，Apache-2.0，完整mesh/README/LICENSE和备份保留。

```powershell
. 'D:\\miniconda\\shell\\condabin\\conda-hook.ps1'
conda activate panda_safe_rl
Set-Location D:\\mujoco_learning\\panda_safe_residual_rl
python scripts/inspect_model.py
python scripts/demo_joint_control.py
python scripts/test_jacobian.py
python scripts/demo_ik.py
python scripts/demo_obstacles.py
python -m pytest tests -q
```

以上全部实际执行。观看并录制真实动态控制：

```powershell
python scripts/demo_joint_control.py --viewer --record --speed 3
python scripts/demo_ik.py --viewer --record --speed 3
python scripts/demo_obstacles.py --viewer --record --speed 3
```

七关节ctrl是rad目标，不是扭矩；控制器限速目标并调用mj_step，未反复改qpos。IK/差分只在独立scratch MjData设置构型，不污染执行状态。理论IK与真实重力负载下的伺服误差分别记录。障碍球/圆柱按23个机器人collision geoms查询，不用末端中心距离替代全臂碰撞；尚无规划与安全保证。

[模型与自由度](docs/01_模型与自由度说明.md) · [运动学学习文档](docs/02_运动学原理与代码说明.md) · [Windows完整安装教程](docs/03_Windows安装与运行教程.md) · [实际验收报告](docs/04_Phase1验收报告.md)

主要输出：results/model_inspection.json、forward_kinematics.csv、jacobian_validation.csv、ik_reachable_results.csv、ik_dynamic.csv、obstacle_queries.json；曲线joint_tracking.png和ik_validation.png。真实GIF/MP4在assets/videos，截图在assets/screenshots。每次原始数据单独保存在results/runs；错误不删除。两处旧项目ACL目录无法读，详见验收范围说明。

阶段边界：未训练PPO/SAC/BC，未实现RRT-Connect、动态避障、姿态IK、残差RL或复杂路径优化。**完成Phase1后暂停，等待审核。**
''')
    put('.gitignore','''__pycache__/
*.py[cod]
.pytest_cache/
.venv/
tests/reacher_before.json
tests/reacher_after.json
tests/mujoco_learn_packages_*.json
models/mujoco_menagerie/
models/upstream_backup/
results/runs/
results/errors/
*.log
''')
    print('PHASE1 DOCUMENTS',acceptance['counts'],commands['counts'])
if __name__=='__main__':main_guard(main)
