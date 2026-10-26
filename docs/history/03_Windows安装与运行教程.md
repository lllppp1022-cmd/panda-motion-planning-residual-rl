# Windows安装与运行教程

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
