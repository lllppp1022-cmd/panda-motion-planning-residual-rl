# Phase1验收报告

项目目录：本报告所在panda_safe_residual_rl独立工程；项目名称Motion Planning and Residual Reinforcement Learning for Safe Robotic Manipulation。**仅完成Phase1并暂停，不自动开始Phase2。**

## A 软件环境

独立环境D:\miniconda\envs\panda_safe_rl，Python3.11.17，MuJoCo3.16.0，NumPy2.4.6；完整安装版本见requirements-lock.txt和system_info.json，pip check实际通过。Windows平台内部版本10.0.26200、DisplayVersion25H2、UBR9168；注册表ProductName原样为Windows 10 Pro，不凭该兼容字符串另猜系统发行名称。

实测nvidia-smi为NVIDIA GeForce RTX 3060, 560.94，与任务背景的RTX4070不同。OpenGL 4.6.0 NVIDIA 560.94，renderer为NVIDIA GeForce RTX 3060/PCIe/SSE2，实际渲染检查通过。GPU用于图形显示；物理为CPU原生mj_step，不是GPU/MJX训练。新环境未安装torch、SB3等神经网络框架，未训练任何网络。旧mujoco_learn包列表前后完全一致。

## B 模型

上游commit `cda5236f98383f6fbaa4820c6b18dba892cd087f`，Apache-2.0，80文件完整备份且SHA逐一匹配。实际nq/nv/nu=9/9/8；七个机械臂hinge+两个finger slide，1个equality与1个tendon，第8执行器是夹爪，不是第8臂关节。tool_site固定于hand，局部[0,0,.1034]m。真实ID、地址、控制/力矩范围见model_inspection.json及01文档。派生XML独立生成，原model git diff为空。

## C 控制实验

4500次真实积分、9秒仿真：保持2秒、joint1单独3秒、七关节共同4秒。原始关节target/ctrl/q/qdot/time全保存。单关节实际增量0.120000rad；七关节最终阶段各关节都有运动，实际增量为[-0.2199999999999564, -0.12090484660335397, 0.10013384656870791, -0.12005428880776314, 0.09994643967596148, 0.11990882120784074, 0.10000136855663966]。各阶段最大跟踪误差为[0.006581714241370611, 0.006581714242671284, 0.006367199223830555]rad，无NaN；最大约.00658rad，源自原位置PD在重力负载下的静差，不夸称零误差。命令slew .35rad/s，未对实际关节速度提供安全认证。控制器无qpos/qvel赋值；只在首次初始化及独立运动学scratch允许设置构型。

曲线results/joint_tracking.png，225帧9秒真实GIF与225帧MP4均已解码，最大关节位移范数0.288259rad。首次预跑的组合目标有一个关节沿用前段目标，后续已修正为七关节全部改变并重测；预跑原始目录未删除。

## D FK、Jacobian和IK

FK输出世界坐标及旋转矩阵，已执行指定q命令。30组差分（10构型×3步长）最大绝对误差2.530416e-09，最大Frobenius相对误差3.052752e-09，低于1e-2门槛。完整CSV/矩阵NPZ保留，3×nv结果按真实dof地址抽成3×7。

20个由合法FK采样的已知可达目标全部在.001m求解阈值下收敛，平均0.1718mm、最大0.9744mm。DLS最多200次，lambda.03，单步.08rad，关节限位与回溯检查。超链长目标明确返回unreachable_bound，1次迭代预算目标返回max_iterations；非有限目标返回invalid_target而非伪造误差0。

真实伺服进一步执行3个IK目标：7500步、15秒；最后250步即.5秒均小于原定1cm标准，没有扩大阈值。理论解误差与动态误差区分如下：

| Target | IK理论误差mm | 实际末端误差mm | 最后250步最大误差mm |
|---|---:|---:|---:|
| 0 | 0.0343 | 5.8072 | 5.8072 |
| 1 | 0.7866 | 7.5133 | 7.5133 |
| 2 | 0.0472 | 7.4850 | 7.4850 |

375帧15秒实际控制GIF与MP4均解码通过。此结论仅限这3个执行目标和20个局部可达样本，不代表所有工作空间、姿态IK或障碍安全。

## E Viewer、截图与视频

joint/IK/obstacle三个真实场景均通过mujoco.viewer.launch_passive打开Windows原生窗口并正常关闭；持锁改变模型/data，sync放锁外。另有pytest实际打开/关闭测试。画面与控制积分共享同一动态MjData，不是qpos回放。

录像：assets/videos/panda_joint_control.gif/.mp4、panda_ik_servo.gif/.mp4，媒体验收与原始帧SHA/time/q日志见media_manifest.json及对应run目录。截图assets/screenshots为同一仿真状态离屏图，不冒称桌面截屏；障碍图是静态查询图，不冒称GIF。

## F 碰撞接口

两障碍为sphere半径.06m、cylinder半径.065m/半高.18m；mocap用于改变障碍位姿，不传送机器人。明确命名23个机器人collision geoms，查询46个外部对，不用末端到中心距离代替。

初始clear最小表面距离0.739362m、0外部contact；把球移到link4碰撞mesh世界中心，最小有符号距离-0.035983m，1条外部接触；移回后恢复0.739362m和0外部contact。机器人关节q保持不变。该碰撞是可解释的负测试夹具，只作静态query，不执行冲入障碍的控制。另与两sphere解析距离及fromto表面距离交叉验证。

mj_geomDistance实际3.16.0绑定已检查，距离上限distmax2m、带截断标记，接触通过mj_forward更新。保留上游相邻体排除；目前仅指定外部障碍对，未完成全自碰撞审计、连续碰撞、规划或安全保证。

## G 自动验收和保留记录

**18个pytest通过、0失败、0跳过；10条实际CLI/检查命令全部退出0。** Phase1需求验收14通过、0失败、1项范围限制跳过：

| 验收项 | 状态 |
|---|---|
| 01_MuJoCo导入与独立环境 | passed |
| 02_Panda实际加载 | passed |
| 03_七关节真实ID地址映射 | passed |
| 04_执行器与限位控制含义 | passed |
| 05_单关节真实运动 | passed |
| 06_七关节伺服有限与真实控制 | passed |
| 07_FK有限姿态与世界坐标 | passed |
| 08_Jacobian数值差分 | passed |
| 09_已知可达位置IK | passed |
| 10_IK合理失败与隔离 | passed |
| 11_障碍物表面距离与外部接触 | passed |
| 12_Windows原生Viewer打开关闭 | passed |
| 13_所有演示CLI和pytest实际执行 | passed |
| 14_旧Reacher与旧环境不改动 | passed |
| 15_历史拒读临时目录 | skipped |

7865个可读旧Reacher文件前后SHA/清单一致，旧环境包版本一致。两处历史ACL拒读临时目录无法读取，未改权限，不能宣称检查了这些不可访问内容；缓存/.git明确排除。完整逐文件记录在tests/reacher_before.json、reacher_after.json及reacher_integrity.json。

原始结果保存在results/runs，根目录是选中结果别名；错误保存在results/errors。首次环境打印np.bool序列化失败已修复重测；首次pytest只有缓存目录权限警告，最终关闭非必要缓存后18项重测通过。错误记录没有删除或改成“从未发生”。没有改变关节/位置成功阈值以制造通过。

## H Phase2建议（仅建议，未执行）

1. 在当前23个collision geoms基础上明确静态场景与自碰撞pair集合，区分需排除的相邻/固定连接体和真实自碰撞；为限位外、边界接触、near-miss添加可解释夹具。
2. 将state_valid(q)限制在独立MjData并显式返回碰撞对象/最小距离，加入关节路径插值的离散edge验证及分辨率研究；离散检查不等于连续安全。
3. 冻结场景、起终点、随机种子和预算后再实现关节空间RRT-Connect；报告规划失败、边界、自碰撞和稠密复核，不直接拿位置IK成功当路径可行。
4. 规划通过后才研究时间参数化、当前原位置伺服跟踪及扰动；残差RL继续留待后续阶段，不在本次安装/训练。

**Phase1已结束并等待用户审核。没有实现RRT-Connect、残差PPO、动态避障或复杂优化。**
