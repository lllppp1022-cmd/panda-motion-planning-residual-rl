"""Audit historical D failures without modifying historical trajectories."""
from _bootstrap import main_guard
import csv,json
from pathlib import Path
import numpy as np
import mujoco
from src.common import ROOT,write_json,write_csv
from src.control.validated_path import load_verified_path


def main():
    read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
    batch=Path(read(ROOT/'results/phase2c_batch_latest.json')['run']);runs=read(batch/'complete.json')['results']
    rows=[];detail=[]
    for r in runs:
        if r['scene']!='D_two_obstacles':continue
        with (Path(r['run_directory'])/'tracking.csv').open(encoding='utf-8-sig',newline='') as f:data=list(csv.DictReader(f))
        first=r['first_violation'];t=first['time'];row=min(data,key=lambda x:abs(float(x['time'])-t))
        qref=np.array([float(row[f'q_ref_{i}']) for i in range(1,8)])
        qa=np.array([float(row[f'q_actual_{i}']) for i in range(1,8)])
        ctrl=np.array([float(row[f'ctrl_{i}']) for i in range(1,8)])
        vel=np.array([float(row[f'dq_actual_{i}']) for i in range(1,8)])
        bundle=load_verified_path('D_two_obstacles',42);c=bundle.checker;m=c.model
        pair=first['monitor']['offending_geom_pairs'][0];a,b=pair['id1'],pair['id2']
        def distance(q):
            c.state_valid(q)
            return float(mujoco.mj_geomDistance(m,c.data,a,b,.1,None))
        refgap=distance(qref);actualgap=distance(qa);gradients={}
        for eps in [1e-3,5e-4,1e-4,1e-5]:
            gradient=[]
            for j in range(7):
                delta=np.zeros(7);delta[j]=eps
                gradient.append((distance(qref+delta)-distance(qref-delta))/(2*eps))
            gradients[str(eps)]=np.asarray(gradient)
        gradient=gradients['0.001'];error=qa-qref
        directional=[dict(fraction=fraction,distance_m=distance(qref+fraction*error)) for fraction in [0.,.01,.1,.25,.5,.75,1.]]
        c.state_valid(qa);c.data.qvel[:]=0.;mujoco.mj_forward(m,c.data)
        kp=m.actuator_gainprm[c.robot.actuator_ids,0];kd=-m.actuator_biasprm[c.robot.actuator_ids,2]
        gravity_force=c.data.qfrc_bias[c.robot.dof_indices].copy()
        command=ctrl-qref;gravity=-gravity_force/kp;velocity=-kd/kp*vel
        residual=error-command-gravity-velocity
        raw_force=kp*(ctrl-qa)-kd*vel;limits=m.actuator_forcerange[c.robot.actuator_ids]
        saturated=(raw_force<limits[:,0]-1e-9)|(raw_force>limits[:,1]+1e-9)
        prehold=[x for x in data if x['phase']=='prehold'];active=[x for x in data if x['phase'] in ('trajectory','settle')]
        def accel_max(part):
            values=[(abs(float(x[f'ddq_actual_fd_{j}'])),float(x['time']),j) for x in part for j in range(1,8)]
            return max(values)
        pre_max=accel_max(prehold);motion_max=accel_max(active)
        hold=[x for x in data if float(x['time'])>=t-1e-9]
        qlast=np.array([float(hold[-1][f'q_actual_{i}']) for i in range(1,8)])
        final_ctrl=np.array([float(hold[-1][f'ctrl_{i}']) for i in range(1,8)])
        contributions=dict(command=gradient*command,gravity_quasistatic=gradient*gravity,velocity_damping=gradient*velocity,
                           residual_transient_passive_coriolis_constraints=gradient*residual)
        probe=dict(scene=r['scene'],speed=r['reference_speed_rad_s'],repeat=r['repeat'],first_violation=first,
            q_ref=qref,q_actual=qa,ctrl=ctrl,qvel_actual=vel,joint_error=error,reference_pair_gap_m=refgap,actual_pair_gap_fixed_fingers_m=actualgap,
            actual_pair_gap_live_m=pair['distance_m'],distance_gradient_m_per_rad=gradient,gradient_resolution_comparison=gradients,
            directional_distance_samples=directional,
            observed_pair_gap_change_m=actualgap-refgap,linearized_gap_change_m=float(gradient@error),
            linearization_residual_m=float(actualgap-refgap-gradient@error),
            command_error_rad=command,gravity_quasistatic_error_estimate_rad=gravity,velocity_damping_error_term_rad=velocity,
            residual_error_rad=residual,local_gap_contributions_m=contributions,servo_force_clip_at_first_violation=saturated,
            prehold_acceleration_peak=dict(value_rad_s2=pre_max[0],time_s=pre_max[1],joint=pre_max[2]),
            motion_acceleration_peak=dict(value_rad_s2=motion_max[0],time_s=motion_max[1],joint=motion_max[2]),
            post_stop_duration_s=float(hold[-1]['time'])-t,post_stop_joint_displacement_rad=qlast-qa,
            post_stop_joint_L2_displacement_rad=float(np.linalg.norm(qlast-qa)),
            post_stop_final_ctrl=final_ctrl,hold_target_minus_first_actual_rad=final_ctrl-qa,
            post_stop_drift_gravity_cosine=float(np.dot(qlast-qa,gravity)/(np.linalg.norm(qlast-qa)*np.linalg.norm(gravity))),
            post_stop_minimum_gap_m=min(float(x['minimum_clearance_m']) for x in hold),
            post_stop_min_time_s=min(hold,key=lambda x:float(x['minimum_clearance_m']))['time'],
            history_source=r['run_directory'],local_estimates_only=True)
        detail.append(probe)
        for j in range(7):
            rows.append(dict(speed_rad_s=r['reference_speed_rad_s'],deterministic_repeat=r['repeat'],time_s=t,joint=j+1,
                q_ref=qref[j],q_actual=qa[j],ctrl=ctrl[j],qvel=vel[j],joint_error_rad=error[j],gradient_m_per_rad=gradient[j],
                local_gap_change_m=gradient[j]*error[j],command_error_rad=command[j],gravity_quasistatic_rad=gravity[j],
                velocity_damping_rad=velocity[j],residual_rad=residual[j],reference_pair_gap_m=refgap,
                actual_pair_gap_m=actualgap,linearization_residual_m=probe['linearization_residual_m']))
    write_csv(ROOT/'results/phase2d_tracking_clearance_analysis.csv',rows)
    write_json(ROOT/'results/phase2d_failure_analysis.json',dict(historical_failures=detail,execution_threshold_m=.005,
        notes='Local decomposition and distance linearization are diagnostics, not global safety guarantees. Gravity term is a local estimate, not a new controller.'))
    tables=[]
    for d in detail[::2]:
        tables.append(f"| {d['speed']:.2f} | {d['first_violation']['time']:.6f} | {d['reference_pair_gap_m']*1000:.6f} | {d['actual_pair_gap_live_m']*1000:.6f} | {d['observed_pair_gap_change_m']*1000:.6f} | {d['linearized_gap_change_m']*1000:.6f} | {d['post_stop_minimum_gap_m']*1000:.6f} |")
    d=detail[0];parts=d['local_gap_contributions_m'];parts_text='\n'.join(f'- {k}：{np.sum(v)*1000:.6f} mm（局部线性距离贡献）' for k,v in parts.items())
    text=f'''# Phase 2C 双障碍执行失败机制分析

基于未改动的 Phase 2C 正式 D_two_obstacles 四条失败记录，规划种子 42，两档速度各确定性重复两次。原路径最小细化采样间隙 5.756322 mm，执行门槛始终 5 mm。数据源：`{batch}`。

## 首次违规与参考/实际构型

| 速度 rad/s | 首次违规仿真时刻 s | 同时刻参考配对间隙 mm | 实际live间隙 mm | 实际减参考 mm | 局部线性预测 mm | 保持期间最低间隙 mm |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(tables)}

违规配对为 `panda_right_finger_collision_2` ↔ `obstacle_cylinder`。两次重复执行不是独立随机场景；同一速度的结果相同。第一次违规是低于门槛，并不表示已发生接触；历史四次均没有原生禁止接触。

## 七关节误差与局部间隙敏感性

在实际保存的首次违规 q_ref 周围，对同一碰撞配对作中心有限差分，步长 1e-3、5e-4、1e-4 和 1e-5 rad；主要表格采用 1e-3 rad，避免把接近原距离求解容差的极小扰动当作高精度导数。另保存沿实际关节误差方向 0～100% 的距离样本，揭示有限扰动的非线性。计算的是该具体配对的表面距离梯度，不将会切换配对的全局最小距离假定为处处可微。真实模型、原障碍和固定夹爪查询状态保持不变；固定夹爪与原live距离的微小差异另存。

0.10 rad/s 的实际减参考七关节误差为 `{np.round(d['joint_error'],8).tolist()}` rad；梯度为 `{np.round(d['distance_gradient_m_per_rad'],6).tolist()}` m/rad。逐关节数值和贡献见 `results/phase2d_tracking_clearance_analysis.csv`。预测间隙变化与实际变化的残差为 {d['linearization_residual_m']*1000:.6f} mm。应结合不同差分步长的差异和沿误差方向的距离曲线判断近似质量，不能把局部贡献直接当作准确的全局原因占比，更不构成严格安全证明。

## 下发指令、速度滞后和重力静差

原伺服关系为 τ≈Kp(ctrl−q)−Kd qvel，并受原力限约束。因此将保存的误差分解为：

```
q_actual−q_ref = (ctrl−q_ref) − (Kd/Kp) qvel − gravity(q_actual)/Kp + residual
```

其中 gravity 通过独立数据对象在同一 q_actual、零速度下由 mj_forward 的 qfrc_bias 得到，反映局部准静态重力偏差估计；没有把它添加到控制器。residual 包含惯性、科氏/离心、被动阻尼、约束、非线性及可能力限等未单独分离项，不能一概称为“纯跟踪滞后”或“纯重力”。夹爪未保存的极小速度用零近似，局部估计不能代替实际运行。

0.10 rad/s 首次违规时的局部贡献：

{parts_text}

实际 ctrl 与 q_ref 的差值、七关节速度、预测重力项、速度阻尼项和剩余项均单列保存。首次违规处按保存的 q/ctrl/qvel 重算原位置执行器力限，结果 `{d['servo_force_clip_at_first_violation'].tolist()}`。静态重力偏差和运动滞后共同改变几何间隙，不能仅以降低速度宣称控制器改进。

## 保持指令后的实际运动

首次违规后轨迹时间冻结，目标转为首次违规的实际构型，原指令变化限速仍有效。0.10 rad/s 后续记录 {d['post_stop_duration_s']:.6f} s，关节L2位移仍有 {d['post_stop_joint_L2_displacement_rad']:.6f} rad；实际间隙进一步降低到 {d['post_stop_minimum_gap_m']*1000:.6f} mm。不是瞬间停止，也没有通过 qpos 重置删除惯性响应。

还需区分“惯性”和“改变保持平衡点”：原控制器依靠位置偏差产生支撑重力的力矩；把位置目标重设为当前实际 q 后，原来这部分比例支撑会被改变，新的平衡点仍有重力静差。保存日志中最终 ctrl 与首次违规实际 q 的差为 `{np.round(d['hold_target_minus_first_actual_rad'],10).tolist()}` rad；停止后位移与当地准静态重力偏差方向的余弦为 {d['post_stop_drift_gravity_cosine']:.6f}。这支持“保持目标变化后重力再平衡也参与漂移”的解释，不能把全部停止后位移称为纯惯性。此处只审计原保持机制，没有修改它或加入补偿。

## 初始化加速度峰值的归因

0.10 rad/s 初始化保持阶段峰值 {d['prehold_acceleration_peak']['value_rad_s2']:.6f} rad/s²，时刻 {d['prehold_acceleration_peak']['time_s']:.6f} s、关节 {d['prehold_acceleration_peak']['joint']}；轨迹运动阶段峰值 {d['motion_acceleration_peak']['value_rad_s2']:.6f} rad/s²。前者是正式轨迹推进前的物理启动/重力瞬态，不应误归因于 RRT 轨迹速度或五次参考的加速度越限。参考限制从来不是实际加速度硬保证。

## 结论与边界

历史路径只有约 0.756 mm 的规划间隙储备，实际跟踪误差足以消耗它。本阶段研究提高几何规划裕度的效果，保留原位置伺服及执行5 mm标准；不以局部梯度或仿真无接触宣称通用安全保证。全部原始历史 CSV、首次违规构型、加速度归因和局部估计见 `results/phase2d_failure_analysis.json`。未训练 PPO/SAC。
'''
    (ROOT/'docs/13_Phase2C失败机制分析.md').write_text(text,encoding='utf-8')
    print('Analyzed',len(detail),'preserved D failures',flush=True)


if __name__=='__main__':main_guard(main)
