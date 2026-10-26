from _bootstrap import *
import numpy as np
import mujoco
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.common import config,write_csv
from src.robot.model_loader import load_model
from src.robot.panda_interface import PandaInterface
from src.robot.joint_controller import JointPositionController
from src.robot.inverse_kinematics import DampedLeastSquaresIK
from src.robot.kinematics import Kinematics
from src.visualization.viewer import LiveViewer,Recorder
def main():
    a=parser('20 FK-reachable DLS targets plus real dynamic IK servo validation').parse_args();dest=destination('demo_ik');m,d=load_model();robot=PandaInterface(m,d);solver=DampedLeastSquaresIK(m);kin=Kinematics(m);rng=np.random.default_rng(config()['sampling_seed']);rows=[];traces=[]
    for i in range(20):
        known=kin.home+rng.uniform(-.35,.35,7);target=kin.forward(known)[0];initial=kin.home+rng.uniform(-.06,.06,7);before=d.qpos.copy();result=solver.solve(target,initial);np.testing.assert_array_equal(d.qpos,before)
        rows.append(dict(target_id=i,target_world=target.tolist(),known_reachable_q=known.tolist(),q_initial=initial.tolist(),q_solution=result.q.tolist(),status=result.status,success=result.success,iterations=result.iterations,theoretical_error_m=result.final_error_m,within_1cm=result.final_error_m<.01))
        for k,error in enumerate(result.errors_m):traces.append(dict(target_id=i,iteration=k,error_m=error))
    failures=[]
    for label,target,iterations in [('beyond_chain_bound',[10,10,10],200),('limited_iteration',kin.forward(kin.home+np.array([.3,-.25,.3,-.25,.3,.3,.25]))[0],1)]:
        r=solver.solve(target,kin.home,max_iterations=iterations);failures.append(dict(case=label,status=r.status,success=r.success,error_m=r.final_error_m,reason=r.reason))
    controller=JointPositionController(robot);live=LiveViewer(robot,a.viewer,a.speed);video=Recorder(robot,dest,'ik_servo',a.record);dynamic=[];dynamic_summary=[]
    try:
        for index in range(3):
            target=np.array(rows[index]['target_world']);r=solver.solve(target,robot.positions());assert r.success
            marker=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'target_marker');d.mocap_pos[m.body_mocapid[marker]]=target
            for _ in range(2500):
                if not live.running():break
                with live.lock():
                    row=controller.step(r.q);row.update(target_id=index,target_x=target[0],target_y=target[1],target_z=target[2],position_error_m=float(np.linalg.norm(robot.tool_pose()[0]-target)));dynamic.append(row);video.capture(f'IK target{index} (position only)')
                live.sync()
            last=[x for x in dynamic if x['target_id']==index][-250:];actual=robot.tool_pose()[0];error=float(np.linalg.norm(actual-target))
            dynamic_summary.append(dict(target_id=index,theoretical_ik_error_m=r.final_error_m,actual_dynamic_error_m=error,actual_world=actual.tolist(),target_world=target.tolist(),last_250_max_error_m=max(x['position_error_m'] for x in last),passed_1cm=error<.01 and all(x['position_error_m']<.01 for x in last)))
    finally:
        write_csv(dest/'ik_dynamic.csv',dynamic);view=live.close();v=video.close()
    write_csv(dest/'ik_reachable_results.csv',rows);write_csv(dest/'ik_iteration_errors.csv',traces);write_json(dest/'ik_failure_cases.json',failures)
    fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for i in range(20):
        t=[r for r in traces if r['target_id']==i];axs[0].semilogy([r['iteration'] for r in t],[max(r['error_m'],1e-9) for r in t],alpha=.7)
    axs[0].axhline(.001,color='black',ls='--');axs[0].set(xlabel='DLS iteration',ylabel='Theoretical error (m)',title='20 FK-reachable targets (position only)')
    axs[1].plot([r['time'] for r in dynamic],[r['position_error_m']*100 for r in dynamic]);axs[1].axhline(1,color='red',ls='--');axs[1].set(xlabel='Actual simulation seconds',ylabel='Actual dynamic XY Z error (cm)',title='Real servo execution;3 target segments')
    fig.tight_layout();fig.savefig(dest/'ik_validation.png',dpi=150);plt.close(fig)
    result=dict(passed=all(r['success'] for r in rows) and all(r['passed_1cm'] for r in dynamic_summary),reachable_targets=20,converged=sum(r['success'] for r in rows),mean_theoretical_error_m=float(np.mean([r['theoretical_error_m'] for r in rows])),maximum_theoretical_error_m=max(r['theoretical_error_m'] for r in rows),failure_cases=failures,dynamic_targets=dynamic_summary,physics_steps=controller.physics_steps,simulation_seconds=float(d.time),kinematics_uses_independent_data=True,viewer=view,video=v)
    write_json(dest/'ik_summary.json',result);write_json(dest/'complete.json',result);publish(dest,['ik_reachable_results.csv','ik_iteration_errors.csv','ik_dynamic.csv','ik_validation.png','ik_summary.json']);print(result);assert result['passed']
if __name__=='__main__':main_guard(main)
