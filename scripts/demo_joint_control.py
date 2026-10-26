from _bootstrap import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.robot.model_loader import load_model
from src.robot.panda_interface import PandaInterface
from src.robot.joint_controller import JointPositionController
from src.visualization.viewer import LiveViewer,Recorder
from src.common import write_csv
def main():
    a=parser('Real seven-joint position-servo dynamics, never qpos playback').parse_args();dest=destination('demo_joint_control');m,d=load_model();robot=PandaInterface(m,d);controller=JointPositionController(robot);home=robot.positions()
    stages=[('hold',home,2.),('single_joint',home+np.array([.12,0,0,0,0,0,0]),3.),('seven_joint',home+np.array([-.10,-.12,.1,-.12,.1,.12,.1]),4.)]
    live=LiveViewer(robot,a.viewer,a.speed);video=Recorder(robot,dest,'joint_control',a.record);rows=[];summary=[]
    try:
        for name,target,seconds in stages:
            start_q=robot.positions();start_time=float(d.time)
            for _ in range(round(seconds/m.opt.timestep)):
                if not live.running():break
                with live.lock():
                    row=controller.step(target);row['phase']=name;rows.append(row);video.capture(name)
                live.sync()
            summary.append(dict(phase=name,start_time=start_time,end_time=float(d.time),target_rad=target.tolist(),actual_rad=robot.positions().tolist(),displacement_rad=(robot.positions()-start_q).tolist(),max_abs_tracking_error_rad=float(np.abs(robot.positions()-target).max()),rms_tracking_error_rad=float(np.sqrt(np.mean((robot.positions()-target)**2)))))
    finally:
        write_csv(dest/'joint_control.csv',rows);view=live.close();v=video.close()
    fig,axs=plt.subplots(4,2,figsize=(12,11));time_axis=np.array([r['time'] for r in rows])
    for i,ax in enumerate(axs.flat):
        if i==7:ax.axis('off');continue
        j=i+1;ax.plot(time_axis,[r[f'q{j}'] for r in rows],label='actual');ax.plot(time_axis,[r[f'target{j}'] for r in rows],'--',label='goal');ax.plot(time_axis,[r[f'command{j}'] for r in rows],':',label='slew-limited ctrl');ax.set(xlabel='Simulation seconds',ylabel='rad',title=f'joint{j}');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(dest/'joint_tracking.png',dpi=150);plt.close(fig)
    result=dict(passed=len(rows)==4500 and all(s['max_abs_tracking_error_rad']<.02 for s in summary),physics_steps=controller.physics_steps,simulation_seconds=float(d.time),qpos_control_writes=0,initialization='single mj_resetDataKeyframe before first step',stages=summary,viewer=view,video=v)
    write_json(dest/'joint_control_summary.json',result);write_json(dest/'complete.json',result);publish(dest,['joint_control.csv','joint_tracking.png','joint_control_summary.json']);print(result);assert result['passed']
if __name__=='__main__':main_guard(main)
