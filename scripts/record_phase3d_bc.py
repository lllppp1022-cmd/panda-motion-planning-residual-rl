"""Five fixed-case actual dynamical reruns; verify video does not change final evaluation."""
from _bootstrap import ROOT,destination,main_guard
from pathlib import Path
import shutil,hashlib
import numpy as np
import imageio.v2 as imageio
from scripts.phase3c_common import read,rows
from src.bc_residual.core import design
from src.bc_residual.runner import run_case
from src.common import write_json,sha


def main():
    final=Path(read(ROOT/'results/phase3d_bc_eval_final_latest.json')['run']);records=read(final/'complete.json')['results'];dest=destination('phase3d_bc_video');outputs=[];count=0
    for method in design()['methods']:
        original=next(r for r in records if r['scene']=='bc_final_box' and r['condition']['name']=='gain88' and r['method']==method)
        r=run_case((original['source'],original['condition'],original['physical_seed'],method,original['checkpoint'],str(dest/method)),record_video=True)
        t=rows(Path(r['run_directory'])/'tracking.csv');old=rows(Path(original['run_directory'])/'tracking.csv');assert t==old,'Recording changed dynamical execution'
        frames=read(dest/method/'bc_video_frames.json');z=np.load(dest/method/'integration_states.npz')
        for f in frames:
            row=t[f['physics_step']-1];index=f['physics_step']-1
            assert f['time']==float(row['time']) and np.array_equal(f['q_actual'],z['qpos'][index,:7])
            assert np.array_equal(f['ctrl'],[float(row[f'ctrl_{j}']) for j in range(1,8)])
        with imageio.get_reader(str(dest/method/'bc_live.mp4')) as reader:decoded=sum(1 for _ in reader)
        assert decoded==len(frames)>10 and len({f['raw_rgb_sha256'] for f in frames})>5
        outputs.append(dict(method=method,result=r,frames=decoded,all_frames_match=True,tracking_exactly_matches_formal_evaluation=True,
            joint_displacement_rad=float(np.max(np.linalg.norm(np.array([f['q_actual'] for f in frames])-frames[0]['q_actual'],axis=1)))))
        print('LIVE VIDEO',method,decoded,r['success'],r['tcp_rmse_m'],flush=True)
    movie=dest/'phase3d_bc_vs_integral.mp4'
    with imageio.get_writer(str(movie),fps=10,codec='libx264',macro_block_size=16) as writer:
        for r in outputs:
            with imageio.get_reader(str(dest/r['method']/'bc_live.mp4')) as reader:
                for frame in reader:writer.append_data(frame);count+=1
    with imageio.get_reader(str(movie)) as reader:assert sum(1 for _ in reader)==count
    shutil.copy2(movie,ROOT/'assets/videos/phase3d_bc_vs_integral.mp4')
    write_json(dest/'complete.json',dict(results=outputs,frames=count,duration_s=count/10,video=str(movie),sha256=sha(movie),actual_video_execution_physics=sum(r['result']['physics_steps'] for r in outputs)))
    write_json(ROOT/'results/phase3d_bc_video_latest.json',dict(run=dest));print('VIDEO COMPLETE',dest,flush=True)


if __name__=='__main__':main_guard(main)
