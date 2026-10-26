from _bootstrap import ROOT,destination,main_guard
from src.rl.dependencies import activate
activate()
import numpy as np
import imageio.v2 as imageio
from src.phase3dc.common import *
from src.phase3dc.environment import design
from src.training_observer.observer import annotate,numbers
from scripts.evaluate_phase3dc import job
import argparse,shutil


def main():
    p=argparse.ArgumentParser();p.add_argument('--resume',type=Path);a=p.parse_args()
    d=a.resume if a.resume else destination('phase3dc_video')
    if a.resume and (d/'phase3dc_policy_comparison.mp4').exists() and not (d/'partial_comparison_before_path_fix.mp4').exists():shutil.copy2(d/'phase3dc_policy_comparison.mp4',d/'partial_comparison_before_path_fix.mp4')
    gate=read(ROOT/'results/phase3dc_2048_gate.json');label='continue' if gate['all_four_passed'] else 'initial'
    train=read(Path(read(ROOT/f'results/phase3dc_training_{label}_latest.json')['run'])/'complete.json')['results']
    from PIL import Image,ImageDraw
    training_frames=0;validations=[]
    # Fixed schedule: FIRST original-training episode from every method/seed.
    initial=read(Path(read(ROOT/'results/phase3dc_training_initial_latest.json')['run'])/'complete.json')['results']
    with imageio.get_writer(str(d/'phase3dc_actual_training.mp4'),fps=10,codec='libx264',macro_block_size=16) as writer:
        for run in initial:
            folder=Path(run['run_directory'])/'actual_training/episode_0000';frames=read(folder/'rgb_frames.json');z=np.load(folder/'integration_states.npz');tracking=csvrows(folder/'tracking.csv')
            with imageio.get_reader(str(folder/'actual_rgb.mp4')) as reader:
                for i,meta in enumerate(frames):
                    j=meta['physics_step']-1;assert np.array_equal(z['qpos'][j],meta['qpos']) and np.array_equal(z['ctrl'][j],meta['ctrl'])
                    assert abs(float(tracking[j]['time'])-meta['time'])<1e-12
                    lines=[f'ACTUAL TRAINING | STOCHASTIC | method {run["method"]} seed {run["seed"]}',
                        f'Episode0 | decision {meta["global_decision"]} | policy update {meta["policy_version"]} | time {meta["time"]:.3f}s',
                        'Actor raw Gaussian: '+numbers(meta['raw_action']),'clipped: '+numbers(meta['clipped_action']),
                        'combined target rad: '+numbers(meta['target_residual']),'ACTUAL applied rad: '+numbers(meta['applied_residual']),
                        'q_ref: '+numbers(meta['q_ref']),'q_actual: '+numbers(meta['q_actual']),
                        f'TCP {meta["tcp_error_m"]*1000:.4f} mm | actual gap {meta["minimum_clearance_m"]*1000:.4f} mm | failure threshold5mm',
                        'last completed decision reward: '+str(meta['last_completed_reward']),
                        'Real live integration RGB; policy may update MID-episode; no deterministic re-predict',
                        'First episode fixed BEFORE outcome inspection; terminal status: '+str(read(folder/'episode_physics_summary.json')['failure_reason'] or 'SUCCESS')]
                    writer.append_data(annotate(reader.get_data(i),lines));training_frames+=1
            validations.append(dict(method=run['method'],seed=run['seed'],frames=len(frames),states_match=True,policy_versions=sorted({f['policy_version'] for f in frames}),joint_displacement_rad=float(np.max(np.ptp(z['qpos'][:,:7],axis=0)))))
            z.close()
    source=sources('test')[0];condition=design()['test_conditions'][0];seed=design()['test_seed'];methods=[(m,None) for m in ('M3','Integral500','Integral20','BC85')]+[(r['method']+'_seed'+str(r['seed']),r['checkpoint']) for r in train]
    pointer=ROOT/'results/phase3dc_evaluation_final_latest.json'
    formal_run=Path(read(pointer)['run']) if pointer.exists() else sorted((ROOT/'results/runs').glob('phase3dc_evaluation_final_*'))[-1]
    assert (formal_run/'final_test_opening.json').exists()
    runs=[];comparison_frames=0
    with imageio.get_writer(str(d/'phase3dc_policy_comparison.mp4'),fps=10,codec='libx264',macro_block_size=16) as writer:
        for name,checkpoint in methods:
            candidates=[d/name/'episode_0000/dc_summary.json',d/name/'bc_summary.json']
            cache=next((p for p in candidates if p.exists()),None)
            r=read(cache) if cache else job((source,condition,seed,name,checkpoint,str(d/name),True));folder=Path(r['run_directory'])
            formal_folder=formal_run/f'{source.stem}_{condition["name"]}_{name}'
            if name.startswith(('A_','B_')):formal_folder=formal_folder/'episode_0000'
            assert (formal_folder/'tracking.csv').exists()
            assert r['source']==str(source) and r['condition']==condition and r['physical_seed']==seed
            assert csvrows(folder/'tracking.csv')==csvrows(formal_folder/'tracking.csv')
            meta=read(folder/'bc_video_frames.json');z=np.load(folder/'integration_states.npz')
            for f in meta:assert np.array_equal(z['qpos'][f['physics_step']-1,:7],f['q_actual'])
            with imageio.get_reader(str(folder/'bc_live.mp4')) as reader:
                for frame in reader:
                    im=Image.fromarray(frame);draw=ImageDraw.Draw(im);draw.rectangle((0,0,960,22),fill='#132133');draw.text((8,3),'BASELINE COMPARISON / CHECKPOINT EVALUATION | '+name,fill='white');writer.append_data(np.asarray(im));comparison_frames+=1
            z.close();runs.append(dict(method=name,result=r,tracking_matches_formal=True,frames=len(meta)))
    for name in ('phase3dc_actual_training.mp4','phase3dc_policy_comparison.mp4'):shutil.copy2(d/name,ROOT/'assets/videos'/name)
    result=dict(training_frames=training_frames,comparison_frames=comparison_frames,training_sources=validations,comparison_runs=runs,
        actual_training_video_sha256=sha(d/'phase3dc_actual_training.mp4'),comparison_video_sha256=sha(d/'phase3dc_policy_comparison.mp4'))
    write_json(d/'complete.json',result);write_json(ROOT/'results/phase3dc_video_latest.json',dict(run=d));print('VIDEOS COMPLETE',d,flush=True)

if __name__=='__main__':main_guard(main)
