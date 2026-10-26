"""Validate real RGB against physical logs and construct annotated actual-training video."""
from _bootstrap import ROOT,destination,main_guard
from pathlib import Path
import json,hashlib,shutil
import numpy as np
import imageio.v2 as imageio
from src.common import sha,write_json,write_csv
from scripts.phase3c_common import read,rows
from src.training_observer.replay import RecordedEpisode
from src.training_observer.observer import annotate


def validate(on,dest):
    records=[];frame_csv=[];output=dest/'phase3d_actual_training.mp4';count=0
    with imageio.get_writer(str(output),fps=10,codec='libx264',macro_block_size=16) as writer:
        for episode,folder in enumerate(sorted((on/'actual_training').glob('episode_*'))):
            if not (folder/'actual_rgb.mp4').exists():continue
            data=RecordedEpisode(on,episode);f=read(folder/'rgb_frames.json');states=np.load(folder/'integration_states.npz');t=data.t;u=data.u
            assert len(states['states'])==len(t)
            assert np.array_equal(states['qpos'][:,data.qindices],np.array([[float(r[f'q_actual_{j}']) for j in range(1,8)] for r in t]))
            assert np.array_equal(states['qvel'][:,data.dindices],np.array([[float(r[f'dq_actual_{j}']) for j in range(1,8)] for r in t]))
            assert np.array_equal(states['ctrl'][:,data.aindices],np.array([[float(r[f'ctrl_{j}']) for j in range(1,8)] for r in t]))
            decoded=[];joint=[];sample_hash=[];versions=set();starts=count
            with imageio.get_reader(str(folder/'actual_rgb.mp4')) as reader:
                for i,rgb in enumerate(reader):
                    meta=f[i];index=int(meta['physics_step'])-1;r=t[index];info=data.information(index);a=info['action'];signal=u[index]
                    assert meta['time']==float(r['time']) and meta['tcp_error_m']==float(r['tcp_error_m'])
                    assert meta['minimum_clearance_m']==float(r['minimum_clearance_m'])
                    assert np.array_equal(meta['qpos'],states['qpos'][index]) and np.array_equal(meta['ctrl'],states['ctrl'][index])
                    assert np.array_equal(meta['applied_residual'],[float(signal[f'residual_rad_{j}']) for j in range(1,8)])
                    assert np.array_equal(meta['target_residual'],[float(signal[f'residual_target_rad_{j}']) for j in range(1,8)])
                    if a:
                        assert meta['global_decision']==a['global_decision'] and meta['policy_version']==a['policy_version']
                        assert np.array_equal(meta['raw_action'],json.loads(a['raw_sample_action']))
                        assert np.array_equal(meta['clipped_action'],json.loads(a['normalized_executed_action']))
                        versions.add(meta['policy_version'])
                    else:assert not meta['global_decision']
                    lines=data.lines(index);lines[0]='ACTUAL STOCHASTIC TRAINING | TRAINING-SITE RGB (logged reward annotation)'
                    writer.append_data(annotate(rgb,lines));decoded.append(hashlib.sha256(rgb.tobytes()).hexdigest());joint.append(meta['q_actual']);count+=1
                    sample_hash.append(meta['raw_rgb_sha256']);frame_csv.append(dict(output_frame=count-1,episode=episode,local_frame=i,time=meta['time'],physics_step=meta['physics_step'],
                        global_decision=meta['global_decision'],policy_version=meta['policy_version'],raw_action=meta['raw_action'],clipped_action=meta['clipped_action'],
                        target_residual=meta['target_residual'],applied_residual=meta['applied_residual'],tcp_error_m=meta['tcp_error_m'],minimum_clearance_m=meta['minimum_clearance_m'],
                        resolved_interval_reward=a['interval_reward'] if a else None,episode_outcome=info['status'],full_state_saved=True))
            assert len(decoded)==len(f)>10
            movement=float(np.max(np.linalg.norm(np.array(joint)-joint[0],axis=1)))
            assert len(set(decoded))>5 and len(set(sample_hash))>5 and movement>.01
            for i in range(0,len(t),100):
                data.set_state(i);expected=np.array([float(t[i][f'tcp_actual_{j}']) for j in range(3)])
                assert np.linalg.norm(data.data.site_xpos[data.site]-expected)<1e-12
            records.append(dict(episode=episode,frames=len(f),duration_s=len(f)/10,simulation_duration_s=float(t[-1]['time']),
                decoded_unique_frames=len(set(decoded)),raw_unique_frames=len(set(sample_hash)),joint_displacement_rad=movement,policy_versions=sorted(versions),
                output_first_frame=starts,outcome=data.summary['failure_reason'] or 'success',all_frames_match_actual_logs=True))
            data.close();states.close()
    with imageio.get_reader(str(output)) as reader:n=sum(1 for _ in reader)
    assert n==count
    write_csv(dest/'actual_video_frames.csv',frame_csv);shutil.copy2(output,ROOT/'assets/videos/phase3d_actual_training.mp4')
    return dict(episodes=records,total_frames=count,duration_s=count/10,output=str(output),published=str(ROOT/'assets/videos/phase3d_actual_training.mp4'),
        sha256=sha(output),bytes=output.stat().st_size,decoded_frames=n,full_state_and_control_match=True,
        rgb_lossy_note='MP4 is lossy: decoded pixels are not claimed bitwise identical to pre-encode RGB; original RGB SHA and physical frame association retained.')


def plots(on):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    samples=rows(on/'training_samples.csv');x=np.array([int(r['decision']) for r in samples]);raw=np.array([json.loads(r['raw_sample_action']) for r in samples]);times=[];target=[];applied=[];tcp=[];rew=[];base=0
    for ep in sorted((on/'actual_training').glob('episode_*')):
        data=RecordedEpisode(on,int(ep.name[-4:]));t=data.t;u=data.u
        for r,s in zip(t,u):
            times.append(base+float(r['time']));target.append([float(s[f'residual_target_rad_{j}']) for j in range(1,8)]);applied.append([float(s[f'residual_rad_{j}']) for j in range(1,8)]);tcp.append(float(r['tcp_error_m']))
        base+=float(t[-1]['time']);data.close()
    dest=ROOT/'assets/plots';fig,axes=plt.subplots(3,1,figsize=(12,8),constrained_layout=True)
    axes[0].plot(x,raw,lw=.4);axes[0].set(ylabel='Raw normalized actions (7)',xlabel='PPO decision')
    axes[1].plot(times,np.array(target)*1000,lw=.45);axes[1].set(ylabel='Target residual (mrad)',xlabel='Cumulative physical simulation time (s)')
    axes[2].plot(times,np.array(applied)*1000,lw=.45);axes[2].set(ylabel='Applied residual (mrad)',xlabel='Cumulative physical simulation time (s)')
    fig.suptitle('ACTUAL STOCHASTIC TRAINING: action -> bounded target -> slew-limited residual');fig.savefig(dest/'phase3d_action_vs_applied.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,6),constrained_layout=True);updates=read(on/'ppo_update_events.json')
    axes[0].plot(x,[float(r['reward']) for r in samples],lw=.6,label='Actual selected R1 reward')
    for u in updates:axes[0].axvline(u['decision_steps'],color='red',ls='--',alpha=.7)
    axes[0].legend();axes[0].set(ylabel='Reward',title='COLLECT -> UPDATE ACTOR / CRITIC -> COLLECT (4 actual updates)')
    d=[u['decision_steps'] for u in updates];axes[1].plot(d,[u['actor_parameter_delta_l2'] for u in updates],'o-',label='Actor delta');axes[1].plot(d,[u['critic_parameter_delta_l2'] for u in updates],'o-',label='Critic delta');axes[1].legend();axes[1].set(xlabel='Global PPO decision',ylabel='Parameter L2 change')
    fig.savefig(dest/'phase3d_ppo_update_timeline.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,4),constrained_layout=True);ax.plot(times,np.array(tcp)*1000,lw=.7);ax.set(xlabel='Cumulative actual physical simulation time (s)',ylabel='TCP error (mm)',title='Actual tracking including reset and budget hold; no success-only filtering');fig.savefig(dest/'phase3d_tcp_error.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True)
    for key in samples[0]:
        if key.startswith('reward_'):ax.plot(x,[float(r.get(key,0)) for r in samples],lw=.7,label=key[7:])
    ax.legend(ncol=5,fontsize=8);ax.set(xlabel='PPO decision',ylabel='Actual R1 reward component');fig.savefig(dest/'phase3d_reward_components.png',dpi=140);plt.close(fig)
    # Independent evaluation of each actual update uses only frozen validation logs from Phase3B.
    pointer=read(ROOT/'results/phase3c_3b_updates_latest.json');evaluations=read(Path(pointer['run'])/'complete.json')['results'];fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True)
    for condition in ('nominal','mild_gain'):
        r=sorted([v for v in evaluations if v['condition']['name']==condition],key=lambda v:v['update']);ax.plot([v['update'] for v in r],[v['result']['tcp_rmse_m']*1000 for v in r],'o-',label=condition)
    ax.set(title='Checkpoint Evaluation: historical 3B updates (independent deterministic replay)',xlabel='Policy update',ylabel='TCP trajectory RMSE (mm)');ax.legend();fig.savefig(dest/'phase3d_independent_evaluation.png',dpi=140);plt.close(fig)


def main():
    root=Path(read(ROOT/'results/phase3d_training_latest.json')['run']);equivalence=read(root/'equivalence.json');assert equivalence['passed']
    assert equivalence['update_steps_off']==equivalence['update_steps_on']==[256,512,768,1024]
    on=Path(equivalence['on']['run']);preview_errors=equivalence['on']['record_errors']
    assert all('PermissionError' in e['error'] for e in preview_errors),'Unexpected capture error: inspect before acceptance'
    dest=destination('phase3d_validation');video=validate(on,dest);plots(on)
    history=Path(read(ROOT/'results/phase3d_history_latest.json')['run']);manifest=read(history/'history_sha256.json')
    assert all(sha(ROOT/name)==h for name,h in manifest.items())
    result=dict(passed=True,training_run=str(root),video=video,observer_equivalence=equivalence,historical_integrity_files=len(manifest),historical_unchanged=True,
        retained_live_preview_file_sharing_errors=preview_errors,preview_fix='Unique immutable PNG paths + bounded JSON replace retry. Verify separately without more PPO training.',
        historical_audit_run=str(history),actual_task_failures=[e for e in equivalence['on']['episodes'] if not e['success'] and e['failure_reason']!='rl_external_truncation'],
        budget_interrupted_episodes=sum(e['failure_reason']=='rl_external_truncation' for e in equivalence['on']['episodes']))
    write_json(dest/'complete.json',result);write_json(ROOT/'results/phase3d_recording_validation.json',result);print('VALIDATED',dest,video['total_frames'],flush=True)


if __name__=='__main__':main_guard(main)
