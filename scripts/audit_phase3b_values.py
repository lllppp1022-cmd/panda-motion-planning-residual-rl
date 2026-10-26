"""Reconstruct historical observations/GAE and replay optimizer offline, never new interaction."""
from _bootstrap import ROOT,destination,main_guard
import csv,json
from pathlib import Path
import numpy as np
from src.common import write_json,write_csv
from src.rl.dependencies import activate
activate()
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from src.planning.collision_checker import CollisionChecker
from src.experiments.phase3c import reward_r1

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def rows(p):
    with Path(p).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def main():
    dest=destination('phase3c_3b_audit');old=Path(read(ROOT/'results/phase3b_acceptance.json')['training_run']);samples=rows(old/'training_samples.csv')
    cfg=read(old/'frozen_training_config.json')['config'];reward_cfg=cfg['reward'];limits=CollisionChecker().robot.limits;middle=limits.mean(axis=1);half=np.diff(limits,axis=1).ravel()/2
    records=[];effects=[];obslist=[];postlist=[];starts=[]
    for folder in sorted((old/'actual_training').glob('episode_*')):
        t=rows(folder/'tracking.csv');u=rows(folder/'control_signals.csv');d=rows(folder/'decisions.csv');duration=read(folder/'frozen_run_config.json')['trajectory_duration_s']
        v=lambda x,k:np.array([float(x[f'{k}_{j}']) for j in range(1,8)])
        initial=np.r_[(v(u[749],'observed_q')-middle)/half,v(u[749],'observed_velocity'),(v(t[749],'q_ref')-middle)/half,v(t[749],'dq_ref')/.15,
            (v(u[749],'observed_q')-v(t[749],'q_ref'))/.05,(v(u[749],'base_target')-middle)/half,np.zeros(21),-1.,np.zeros(14)]
        observation=np.clip(initial,-1,1).astype(np.float32);previous=750;action0=np.zeros(7);progress0=0.;episode_indices=[]
        for decision in d:
            end=int(decision['physics_step']);segment=t[previous:end];signals=u[previous:end];action=np.array(json.loads(decision['action']))
            nums=[{k:float(value) if k not in ('phase','safety_valid') else value for k,value in r.items()} for r in segment]
            error=np.array([v(x,'q_actual')-v(x,'q_ref') for x in segment]);vel=np.array([v(x,'dq_actual')-v(x,'dq_ref') for x in segment])
            terminated=decision['terminated']=='True';truncated=decision['truncated']=='True';success=terminated and float(decision['reward_success'])>0
            parts=dict(joint=-reward_cfg['joint_max_penalty']*min(1.,float(np.mean((error/.01)**2))),velocity=-reward_cfg['velocity_max_penalty']*min(1.,float(np.mean((vel/.1)**2))),
                action=-reward_cfg['action_penalty']*float(np.mean(action**2)),smoothness=-reward_cfg['smoothness_max_penalty']*min(1.,float(np.mean((action-action0)**2))),
                time=-reward_cfg['time_penalty'],progress=reward_cfg['progress_total']*max(0.,float(segment[-1]['progress'])-progress0),
                success=reward_cfg['success_bonus'] if success else 0.,failure=-reward_cfg['failure_penalty'] if (terminated or truncated) and not success else 0.,
                contact=-reward_cfg['contact_penalty'] if terminated and float(decision['reward_contact'])<0 else 0.)
            for k,value in parts.items():assert abs(value-float(decision['reward_'+k]))<1e-10
            r1,r1parts=reward_r1(nums,action,action0,progress0,duration,success,terminated or truncated,parts['contact']<0)
            index=len(records);episode_indices.append(index)
            records.append(dict(decision=index+1,episode=folder.name,episode_decision=int(decision['decision']),reward_R0=sum(parts.values()),reward_R1_offline=r1,
                **{f'R0_{k}':x for k,x in parts.items()},**{f'R1_{k}':x for k,x in r1parts.items()},terminated=terminated,truncated=truncated,
                complete_episode_return_available=bool(d[-1]['terminated']=='True'),raw_gaussian_action=samples[index]['raw_gaussian_action']))
            applied=np.array([v(x,'residual_rad') for x in signals]);target=np.array([v(x,'residual_target_rad') for x in signals])
            prev_applied=np.array([v(u[j-1],'residual_rad') if j else np.zeros(7) for j in range(previous,end)])
            prev_ctrl=np.array([v(t[j-1],'ctrl') if j else read(folder/'initialization.json')['q'] for j in range(previous,end)])
            delayed=np.array([v(x,'delayed_target') for x in signals]);request=abs(target-prev_applied)>.01*.002+1e-12
            effects.append(dict(decision=index+1,episode=folder.name,physics_steps=len(segment),target_max_rad=float(np.max(abs(target))),applied_max_rad=float(np.max(abs(applied))),
                target_applied_rmse_rad=float(np.sqrt(np.mean((target-applied)**2))),residual_slew_intervention_physics_steps=int(np.sum(np.any(request,axis=1))),
                total_command_slew_intervention_physics_steps=int(np.sum(np.any(abs(delayed-prev_ctrl)>.35*.002+1e-12,axis=1))),
                tcp_rmse_m=float(np.sqrt(np.mean([float(x['tcp_error_m'])**2 for x in segment]))),joint_rmse_rad=float(np.sqrt(np.mean(error**2)))))
            obslist.append(observation.copy());observation=np.array(json.loads(decision['observation']),dtype=np.float32);postlist.append(observation.copy());starts.append(int(decision['decision'])==1)
            previous=end;action0=action;progress0=float(segment[-1]['progress'])
        discounted=0.
        for idx in reversed(episode_indices):
            discounted=records[idx]['reward_R0']+.995*discounted;records[idx]['observed_discounted_return_R0']=discounted
    assert len(records)==1024
    torch.set_num_threads(1);obs=np.array(obslist);rawactions=np.array([json.loads(x['raw_gaussian_action']) for x in samples],dtype=np.float32)
    original_events=read(old/'ppo_update_events.json');audits=[];checkpoints=[old/'checkpoints/ppo_initial_zero_mean.zip']+[old/f'checkpoints/ppo_after_update_{i:02d}.zip' for i in range(1,5)]
    for block in range(4):
        model=PPO.load(str(checkpoints[block]),device='cpu');model.set_logger(configure(folder=None,format_strings=[]));rb=model.rollout_buffer
        sl=slice(block*256,(block+1)*256);values=[];logp=[];means=[];stds=[]
        with torch.no_grad():
            for o,a in zip(obs[sl],rawactions[sl]):
                ot=torch.as_tensor(o[None]);at=torch.as_tensor(a[None]);vv,lp,_=model.policy.evaluate_actions(ot,at);dist=model.policy.get_distribution(ot).distribution
                values.append(float(vv.item()));logp.append(float(lp.item()));means.append(dist.mean.numpy()[0]);stds.append(dist.stddev.numpy()[0])
            last_value=model.policy.predict_values(torch.as_tensor(postlist[(block+1)*256-1][None]))
        rb.reset();rb.observations=obs[sl,None].copy();rb.actions=rawactions[sl,None].copy();rb.values=np.array(values,dtype=np.float32)[:,None]
        rb.log_probs=np.array(logp,dtype=np.float32)[:,None];rb.rewards=np.array([float(x['reward']) for x in samples[sl]],dtype=np.float32)[:,None]
        rb.episode_starts=np.array(starts[sl],dtype=np.float32)[:,None];rb.full=True;rb.pos=256
        rb.compute_returns_and_advantage(last_value,np.array([samples[(block+1)*256-1]['done']=='True']))
        advantages=rb.advantages.copy().ravel();targets=rb.returns.copy().ravel();mse=float(np.mean((targets-np.array(values))**2));ev=1-float(np.var(targets-np.array(values)))/float(np.var(targets))
        for j in range(256):records[block*256+j].update(update=block+1,critic_prediction=values[j],gae_advantage=float(advantages[j]),gae_value_target=float(targets[j]),
            actor_mean=json.dumps(means[j].tolist()),actor_std=json.dumps(stds[j].tolist()),old_action_log_prob=logp[j])
        np.savez_compressed(dest/f'update{block+1}_reconstructed_buffer.npz',observations=rb.observations,actions=rb.actions,rewards=rb.rewards,values=rb.values,
            advantages=rb.advantages,returns=rb.returns,log_probs=rb.log_probs,episode_starts=rb.episode_starts)
        # Replay the original seed42 permutation stream, not a fresh online run.
        rng=np.random.RandomState(42)
        for _ in range(block*5):rng.permutation(256)
        np.random.set_state(rng.get_state());model._current_progress_remaining=1-(block+1)/4;model.train()
        expected=PPO.load(str(checkpoints[block+1]),device='cpu')
        parameter_error=max(float(torch.max(abs(v-expected.policy.state_dict()[k]))) for k,v in model.policy.state_dict().items())
        replay_loss=float(model.logger.name_to_value['train/value_loss']);actual_loss=original_events[block]['value_loss']
        audits.append(dict(update=block+1,preoptimization_target_mse=mse,explained_variance_recomputed=ev,
            explained_variance_logged=original_events[block]['explained_variance'],advantage_mean=float(np.mean(advantages)),advantage_std=float(np.std(advantages)),
            advantage_min=float(np.min(advantages)),advantage_max=float(np.max(advantages)),max_value_target=float(np.max(targets)),
            logged_mean_minibatch_value_loss=actual_loss,replayed_mean_minibatch_value_loss=replay_loss,
            loss_abs_difference=abs(actual_loss-replay_loss),replayed_parameter_max_abs_error=parameter_error))
        print('AUDIT UPDATE',audits[-1],flush=True)
    write_csv(ROOT/'results/phase3c_reward_value_audit.csv',records);write_csv(ROOT/'results/phase3c_action_effectiveness.csv',effects)
    summary=dict(updates=audits,recomputed_decisions=1024,all_components_match=True,bootstrap='rollout-specific frozen pre-update critic, correct terminal masks',
        incomplete_MC='last episode observed discounted prefix only, no invented unseen future',offline_optimizer_replay=True,
        reward_R0_std=float(np.std([x['reward_R0'] for x in records])),reward_R1_std=float(np.std([x['reward_R1_offline'] for x in records])),
        residual_slew_steps=sum(x['residual_slew_intervention_physics_steps'] for x in effects),total_command_slew_steps=sum(x['total_command_slew_intervention_physics_steps'] for x in effects))
    write_json(dest/'complete.json',summary);write_json(ROOT/'results/phase3c_3b_audit_latest.json',dict(run=dest));print('COMPLETE',dest,flush=True)

if __name__=='__main__':main_guard(main)
