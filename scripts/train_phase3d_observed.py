"""Bounded original PPO flow plus optional owner-thread RGB recorder."""
from _bootstrap import ROOT,destination,main_guard
import argparse,json,subprocess,sys,time,shutil
from pathlib import Path
import numpy as np
from src.common import write_json,write_csv,sha
from src.rl.dependencies import activate
activate()
import torch
from stable_baselines3 import PPO
from src.training_observer.observer import ObservedEnv,atomic_json
from scripts.phase3c_common import sources,read,rows
from scripts.train_phase3c import PilotPPO,Callback
from scripts.train_phase3b import parameter_groups,tensor_hash,delta

ACTIVE=None


class VisualPPO(PilotPPO):
    def train(self):
        env=ACTIVE
        atomic_json(env.dest/'live_status.json',dict(status='UPDATING ACTOR / CRITIC',global_decision=self.num_timesteps,
            policy_version=env.policy_version,episode=env.episode,episode_return=env.c_return))
        print('UPDATING ACTOR / CRITIC',self.num_timesteps,flush=True)
        super().train();env.policy_version=len(self.audit_events)
        rb=self.rollout_buffer;history=env.actor_history[-256:]
        self.audit_events[-1].update(gae_quantiles=np.quantile(rb.advantages,[0,.05,.5,.95,1]).tolist(),
            value_target_quantiles=np.quantile(rb.returns,[0,.05,.5,.95,1]).tolist(),
            actor_mean_average=np.mean([r['mean'] for r in history],axis=0).tolist(),
            actor_std_average=np.mean([r['std'] for r in history],axis=0).tolist(),
            active_episode=env.episode,active_episode_return=env.c_return,episode_state='partial episode may cross update')
        write_json(Path(self.audit_dest)/'ppo_update_events.json',self.audit_events)
        write_csv(Path(self.audit_dest)/'ppo_update_events.csv',self.audit_events)
        self.save(str(Path(self.audit_dest)/f'checkpoints/ppo_after_update_{env.policy_version:02d}'))
        atomic_json(env.dest/'live_status.json',dict(status='COLLECTING EXPERIENCE',global_decision=self.num_timesteps,policy_version=env.policy_version,episode=env.episode))
        print('COLLECTING EXPERIENCE / policy',env.policy_version,flush=True)


class VisualCallback(Callback):
    def __init__(self,env,dest):super().__init__(env,dest);self.pre=[];self.post=[]
    def _on_step(self):
        self.pre.append(self.locals['obs_tensor'].cpu().numpy()[0].copy());self.post.append(self.locals['new_obs'][0].copy())
        result=super()._on_step()
        return result and not (self.raw.dest/'abort_training.request').exists()


def job(dest,args,record,render):
    global ACTIVE
    dest.mkdir();(dest/'checkpoints').mkdir();torch.set_num_threads(1)
    env=ObservedEnv(sources('train'),dest/'actual_training',version=args.reward,seed=args.seed,mode='training',
        record=record,render_training=render,record_every=args.record_every_n_episodes,max_recorded=args.max_recorded_episodes,record_fps=args.fps)
    env.actor_history=[];ACTIVE=env;display=None
    if render:
        display=subprocess.Popen([sys.executable,str(ROOT/'scripts/watch_ppo_training.py'),'--live-stream',str(env.dest)],stdout=(dest/'display.log').open('w'),stderr=subprocess.STDOUT)
    cfg=read(ROOT/'configs/phase3b_smoke.json');options={k:v for k,v in cfg['ppo'].items() if k not in ('net_arch','initial_log_std')}
    model=VisualPPO('MlpPolicy',env,seed=args.seed,device='cpu',verbose=0,tensorboard_log=str(dest/'tensorboard'),
        policy_kwargs=dict(net_arch=dict(pi=[64,64],vf=[64,64]),log_std_init=cfg['ppo']['initial_log_std']),**options)
    with torch.no_grad():model.policy.action_net.weight.zero_();model.policy.action_net.bias.zero_()
    model.audit_events=[];model.audit_dest=str(dest);initial=parameter_groups(model)
    def capture_intent(module,inputs,output):
        with torch.no_grad():dist=module.get_distribution(inputs[0]).distribution
        raw=output[0].detach().cpu().numpy()[0].copy()
        env.intent=dict(raw=raw.tolist(),clipped=np.clip(raw,-1,1).tolist(),mean=dist.mean.cpu().numpy()[0].tolist(),std=dist.stddev.cpu().numpy()[0].tolist())
        env.actor_history.append(dict(env.intent))
    hook=model.policy.register_forward_hook(capture_intent);model.save(str(dest/'checkpoints/initial'));cb=VisualCallback(env,dest)
    write_json(dest/'frozen_config.json',dict(arguments=vars(args),record=record,render=render,ppo=cfg['ppo'],source=sources('train'),
        source_hashes={str(p.relative_to(ROOT)):sha(p) for directory in ('src/rl','src/experiments','src/training_observer') for p in (ROOT/directory).glob('*.py')}))
    snapshot=dest/'observer_source_snapshot';snapshot.mkdir()
    for file in [ROOT/'scripts/train_phase3d_observed.py',ROOT/'scripts/watch_ppo_training.py',*list((ROOT/'src/training_observer').glob('*.py'))]:shutil.copy2(file,snapshot/file.name)
    start=time.perf_counter();error=None;before_close=0
    try:
        print('COLLECTING EXPERIENCE',dest,flush=True);model.learn(total_timesteps=args.decisions,callback=cb,tb_log_name='actual',progress_bar=False)
        model.logger.dump(model.num_timesteps);before_close=env.total_physics_steps
    except BaseException as exc:
        error=repr(exc);write_json(dest/'training_error.json',dict(error=error));raise
    finally:
        env.active_decision=0;env.close();hook.remove()
        atomic_json(env.dest/'live_status.json',dict(status='TRAINING STOPPED' if (env.dest/'abort_training.request').exists() else 'TRAINING COMPLETE',global_decision=model.num_timesteps,policy_version=env.policy_version))
        (env.dest/'stream_complete').write_text('complete',encoding='utf-8')
    model.save(str(dest/'checkpoints/final'));write_csv(dest/'training_samples.csv',cb.samples)
    np.savez_compressed(dest/'observation_sequences.npz',pre=np.array(cb.pre),post=np.array(cb.post))
    after=parameter_groups(model);np.savez_compressed(dest/'parameters.npz',**{k:v.cpu().numpy() for k,v in model.policy.state_dict().items()})
    torch.save(model.policy.optimizer.state_dict(),dest/'optimizer.pt')
    r=dict(run=str(dest),record=record,render=render,decisions=model.num_timesteps,physics_total=env.total_physics_steps,
        reset_physics=env.total_reset_physics_steps,decision_physics=sum(x['physical_steps'] for x in cb.samples),cleanup_physics=env.total_physics_steps-before_close,
        initial_actor_sha256=tensor_hash(initial[0]),initial_critic_sha256=tensor_hash(initial[1]),final_actor_sha256=tensor_hash(after[0]),final_critic_sha256=tensor_hash(after[1]),
        actor_delta_l2=delta(initial[0],after[0]),critic_delta_l2=delta(initial[1],after[1]),updates=model.audit_events,
        episodes=env.episode_results,record_errors=env.record_errors,wall_s=time.perf_counter()-start,error=error,
        checkpoint=str(dest/'checkpoints/final.zip'),checkpoint_sha256=sha(dest/'checkpoints/final.zip'),
        stop_reason='user_abort' if (env.dest/'abort_training.request').exists() else 'decision_budget',display_pid=display.pid if display else None)
    write_json(dest/'complete.json',r);return r


def recursive_equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(recursive_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(recursive_equal(x,y) for x,y in zip(a,b))
    return a==b


def compare(a,b):
    da,db=Path(a['run']),Path(b['run']);x=rows(da/'training_samples.csv');y=rows(db/'training_samples.csv')
    assert x==y,'Observed stochastic samples/rewards differ'
    obs1=np.load(da/'observation_sequences.npz');obs2=np.load(db/'observation_sequences.npz')
    assert all(np.array_equal(obs1[k],obs2[k]) for k in obs1.files)
    params1=np.load(da/'parameters.npz');params2=np.load(db/'parameters.npz')
    assert all(np.array_equal(params1[k],params2[k]) for k in params1.files)
    assert recursive_equal(torch.load(da/'optimizer.pt',weights_only=True),torch.load(db/'optimizer.pt',weights_only=True))
    for ea,eb in zip(sorted((da/'actual_training').glob('episode_*')),sorted((db/'actual_training').glob('episode_*'))):
        assert rows(ea/'tracking.csv')==rows(eb/'tracking.csv')
        assert rows(ea/'decisions.csv')==rows(eb/'decisions.csv')
        sa,sb=rows(ea/'control_signals.csv'),rows(eb/'control_signals.csv')
        assert [{k:v for k,v in r.items() if k!='algorithm_s'} for r in sa]==[{k:v for k,v in r.items() if k!='algorithm_s'} for r in sb]
    assert a['physics_total']==b['physics_total']
    return dict(passed=True,decision_samples_exact=True,observations_exact=True,tracking_exact=True,control_signals_exact_except_wall_timing=True,
        rewards_done_exact=True,network_parameters_bitwise_equal=True,optimizer_state_bitwise_equal=True,
        update_steps_off=[u['decision_steps'] for u in a['updates']],update_steps_on=[u['decision_steps'] for u in b['updates']],
        final_checkpoint_zip_hash_equal=a['checkpoint_sha256']==b['checkpoint_sha256'],
        zip_caveat='archive metadata/paths may differ; every policy tensor and Adam state compared exactly',off=a,on=b)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--decisions',type=int,choices=[256,512,1024],default=1024)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--reward',choices=['R0','R1'],default='R1');p.add_argument('--compare-observer',action='store_true')
    p.add_argument('--render-training',action='store_true');p.add_argument('--record-training-rollouts',action='store_true')
    p.add_argument('--record-every-n-episodes',type=int,default=1);p.add_argument('--max-recorded-episodes',type=int,default=4);p.add_argument('--fps',type=int,choices=[5,10],default=10)
    p.add_argument('--reference-off',type=Path,help='Reuse an already archived matching observer-off run for recorder-fix verification')
    a=p.parse_args()
    if a.record_every_n_episodes<1 or not 0<=a.max_recorded_episodes<=8:raise ValueError('Bounded positive recording schedule required')
    dest=destination('phase3d_observation')
    if a.compare_observer or a.reference_off:
        if a.reference_off:
            off=read(a.reference_off/'complete.json');cfg=read(a.reference_off/'frozen_config.json')
            assert not off['record'] and off['decisions']==a.decisions and cfg['arguments']['seed']==a.seed and cfg['arguments']['reward']==a.reward
        else:off=job(dest/'observer_off',a,False,False)
        on=job(dest/'observer_on',a,True,a.render_training)
        result=compare(off,on);write_json(dest/'equivalence.json',result)
    else:result=job(dest/'training',a,a.record_training_rollouts or a.render_training,a.render_training)
    write_json(dest/'complete.json',result);write_json(ROOT/'results/phase3d_training_latest.json',dict(run=dest));print('COMPLETE',dest,flush=True)


if __name__=='__main__':main_guard(main)
