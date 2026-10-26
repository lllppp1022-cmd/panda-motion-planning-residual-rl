from _bootstrap import ROOT,destination,main_guard
from src.rl.dependencies import activate
activate()
import argparse,concurrent.futures,pickle,random,time
import numpy as np
import torch
from src.phase3dc.common import *
from src.phase3dc.environment import DCEnv,design
from src.phase3dc.policy import make_model
from scripts.train_phase3c import PilotPPO,Callback
from scripts.train_phase3b import parameter_groups,delta,tensor_hash


class DCPPO(PilotPPO):
    def train(self):
        env=self.get_env().envs[0].unwrapped
        print('UPDATING ACTOR / CRITIC',self.num_timesteps,flush=True)
        super().train();env.policy_version=len(self.audit_events)
        self.audit_events[-1].update(gae_quantiles=np.quantile(self.rollout_buffer.advantages,[0,.05,.5,.95,1]).tolist(),
            value_target_quantiles=np.quantile(self.rollout_buffer.returns,[0,.05,.5,.95,1]).tolist())
        write_json(Path(self.audit_dest)/'ppo_update_events.json',self.audit_events);write_csv(Path(self.audit_dest)/'ppo_update_events.csv',self.audit_events)
        print('COLLECTING EXPERIENCE',self.num_timesteps,flush=True)


class DCCallback(Callback):
    def _on_step(self):
        keep=super()._on_step();info=self.locals['infos'][0]
        self.samples[-1].update({k:info[k] for k in ('actor_request','integral20_request','correction_target','composed_target','effective_correction')})
        return keep


def job(args):
    method,seed,folder,prior=args;folder=Path(folder);folder.mkdir();(folder/'checkpoints').mkdir();torch.set_num_threads(1)
    offset=read(Path(prior)/'complete.json')['next_episode_offset'] if prior else 0
    env=DCEnv(sources('train'),folder/'actual_training',method=method,seed=seed,episode_offset=offset,mode='training',record=True,
        record_every=3,max_recorded=2,record_fps=10)
    if prior:
        old=read(Path(prior)/'complete.json');model=DCPPO.load(old['checkpoint'],env=env,device='cpu',tensorboard_log=str(folder/'tensorboard'))
        with (Path(prior)/'rng_state.pkl').open('rb') as f:rng=pickle.load(f)
        np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch']);random.setstate(rng['python'])
    else:model=make_model(env,seed,bc_checkpoint(),method,cls=DCPPO,tensorboard_log=str(folder/'tensorboard'));model.audit_events=[]
    env.policy_version=len(model.audit_events)
    if prior:env.total_decisions=model.num_timesteps
    model.audit_dest=str(folder);initial=parameter_groups(model);cb=DCCallback(env,folder)
    def intent(module,inputs,outputs):
        with torch.no_grad():dist=module.get_distribution(inputs[0]).distribution
        raw=outputs[0].detach().cpu().numpy()[0]
        env.intent=dict(raw=raw.tolist(),clipped=np.clip(raw,-1,1).tolist(),mean=dist.mean.cpu().numpy()[0].tolist(),std=dist.stddev.cpu().numpy()[0].tolist())
    hook=model.policy.register_forward_hook(intent);model.save(str(folder/'checkpoints/initial'));started=time.perf_counter()
    write_json(folder/'frozen_config.json',dict(config=design(),method=method,seed=seed,bc_checkpoint_sha256=sha(bc_checkpoint()),source_hashes=code_hashes(),prior=prior))
    try:
        model.learn(total_timesteps=2048,reset_num_timesteps=not bool(prior),callback=cb,tb_log_name=method,progress_bar=False);model.logger.dump(model.num_timesteps);before_close=env.total_physics_steps
    except BaseException as exc:
        write_json(folder/'training_failure.json',dict(error=repr(exc),collected=len(cb.samples)));write_csv(folder/'training_samples.csv',cb.samples);raise
    finally:env.active_decision=0;env.close();hook.remove()
    model.save(str(folder/'checkpoints/final'));after=parameter_groups(model)
    with (folder/'rng_state.pkl').open('wb') as f:pickle.dump(dict(numpy=np.random.get_state(),torch=torch.get_rng_state(),python=random.getstate()),f)
    write_csv(folder/'training_samples.csv',cb.samples)
    r=dict(method=method,seed=seed,run_directory=str(folder),stage_decisions=len(cb.samples),cumulative_decisions=model.num_timesteps,
        total_physics=env.total_physics_steps,reset_physics=env.total_reset_physics_steps,decision_physics=sum(s['physical_steps'] for s in cb.samples),cleanup_physics=env.total_physics_steps-before_close,
        actor_delta_l2=delta(initial[0],after[0]),critic_delta_l2=delta(initial[1],after[1]),initial_actor_sha256=tensor_hash(initial[0]),final_actor_sha256=tensor_hash(after[0]),
        updates=model.audit_events[-8:],completed=cb.completed,episode_results=env.episode_results,next_episode_offset=offset+env.episode+1,record_errors=env.record_errors,
        checkpoint=str(folder/'checkpoints/final.zip'),checkpoint_sha256=sha(folder/'checkpoints/final.zip'),wall_s=time.perf_counter()-started,source_hashes=code_hashes())
    write_json(folder/'complete.json',r);print('COMPLETE',method,seed,r['total_physics'],flush=True);return r


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['initial','continue'],default='initial');a=p.parse_args()
    assert read(Path(read(ROOT/'results/phase3dc_interface_latest.json')['run'])/'complete.json')['passed']
    prior=[]
    if a.stage=='continue':
        gate=read(ROOT/'results/phase3dc_2048_gate.json');assert gate['all_four_passed'],'common gate failed; no continuation'
        prior=read(Path(gate['training_run'])/'complete.json')['results'];assert all(r['source_hashes']==code_hashes() for r in prior)
    dest=destination('phase3dc_training_'+a.stage);jobs=[]
    for method in design()['methods']:
        for seed in design()['seeds']:
            old=next((r['run_directory'] for r in prior if r['method']==method and r['seed']==seed),None)
            jobs.append((method,seed,str(dest/f'{method}_seed{seed}'),old))
    write_json(dest/'before_training.json',dict(jobs=jobs,source_hashes=code_hashes(),budget_per_group=2048));results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        for r in pool.map(job,jobs):results.append(r)
    write_json(dest/'complete.json',dict(results=results,stage=a.stage));write_json(ROOT/f'results/phase3dc_training_{a.stage}_latest.json',dict(run=dest))

if __name__=='__main__':main_guard(main)
