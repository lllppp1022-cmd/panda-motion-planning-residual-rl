"""Two seeds x two rewards, common 2048 gate, at most4096 decisions per group."""
if __package__:
    from scripts._bootstrap import ROOT,destination,main_guard
else:
    from _bootstrap import ROOT,destination,main_guard
import argparse,concurrent.futures,json,pickle,random,time
from pathlib import Path
import numpy as np
from src.common import write_json,write_csv,sha
from src.rl.dependencies import activate,provenance
activate()
import torch
from stable_baselines3.common.callbacks import BaseCallback
from src.experiments.phase3c import DiagnosticEnv,design
from scripts.phase3c_common import sources,read
from scripts.train_phase3b import AuditedPPO,parameter_groups,delta

def code_hashes():
    files=list((ROOT/'src/experiments').glob('*.py'))+[ROOT/'scripts/train_phase3c.py',ROOT/'scripts/phase3c_common.py',ROOT/'configs/phase3c_pilot.json']
    return {str(p.relative_to(ROOT)):sha(p) for p in files}

class PilotPPO(AuditedPPO):
    def train(self):
        index=len(self.audit_events)+1;rb=self.rollout_buffer
        np.savez_compressed(Path(self.audit_dest)/f'rollout_{index:02d}.npz',observations=rb.observations,actions=rb.actions,rewards=rb.rewards,
            values=rb.values,advantages=rb.advantages,returns=rb.returns,log_probs=rb.log_probs,episode_starts=rb.episode_starts)
        diagnostic=dict(advantage_mean=float(np.mean(rb.advantages)),advantage_std=float(np.std(rb.advantages)),
            advantage_min=float(np.min(rb.advantages)),advantage_max=float(np.max(rb.advantages)),
            target_min=float(np.min(rb.returns)),target_max=float(np.max(rb.returns)),
            preoptimization_value_mse=float(np.mean((rb.returns-rb.values)**2)))
        super().train();self.audit_events[-1].update(diagnostic)
        write_json(Path(self.audit_dest)/'ppo_update_events.json',self.audit_events);write_csv(Path(self.audit_dest)/'ppo_update_events.csv',self.audit_events)
        self.save(str(Path(self.audit_dest)/f'checkpoints/ppo_after_update_{index:02d}'))

class Callback(BaseCallback):
    def __init__(self,raw,dest):super().__init__();self.raw=raw;self.dest=dest;self.samples=[];self.completed=[]
    def _on_step(self):
        info=self.locals['infos'][0]
        with torch.no_grad():dist=self.model.policy.get_distribution(self.locals['obs_tensor']).distribution
        self.samples.append(dict(decision=self.num_timesteps,reward=float(self.locals['rewards'][0]),R0_reward=info['R0_reward'],R1_reward=info['R1_reward'],
            physical_steps=info['physical_steps_this_decision'],actual_physics_stage=self.raw.total_physics_steps,condition=info['condition_name'],
            done=bool(self.locals['dones'][0]),success=info['success'],reason=info['reason'],safety_violation=info['safety_violation'],prohibited_contact=info['prohibited_contact'],
            critic_prediction=float(self.locals['values'].item()),actor_mean=json.dumps(dist.mean.cpu().numpy()[0].tolist()),actor_std=json.dumps(dist.stddev.cpu().numpy()[0].tolist()),
            raw_sample_action=json.dumps(self.locals['actions'][0].tolist()),normalized_executed_action=json.dumps(self.locals['clipped_actions'][0].tolist()),
            **{f'reward_{k}':v for k,v in info['reward_components_selected'].items()}))
        if self.locals['dones'][0]:self.completed.append(dict(decision=self.num_timesteps,condition=info['condition_name'],success=info['success'],reason=info['reason']))
        if self.num_timesteps%256==0:
            write_csv(self.dest/'training_samples.csv',self.samples);print('COLLECTED',self.num_timesteps,self.raw.total_physics_steps,flush=True)
        return True

def training_job(args):
    version,seed,folder,prior=args;folder=Path(folder);(folder/'checkpoints').mkdir(parents=True);torch.set_num_threads(1)
    cfg3b=read(ROOT/'configs/phase3b_smoke.json');opts={k:v for k,v in cfg3b['ppo'].items() if k not in ('net_arch','initial_log_std')}
    offset=read(Path(prior)/'complete.json')['next_episode_offset'] if prior else 0
    env=DiagnosticEnv(sources('train'),folder/'actual_training',version=version,seed=seed,episode_offset=offset,mode='training')
    if prior:
        last=read(Path(prior)/'complete.json');model=PilotPPO.load(last['checkpoint'],env=env,device='cpu',tensorboard_log=str(folder/'tensorboard'))
        with (Path(prior)/'rng_state.pkl').open('rb') as f:rng=pickle.load(f)
        np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch']);random.setstate(rng['python'])
    else:
        model=PilotPPO('MlpPolicy',env,seed=seed,device='cpu',tensorboard_log=str(folder/'tensorboard'),verbose=0,
            policy_kwargs=dict(net_arch=dict(pi=[64,64],vf=[64,64]),log_std_init=cfg3b['ppo']['initial_log_std']),**opts)
        with torch.no_grad():model.policy.action_net.weight.zero_();model.policy.action_net.bias.zero_()
        model.audit_events=[]
    model.audit_dest=str(folder);before=parameter_groups(model);callback=Callback(env,folder);start=time.perf_counter()
    write_json(folder/'frozen_config.json',dict(version=version,seed=seed,config=design(),source_hashes=code_hashes(),prior_stage=prior,
        episode_offset=offset,stage_decision_budget=2048,dependencies=provenance()))
    model.save(str(folder/'checkpoints/stage_initial'))
    try:
        model.learn(total_timesteps=2048,reset_num_timesteps=not bool(prior),callback=callback,tb_log_name=f'{version}_{seed}',progress_bar=False)
        model.logger.dump(model.num_timesteps);physics_before_close=env.total_physics_steps
    except BaseException as exc:
        write_json(folder/'training_failure.json',dict(error=repr(exc),collected=len(callback.samples),source_hashes=code_hashes()))
        write_csv(folder/'training_samples.csv',callback.samples);raise
    finally:env.close()
    after=parameter_groups(model);checkpoint=folder/f'checkpoints/ppo_{model.num_timesteps}.zip';model.save(str(checkpoint))
    with (folder/'rng_state.pkl').open('wb') as f:pickle.dump(dict(numpy=np.random.get_state(),torch=torch.get_rng_state(),python=random.getstate()),f)
    write_csv(folder/'training_samples.csv',callback.samples)
    updates=model.audit_events[-8:]
    result=dict(version=version,seed=seed,stage_decisions=len(callback.samples),cumulative_decisions=model.num_timesteps,
        stage_training_physics=sum(r['physical_steps'] for r in callback.samples),stage_reset_physics=env.total_reset_physics_steps,
        stage_cleanup_physics=env.total_physics_steps-physics_before_close,stage_total_physics=env.total_physics_steps,
        completed_episodes=callback.completed,episode_results=env.episode_results,next_episode_offset=offset+env.episode+1,
        checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),prior=prior,actor_delta_l2=delta(before[0],after[0]),critic_delta_l2=delta(before[1],after[1]),
        updates=updates,wall_s=time.perf_counter()-start,source_hashes=code_hashes(),run_directory=str(folder))
    write_json(folder/'complete.json',result);print('GROUP COMPLETE',version,seed,model.num_timesteps,env.total_physics_steps,flush=True);return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['initial','continue'],required=True);a=parser.parse_args()
    check=read(Path(read(ROOT/'results/phase3c_validation_latest.json')['run'])/'complete.json')
    if not check['passed']:raise ValueError('Independent control gate failed')
    dest=destination('phase3c_training_'+a.stage);cfg=design();prior_results=[]
    if a.stage=='continue':
        gate=read(ROOT/'results/phase3c_2048_gate.json')
        if not gate['all_four_passed']:raise ValueError('Common budget gate failed; no group can continue')
        prior_results=read(Path(gate['training_run'])/'complete.json')['results']
        if any(r['source_hashes']!=code_hashes() for r in prior_results):raise ValueError('Frozen source/config changed')
    jobs=[]
    for seed in cfg['training_seeds']:
        for version in cfg['rewards']:
            prior=next((r['run_directory'] for r in prior_results if r['seed']==seed and r['version']==version),None)
            jobs.append((version,seed,str(dest/f'{version}_seed{seed}'),prior))
    write_json(dest/'before_training.json',dict(jobs=jobs,source_hashes=code_hashes(),config=cfg))
    results=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
        future_jobs={pool.submit(training_job,j):j for j in jobs}
        for f in concurrent.futures.as_completed(future_jobs):
            try:results.append(f.result())
            except BaseException as exc:
                write_json(dest/'worker_failure.json',dict(job=future_jobs[f],error=repr(exc)));raise
    write_json(dest/'complete.json',dict(results=results,stage=a.stage));write_json(ROOT/f'results/phase3c_training_{a.stage}_latest.json',dict(run=dest))
    print('STAGE COMPLETE',dest,flush=True)

if __name__=='__main__':main_guard(main)
