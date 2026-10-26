"""Exactly 1024/2048 decision-step smoke PPO; prerequisites cannot be bypassed."""
if __package__:
    from scripts._bootstrap import ROOT,destination,main_guard
else:
    from _bootstrap import ROOT,destination,main_guard
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from src.common import write_json,write_csv,sha
from src.rl.dependencies import activate,provenance
activate()
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from src.rl.panda_residual_env import PandaResidualEnv,config
from scripts.validate_phase3b import sources,read

def code_hashes():
    files=list((ROOT/'src/rl').glob('*.py'))+[ROOT/'configs/phase3b_smoke.json',ROOT/'scripts/train_phase3b.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in files}

def parameter_groups(model):
    d=model.policy.state_dict();actor={k:v.detach().cpu().clone() for k,v in d.items() if 'policy_net' in k or 'action_net' in k or k=='log_std'}
    critic={k:v.detach().cpu().clone() for k,v in d.items() if 'value_net' in k}
    return actor,critic
def tensor_hash(d):return hashlib.sha256(b''.join(k.encode()+v.numpy().tobytes() for k,v in sorted(d.items()))).hexdigest()
def delta(a,b):return float(np.sqrt(sum(torch.sum((a[k]-b[k])**2).item() for k in a)))

class AuditedPPO(PPO):
    def train(self):
        before=parameter_groups(self);start=time.perf_counter();super().train();after=parameter_groups(self)
        keys=('policy_gradient_loss','value_loss','approx_kl','clip_fraction','entropy_loss','explained_variance')
        metrics={key:(float(self.logger.name_to_value['train/'+key]) if np.isfinite(self.logger.name_to_value['train/'+key]) else None) for key in keys}
        if any(metrics[k] is None for k in keys if k!='explained_variance'):raise FloatingPointError('Nonfinite PPO optimization metric')
        event=dict(update=len(self.audit_events)+1,decision_steps=self.num_timesteps,optimizer_epochs_total=self._n_updates,
            actor_parameter_delta_l2=delta(before[0],after[0]),critic_parameter_delta_l2=delta(before[1],after[1]),
            actor_sha256=tensor_hash(after[0]),critic_sha256=tensor_hash(after[1]),optimization_wall_s=time.perf_counter()-start,**metrics)
        self.audit_events.append(event);write_json(Path(self.audit_dest)/'ppo_update_events.json',self.audit_events)
        write_csv(Path(self.audit_dest)/'ppo_update_events.csv',self.audit_events)
        self.save(str(Path(self.audit_dest)/f'checkpoints/ppo_after_update_{len(self.audit_events):02d}'))
        print('UPDATE',event,flush=True)

class SamplingLog(BaseCallback):
    def __init__(self,env,dest):super().__init__();self.raw=env;self.dest=dest;self.samples=[];self.completed=[]
    def _on_step(self):
        info=self.locals['infos'][0];action=self.locals['actions'][0];executed=self.locals['clipped_actions'][0]
        row=dict(decision=self.num_timesteps,global_physics_steps=self.raw.total_physics_steps,
            physical_steps_this_decision=info['physical_steps_this_decision'],reward=float(self.locals['rewards'][0]),
            done=bool(self.locals['dones'][0]),success=info['success'],safety_violation=info['safety_violation'],prohibited_contact=info['prohibited_contact'],
            raw_gaussian_action=json.dumps(action.tolist()),executed_normalized_action=json.dumps(executed.tolist()),
            sample_mode='stochastic_actual_training',**{f'reward_{k}':v for k,v in info['reward_components'].items()})
        self.samples.append(row)
        if row['done']:
            self.completed.append(dict(decision=self.num_timesteps,success=info['success'],reason=info['reason']))
            print('COMPLETED EPISODE',self.completed[-1],flush=True)
        if self.num_timesteps%256==0:
            write_csv(self.dest/'training_samples.csv',self.samples)
            print('COLLECTED',self.num_timesteps,'DECISIONS',self.raw.total_physics_steps,'PHYSICS',flush=True)
        return True

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--decisions',type=int,choices=[1024,2048],default=1024);a=parser.parse_args()
    cfg=config()
    if a.decisions!=cfg['decisions']:raise ValueError('Change and revalidate frozen design explicitly before changing the smoke budget')
    gates={'interface':read(Path(read(ROOT/'results/phase3b_interface_latest.json')['run'])/'complete.json')['passed'],
        'zero':read(ROOT/'results/phase3b_zero_residual_equivalence.json')['passed'],
        'nonzero':read(ROOT/'results/phase3b_action_safety_tests.json')['passed']}
    if not all(gates.values()):raise ValueError('Prerequisites not passed')
    dest=destination('phase3b_training');(dest/'checkpoints').mkdir();torch.set_num_threads(1)
    write_json(dest/'frozen_training_config.json',dict(config=cfg,source_hashes=code_hashes(),dependencies=provenance(),gates=gates,
        source_paths=sources('train'),actual_decision_budget=a.decisions))
    env=PandaResidualEnv(sources('train'),dest/'actual_training',mode='training')
    options={k:v for k,v in cfg['ppo'].items() if k not in ('initial_log_std','net_arch')}
    model=AuditedPPO('MlpPolicy',env,seed=cfg['seed'],device='cpu',tensorboard_log=str(dest/'tensorboard'),verbose=1,
        policy_kwargs=dict(net_arch=dict(pi=cfg['ppo']['net_arch'],vf=cfg['ppo']['net_arch']),log_std_init=cfg['ppo']['initial_log_std']),**options)
    with torch.no_grad():model.policy.action_net.weight.zero_();model.policy.action_net.bias.zero_()
    model.audit_events=[];model.audit_dest=str(dest)
    model.save(str(dest/'checkpoints/ppo_initial_zero_mean'));initial=parameter_groups(model)
    callback=SamplingLog(env,dest);began=time.perf_counter();error=None
    try:
        model.learn(total_timesteps=a.decisions,callback=callback,tb_log_name='smoke',progress_bar=False)
        model.logger.dump(model.num_timesteps)
        training_physics_before_close=env.total_physics_steps
    except BaseException as exc:
        error=repr(exc);write_json(dest/'training_error.json',dict(error=error));raise
    finally:env.close()
    final=parameter_groups(model);model.save(str(dest/'checkpoints/ppo_smoke_final'))
    write_csv(dest/'training_samples.csv',callback.samples);write_csv(ROOT/'results/phase3b_smoke_training.csv',callback.samples)
    probe=np.zeros((1,78),dtype=np.float32);pred=model.predict(probe,deterministic=True)[0]
    loaded=PPO.load(str(dest/'checkpoints/ppo_smoke_final'),device='cpu');pred2=loaded.predict(probe,deterministic=True)[0]
    equal=bool(np.array_equal(pred,pred2));sampled=np.array([json.loads(x['executed_normalized_action']) for x in callback.samples])
    result=dict(actual_ppo_decisions=model.num_timesteps,training_step_integrations=sum(x['physical_steps_this_decision'] for x in callback.samples),
        physical_steps_before_cleanup=training_physics_before_close,actual_physics_integrations_total=env.total_physics_steps,
        reset_hold_integrations=env.total_reset_physics_steps,cleanup_hold_integrations=env.total_physics_steps-training_physics_before_close,
        ppo_update_calls=len(model.audit_events),optimizer_epochs=model._n_updates,
        actor_delta_l2=delta(initial[0],final[0]),critic_delta_l2=delta(initial[1],final[1]),
        initial_actor_sha256=tensor_hash(initial[0]),final_actor_sha256=tensor_hash(final[0]),
        initial_critic_sha256=tensor_hash(initial[1]),final_critic_sha256=tensor_hash(final[1]),
        action_head_weight_norm=float(torch.linalg.vector_norm(model.policy.action_net.weight).item()),
        exact_reload_prediction=equal,stochastic_action_nonzero_fraction=float(np.mean(np.any(sampled!=0,axis=1))),
        maximum_executed_normalized_action=float(np.max(abs(sampled))),completed_episodes=callback.completed,
        episode_results=env.episode_results,wall_s=time.perf_counter()-began,error=error,source_hashes=code_hashes(),
        final_model=str(dest/'checkpoints/ppo_smoke_final.zip'),final_model_sha256=sha(dest/'checkpoints/ppo_smoke_final.zip'),
        warning='A smoke experiment only; no automatic long training or claim of PPO superiority')
    assert result['actual_ppo_decisions']==a.decisions and result['actor_delta_l2']>0 and result['critic_delta_l2']>0 and equal
    assert callback.completed,'At least one complete real episode required'
    write_json(dest/'complete.json',result);write_json(ROOT/'results/phase3b_training_latest.json',dict(run=dest));print('TRAINING COMPLETE',dest,flush=True)

if __name__=='__main__':main_guard(main)
