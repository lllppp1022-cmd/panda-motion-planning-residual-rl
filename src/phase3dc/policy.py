from src.rl.dependencies import activate
activate()
import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from src.bc_residual.core import Policy
from scripts.train_phase3b import config
from src.phase3dc.environment import design
from stable_baselines3 import PPO


class BoundedMeanPolicy(ActorCriticPolicy):
    """Untransformed Gaussian, TANH MEAN, clipped sampled actions at env boundary.

    BC deterministic action is tanh(linear(tanhMLP(normalize(obs)))). This is not
    a tanh-squashed Gaussian and needs no change-of-variables log likelihood.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.register_buffer('bc_mean',torch.zeros(85));self.register_buffer('bc_std',torch.ones(85))
    def extract_features(self,obs,*args,**kwargs):return super().extract_features((obs-self.bc_mean)/self.bc_std,*args,**kwargs)
    # SB3's prediction helpers explicitly call BaseModel.extract_features and
    # bypass this override, so they require the SAME normalization explicitly.
    def get_distribution(self,obs):return super().get_distribution((obs-self.bc_mean)/self.bc_std)
    def predict_values(self,obs):return super().predict_values((obs-self.bc_mean)/self.bc_std)
    def _get_action_dist_from_latent(self,latent_pi):return self.action_dist.proba_distribution(torch.tanh(self.action_net(latent_pi)),self.log_std)


def initialize(model,checkpoint,method):
    bc=Policy(checkpoint);p=model.policy
    with torch.no_grad():
        p.bc_mean.copy_(bc.mean);p.bc_std.copy_(bc.std)
        if method=='A':
            for source,dest in [(bc.model.net[0],p.mlp_extractor.policy_net[0]),(bc.model.net[2],p.mlp_extractor.policy_net[2]),(bc.model.net[4],p.action_net)]:
                dest.weight.copy_(source.weight);dest.bias.copy_(source.bias)
        else:p.action_net.weight.zero_();p.action_net.bias.zero_()


def make_model(env,seed,checkpoint,method,cls=PPO,**kwargs):
    opts={k:v for k,v in config()['ppo'].items() if k not in ('net_arch','initial_log_std')}
    m=cls(BoundedMeanPolicy,env,seed=seed,device='cpu',verbose=0,policy_kwargs=dict(net_arch=dict(pi=[64,64],vf=[64,64]),activation_fn=nn.Tanh,log_std_init=float(__import__('math').log(design()['initial_std']))),**opts,**kwargs)
    initialize(m,checkpoint,method);return m
