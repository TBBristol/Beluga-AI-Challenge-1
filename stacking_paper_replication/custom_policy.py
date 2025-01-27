from stable_baselines3.common.policies import ActorCriticPolicy
from torch.distributions import Categorical
from self_attention_extractor import SelfAttentionBlock
import torch.nn as nn
import torch as th
from stable_baselines3.common.distributions import CategoricalDistribution
import numpy as np
import torch.nn.functional as F


class MyCategoricalDistribution(nn.Module, CategoricalDistribution):
    def __init__(self, action_dim: int):
        nn.Module.__init__(self)
        CategoricalDistribution.__init__(self, action_dim)
        self.logits = None

    def proba_distribution(self, logits: th.Tensor, *args, **kwargs) -> "MyCategoricalDistribution":
        super().proba_distribution(logits, *args, **kwargs)
        self.logits = logits
        return self

    @property
    def shape(self):
        if self.logits is None:
            return None
        return self.logits.shape

    def clone(self) -> "MyCategoricalDistribution":
        new_dist = MyCategoricalDistribution(self.action_dim)
        if self.logits is not None:
            new_dist.logits = self.logits.clone()
            new_dist.distribution = th.distributions.Categorical(logits=new_dist.logits)
        return new_dist

    def to(self, device: th.device):
        if self.logits is not None:
            self.logits = self.logits.to(device)
            self.distribution = th.distributions.Categorical(logits=self.logits)
        return self

    def cpu(self):
        return self.to(th.device("cpu"))

    def numpy(self):
        """
        If something calls dist.numpy(), we'll return the distribution's probabilities
        as a NumPy array. (Or you could return logits, etc.)
        """
        if self.distribution is None:
            return None
        # distribution.probs is shape (batch_size, n_actions)
        # detach => remove from graph, cpu => host, numpy() => array
        return self.distribution.probs.detach().cpu().numpy()

def make_mlp(layer_sizes):
    """Utility: create an MLP from a list of layer sizes, with ReLU in-between."""
    layers = []
    for i in range(len(layer_sizes) - 1):
        in_size = layer_sizes[i]
        out_size = layer_sizes[i+1]
        layers.append(nn.Linear(in_size, out_size))
        if i < len(layer_sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class CustomAttentionPolicy(ActorCriticPolicy):
    """
    PPO actor-critic with:
      - Self-attention over (n_stacks, 6) -> (n_stacks, 256)
      - Actor MLP( n_stacks*256 -> [128,128,32, n_stacks] )
      - Critic MLP( 256 -> [128,128,32, 1] ) with mean pooling across n_stacks dimension.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # We assume observation shape = (n_stacks, 6)
        n_stacks, in_dim = self.observation_space.shape
        embed_dim = 256
        n_heads = 4
        
        # Create the self-attention block
        self.self_attention = SelfAttentionBlock(
            n_heads=n_heads,
            embed_dim=embed_dim,
            in_dim=in_dim
        )
        # Create a SB3 CategoricalDistribution for discrete actions
        self.action_dist = MyCategoricalDistribution(self.action_space.n)
        
        # Actor: flatten -> MLP -> (B, n_stacks) logits
        #actor_in_dim = n_stacks * embed_dim
        # MLP(128,128,32,n_stacks)
        self.actor_net = make_mlp([embed_dim, 128, 128, 32, 1])
        
        # Critic: mean-pool -> MLP -> (B,1)
        # input is (B, 256) after mean pooling
        critic_in_dim = embed_dim
        # MLP(128,128,32,1)
        self.critic_net = make_mlp([critic_in_dim, 128, 128, 32, 1])
        
        # IMPORTANT: we must call this so SB3 knows which parameters to optimize
        self._initialize_weights_and_modules()

        self.large_negative = -1e4
        
        # Add layer normalization to the actor network
        self.actor_norm = nn.LayerNorm(self.actor_net[-1].out_features)
        
        self.max_grad_norm = 0.5  # Add gradient clipping threshold

    def _get_action_mask(self, obs):
        """Create a mask for invalid actions."""
        heights = obs[:, :, 0]  # (B, n_stacks)
        # Convert to float for gradient propagation

        return (heights < 0.99).float()  # Changed from boolean to float 1 is stackable 0 is not

    def _initialize_weights_and_modules(self):
        # A hook to move the newly created modules to the correct device
        # and set up any other policy internals in SB3.
        # By default, SB3 calls self.to(self.device) after this.
        modules = [
            self.self_attention,
            self.actor_net,
            self.critic_net
        ]
        for module in modules:
            module.to(self.device)

    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False
    ):
        """
        Required by ActorCriticPolicy in SB3, but we typically define
        separate forward_actor, forward_critic, get_distribution, etc.
        We'll just implement it inline here for clarity.
        """
        # 1) Self-attention: (B, n_stacks, 6) -> (B, n_stacks, 256)
        attn_output = self.self_attention(obs)
        
        # Actor branch
        B, n_stacks, embed_dim = attn_output.shape
        actor_in = attn_output.reshape(B, n_stacks, embed_dim)
        
        # Scale inputs
        #actor_in = actor_in / (actor_in.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Get logits 
        logits = self.actor_net(actor_in) #(B,n_stacks,1)
        logits = logits.squeeze(-1) #(B,n_stacks)
        
        
        # Masking
        action_mask = self._get_action_mask(obs)
      
        #masked_logits = logits * action_mask + (1 - action_mask) * self.large_negative #Apply large neg to masked values]
        masked_logits = logits.masked_fill(action_mask == 0, float('-inf'))
        masked_logits = F.softmax(masked_logits, dim  = -1)
       
        dist = self.action_dist.proba_distribution(masked_logits)
        
        if deterministic:
            actions = th.argmax(logits, dim=-1)
        else:
            actions = dist.sample()
        
        # 3) Critic branch 
        # Mean-pool => (B, 256)
        critic_in = attn_output.mean(dim=1)
        
        values = self.critic_net(critic_in) #(B,1)
        values = self.critic_net(critic_in).squeeze(-1)  # (B,)
        log_probs = dist.log_prob(actions)
        #TODO check log probs
        return actions, values, log_probs

    def get_distribution(self, obs: th.Tensor):
        """
        SB3 calls this to get the distribution over actions.
        """
        # 1) Self-attention: (B, n_stacks, 6) -> (B, n_stacks, 256)
        attn_output = self.self_attention(obs)
        
        # Actor branch
        B, n_stacks, embed_dim = attn_output.shape
        actor_in = attn_output.reshape(B, n_stacks, embed_dim)
        
        # Scale inputs
        #actor_in = actor_in / (actor_in.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Get logits 
        logits = self.actor_net(actor_in) #(B,n_stacks,1)
        logits = logits.squeeze(-1) #(B,n_stacks)
        
        
        # Masking
        action_mask = self._get_action_mask(obs)
      
        #masked_logits = logits * action_mask + (1 - action_mask) * self.large_negative #Apply large neg to masked values]
        masked_logits = logits.masked_fill(action_mask == 0, float('-inf'))
        masked_logits = F.softmax(masked_logits, dim  = -1)
       
        dist = self.action_dist.proba_distribution(masked_logits)
        return dist

    def forward_actor(self, obs: th.Tensor):
        """
        Return the policy logits (B, n_stacks).
        """
         # 1) Self-attention: (B, n_stacks, 6) -> (B, n_stacks, 256)
        attn_output = self.self_attention(obs)
        
        # Actor branch
        B, n_stacks, embed_dim = attn_output.shape
        actor_in = attn_output.reshape(B, n_stacks, embed_dim)
        
        # Scale inputs
        #actor_in = actor_in / (actor_in.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Get logits 
        logits = self.actor_net(actor_in) #(B,n_stacks,1)
        logits = logits.squeeze(-1) #(B,n_stacks)
        
        
        # Masking
        action_mask = self._get_action_mask(obs)
      
        #masked_logits = logits * action_mask + (1 - action_mask) * self.large_negative #Apply large neg to masked values]
        masked_logits = logits.masked_fill(action_mask == 0, float('-inf'))
        masked_logits = F.softmax(masked_logits, dim  = -1)
        
        return masked_logits

    def forward_critic(self, obs: th.Tensor):
        """
        Return the value estimate (B,).
        """
        attn_output = self.self_attention(obs)
        critic_in = attn_output.mean(dim=1)   # (B, 256)
        values = self.critic_net(critic_in).squeeze(-1)  # (B,)
        return values

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor):
        
       # 1) Self-attention: (B, n_stacks, 6) -> (B, n_stacks, 256)
        attn_output = self.self_attention(obs)
        
        # Actor branch
        B, n_stacks, embed_dim = attn_output.shape
        actor_in = attn_output.reshape(B, n_stacks, embed_dim)
        
        # Scale inputs
        #actor_in = actor_in / (actor_in.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Get logits 
        logits = self.actor_net(actor_in) #(B,n_stacks,1)
        logits = logits.squeeze(-1) #(B,n_stacks)
        
        
        # Masking
        action_mask = self._get_action_mask(obs)
      
        #masked_logits = logits * action_mask + (1 - action_mask) * self.large_negative #Apply large neg to masked values]
        masked_logits = logits.masked_fill(action_mask == 0, float('-inf'))
        masked_logits = F.softmax(masked_logits, dim  = -1)
        
        dist = self.action_dist.proba_distribution(logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        
        # Scale critic input
        critic_in = attn_output.mean(dim=1)
   
        values = self.critic_net(critic_in) #(B,1)
        values = self.critic_net(critic_in).squeeze(-1)  # (B,)
        return values, log_prob, entropy

   