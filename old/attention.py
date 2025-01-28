import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from old.stacking_env import StackingEnv

import torch.nn as nn
import random


class DefaultAttention(nn.Module):
    def __init__(self,env,embed_dim,num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.env = env
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * self.num_heads == self.embed_dim #Needs to be divisible

        #nn.linear(in_features,out_features)
        self.query = nn.Linear(self.env.observation_space.shape[1], self.embed_dim) #Input: (*,H_in) where * is any number of dims inc none and H_in = in features
        self.key = nn.Linear(self.env.observation_space.shape[1], self.embed_dim)
        self.value = nn.Linear(self.env.observation_space.shape[1], self.embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        #critic is a single value per batch taking in mean of the embeddings so in shape is batch,racks
        self.critic = nn.Sequential(nn.Linear(self.env.observation_space.shape[0], 128), nn.ReLU(), nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))
        self.actor = nn.Sequential(nn.Linear(self.env.observation_space.shape[0]*self.embed_dim, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 100))

    def forward(self,observations):
        observations, possible_actions = observations


        print(observations.shape) #batch, racks,features
        attn_output = self.encode_observations(observations)
        #attn_output shape is batch,racks,embed_dim
        #Value is a single score per batch and is a the mean of the embedding put through the critic network 
        means = torch.mean(attn_output,dim=2)
        #means shape is batch, racks
        values = self.critic(means)
        #values shape is batch,1
    
        action_probs = self.decode_actions(attn_output, possible_actions)

        return action_probs, values

    def encode_observations(self,observations):
        #Q,K,V outshape is (*,H_out) so in this case batch,racks,embed_dim
        query = self.query(observations)
        key = self.key(observations)
        value = self.value(observations)

        attn_output, attn_output_weights = self.attention(query, key, value)  #TODO check skip etc
        #attn_output shape is batch,racks,embed_dim
        return attn_output


        

    
    def decode_actions(self,attn_output, possible_actions):
        #attn_output shape is batch,racks,embed_dim
        #We want a score per action  per batch so we need in shape to be batch, all embdeddings
        attn_output = attn_output.reshape(attn_output.shape[0], -1)
        scores = self.actor(attn_output) 
        #scores shape is batch,actions
        masking = [self.env.get_action_mask(p) for p in possible_actions] #needs to be per batch state
        masking = torch.stack(masking)
        scores = scores.masked_fill(masking == 0, float('-inf'))
        action_probs = torch.softmax(scores,dim=-1)

        #[batch, actions]
        return action_probs

    
    def reset_parameters(self):
        ...

    

if __name__ == "__main__":
    
  
    embed_dim = 256
    num_heads = 1
    

    env = StackingEnv("instances/problem_256_s257_j148_r18_oc21_f91.json")
    obs,_ = env.reset()
    data = []
    possible_actions = []
    for i in range(50):
        possible_actions.append(env.get_possible_actions())
        data.append(obs)
      
        observation, reward, terminated, truncated, inf = env.step(random.sample(env.get_possible_actions(),1)[0])
        if terminated or truncated:
            obs,_ = env.reset()

    data = torch.stack(data).to(torch.float32)  #batch, racks, features
    att= DefaultAttention(env, embed_dim, num_heads)
    action_probs, values = att.forward((data,possible_actions))
    print(action_probs.shape)
    print(values.shape)



