from simple_select import Beluga_rack_stack
from cleanrl_ppo import DefaultAttention
import random
import torch

env = Beluga_rack_stack("/Users/ha24583/Documents/GitHub/Beluga-AI-Challenge/instances")
obs,info = env.reset(seed=1)

agent = DefaultAttention(env)
agent.load_agents("/Users/ha24583/Documents/GitHub/Beluga-AI-Challenge/models/model_20000")
cum_reward = 0
noops = 0
stacked = 0

with torch.no_grad():
    for i in range(100):

        poss_actions = env.get_possible_actions()
        action = agent.get_deterministic_action(obs)
        stacked += 1
        if action not in poss_actions:
            noops +=1
        obs, reward, terminated, truncated, info = env.step(action)
        cum_reward += reward
   
   
    
    print(cum_reward)
    print(f"noops{noops}")
    obs,info = env.reset(seed=1)

    stacked = 0
    cum_reward = 0
    for i in range(100):
        action = random.choice(env.get_possible_actions())
        obs, reward, terminated, truncated, info = env.step(action)
        cum_reward += reward

    print(cum_reward)
