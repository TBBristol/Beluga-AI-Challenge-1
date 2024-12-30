import gymnasium as gym
from stacking_env import StackingEnv

gym.register(
    id="gymnasium_env/StackingEnv-v0",
    entry_point=StackingEnv,
)
