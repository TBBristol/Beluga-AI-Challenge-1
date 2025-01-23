import gymnasium as gym

class OldStepAPIWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
       
        return obs, info

    def step(self, action):
        results = self.env.step(action)
        print("Wrapper step => got from env:", results)
        obs, reward, terminated, truncated, info = results
        done = terminated or truncated
        print("Wrapper step => returning 4 items")
        return obs, reward, done, info