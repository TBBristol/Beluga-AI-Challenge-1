from stable_baselines3 import PPO
from custom_policy import CustomAttentionPolicy
from self_attention_extractor import SelfAttentionBlock
import torch as th
import torch.nn as nn
from stacking_paper_env import ContainerStackEnv
from stable_baselines3.common.monitor import Monitor
import os
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
import torch
from stable_baselines3 import PPO
from old_step_api_wrapper import OldStepAPIWrapper

import random
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


def generate_training_set(scale, num_instances=100):
    """
    Generate 'num_instances' container lists, each of length = scale.
    Returns a list of lists. E.g.:
      [
        [2, 0, 1, 4, 3],   # instance 1
        [1, 0, 2, 3, 4],   # instance 2
        ...
      ]
    """
    instances = []
    for _ in range(num_instances):
        # Example: shuffle a list of priorities from 0..scale-1
        containers = list(range(scale))
        random.shuffle(containers)
        instances.append(containers)
    return instances



import os
import time
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

def train_for_scale(scale, max_height=3, max_steps=100, total_timesteps=20000, model_save_path='models'):
    """
    Train a PPO model on a single scale, using random container instances.
    We'll generate a new instance each time we reset by re-creating the env. 
    
    :param scale: number of containers
    :param max_height: stack height
    :param max_steps: optional time limit
    :param total_timesteps: how many steps to train
    :param model_save_path: directory to save the model
    :return: the trained model
    """
    # Create a log directory
    log_dir = f"logs/scale_{scale}"
    os.makedirs(log_dir, exist_ok=True)

    # 1) Generate the training data
    training_data = generate_training_set(scale=scale, num_instances=100)
    
    # Create env
    env = ContainerStackEnv(training_data, max_height, max_steps)
    # Save logs in `log_dir`
    env = Monitor(env, log_dir)

    # Create PPO model (using default MlpPolicy or your custom policy)
     # Build PPO with our CustomAttentionPolicy
    model = PPO(
        policy=CustomAttentionPolicy,
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=256,
        n_epochs=20,
        gamma=0.99,
        ent_coef=0.01,
        vf_coef=0.1,
        max_grad_norm = 1000,
        clip_range=0.2,
        tensorboard_log= "tensorboard_log",
        #target_kl=0.02,
        policy_kwargs={
            "optimizer_class": th.optim.Adam,
            "optimizer_kwargs": dict(
                eps=1e-5,
                weight_decay=1e-5
            ),
        }
    )
    
    # Create callbacks with more frequent checking
    gradient_callback = GradientMonitorCallback(check_freq=500)
    stats_callback = PrintStatsCallback(verbose=1)
    #grad_log_callback = GradLoggingCallback(verbose = 1)
    
    # Train with monitoring
    model.learn(
        total_timesteps=total_timesteps,
        #callback=[gradient_callback,stats_callback],
        log_interval=100
    )
    
    # Plot final gradient history
    gradient_callback.plot_gradient_history()
    
    # Save model
    save_name = f"ppo_scale_{scale}.zip"
    save_path = os.path.join(model_save_path, save_name)
    os.makedirs(model_save_path, exist_ok=True)
    model.save(save_path)
    
    env.close()
    return model, log_dir




import pandas as pd
import matplotlib.pyplot as plt
import glob

def plot_monitor_csv(log_dir):
    # Find the monitor file(s)
    monitor_files = glob.glob(os.path.join(log_dir, "*monitor.csv"))
    if not monitor_files:
        print("No monitor files found in", log_dir)
        return
    
    df_list = []
    for f in monitor_files:
        df = pd.read_csv(f, skiprows=1)  # first row is #gym....
        df_list.append(df)
    
    # Combine if multiple
    data = pd.concat(df_list)
    data['t'] = data.index  # or use 'l' steps
    
    plt.figure(figsize=(8,6))
    plt.plot(data['t'], data['r'], label="episode_reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Progress")
    plt.legend()
    plt.show()

def test_model(scale, model_path, max_height=3, max_steps=50, episodes=5):
    """
    Loads a PPO model and runs some test episodes on newly generated instances with the same scale.
    Prints or logs final rewards.
    """
    from stable_baselines3 import PPO
    
    # Reload model
    model = PPO.load(model_path)
    
    rewards_list = []
    for ep in range(episodes):
        training_data = generate_training_set(scale=scale, num_instances=100)
        # Generate a new env instance
        env = ContainerStackEnv(training_data, max_height, max_steps)  
        
        obs, info = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            # PPO expects shape (n_envs, obs_dim) or (n_envs, *obs_shape)
            # For your custom obs, might need unsqueeze(0)
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, infos = env.step(action)
            ep_reward += reward
            done = terminated or truncated
        env.render()
        rewards_list.append(ep_reward)
        env.close()
    
    print(f"Tested {model_path} for {episodes} episodes at scale={scale}. Rewards: {rewards_list}")
    print(f"Average reward: {sum(rewards_list)/len(rewards_list)}")

def main(scale):
    model_save_dir = "models"
    model, log_dir = train_for_scale(
            scale=scale,
            max_height=3,
            max_steps=1000,
            total_timesteps=1000000,  
            model_save_path=model_save_dir

        )
    plot_monitor_csv(log_dir)
    # Test the model
    saved_model_path = f"{model_save_dir}/ppo_scale_{scale}.zip"
    test_model(scale=scale, model_path=saved_model_path, episodes=3)


class GradientMonitorCallback(BaseCallback):
    def __init__(self, check_freq=1000, verbose=0):
        super(GradientMonitorCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.grad_history = {}

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Access the model's policy to get gradients
            for name, param in self.model.policy.named_parameters():
                if param.grad is not None:
                    grad_mean = param.grad.mean().item()
                    grad_norm = param.grad.norm().item()
                    print(f"Gradient {name}: mean={grad_mean:.6f}, norm={grad_norm:.6f}")
                    
                    # Store for history
                    if name not in self.grad_history:
                        self.grad_history[name] = []
                    self.grad_history[name].append(grad_norm)
            
            # Ensure print flush
            import sys
            sys.stdout.flush()
        
        return True
    

    def plot_gradient_history(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        for name, history in self.grad_history.items():
            plt.plot(history, label=name)
        plt.xlabel('Check Points')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Norms Over Time')
        plt.legend()
        plt.yscale('log')
        plt.show()

# Add a learning rate scheduler callback
from stable_baselines3.common.callbacks import EveryNTimesteps

class GradLoggingCallback(BaseCallback):
    """
    Custom callback that logs gradient norms and policy ratio stats
    *per optimization batch* (not a default SB3 hook).
    """
    def __init__(self, verbose=1):
        super().__init__(verbose=verbose)
        # For storing logs if needed:
        self.batch_logs = []

    def _on_step(self) -> bool:
        """
        This method is called at each step.
        Implement it to satisfy the abstract class requirement.
        """
        return True

    def on_training_batch(
        self,
        grad_norm_before: float,
        grad_norm_after: float,
        ratio_array: np.ndarray,
        approx_kl_div: float,
        epoch: int,
        total_epochs: int
    ):
        """
        Called manually from our custom PPO during each minibatch update.
        
        :param grad_norm_before: Gradient norm before clipping
        :param grad_norm_after: Gradient norm after clipping
        :param ratio_array: NumPy array of exp(new_log_prob - old_log_prob) for the current minibatch
        :param approx_kl_div: Approximate KL divergence for the current minibatch
        :param epoch: Current epoch index (0-based)
        :param total_epochs: Total number of epochs in the training loop
        """
        if self.verbose > 0:
            print(f"[E {epoch+1}/{total_epochs}] Grad Norm (before/after): "
                  f"{grad_norm_before:.4f} / {grad_norm_after:.4f}, "
                  f"Ratio mean: {ratio_array.mean():.3f}, KL: {approx_kl_div:.5f}")

        # Optionally store logs
        log_dict = {
            "epoch": epoch,
            "grad_norm_before": grad_norm_before,
            "grad_norm_after": grad_norm_after,
            "ratio_mean": ratio_array.mean(),
            "ratio_std": ratio_array.std(),
            "ratio_min": ratio_array.min(),
            "ratio_max": ratio_array.max(),
            "approx_kl": approx_kl_div
        }
        self.batch_logs.append(log_dict)

class LearningRateScheduler(BaseCallback):
    def __init__(self, check_freq):
        super().__init__()
        self.check_freq = check_freq
        self.initial_lr = 3e-4
    
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            # Decay learning rate
            progress = self.n_calls / self.model.num_timesteps
            new_lr = self.initial_lr * (1 - progress)
            self.model.learning_rate = max(new_lr, 1e-5)
        return True
    

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class PrintStatsCallback(BaseCallback):
    """
    Custom callback for printing rollout buffer statistics after each rollout.
    """
    def __init__(self, verbose=0):
        super(PrintStatsCallback, self).__init__(verbose=verbose)

    def _on_step(self) -> bool:
        """
        This method is called at each step.
        Implement it to satisfy the abstract class requirement.
        """
        return True
        
    def _on_rollout_end(self) -> bool:
        """
        This method is called after the agent has finished collecting a rollout.
        Here, we print stats from the rollout buffer (advantages, returns, etc.).
        """
        # Access the rollout buffer
        rollout_buffer = self.model.rollout_buffer
        
        # Extract arrays
        advantages = rollout_buffer.advantages
        returns = rollout_buffer.returns
        log_probs = rollout_buffer.log_probs
        values = rollout_buffer.values
        
        # Compute some summary statistics
        adv_mean, adv_std = np.mean(advantages), np.std(advantages)
        ret_mean, ret_std = np.mean(returns), np.std(returns)
        logp_mean, logp_std = np.mean(log_probs), np.std(log_probs)
        val_mean, val_std = np.mean(values), np.std(values)
        
        # Print them (only if verbose > 0)
        if self.verbose > 0:
            print(f"--- Rollout End ---")
            print(f"Advantage mean: {adv_mean:.3f}, std: {adv_std:.3f}")
            print(f"Returns mean:   {ret_mean:.3f}, std: {ret_std:.3f}")
            print(f"Log prob mean:  {logp_mean:.3f}, std: {logp_std:.3f}")
            print(f"Value mean:     {val_mean:.3f},  std: {val_std:.3f}\n")
        
        # Returning True to keep training
        return True
    

class PPOWithGradLogging(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize _last_dones and _callback
        self._last_dones = None
        self._callback = None

    def learn(self, total_timesteps, callback=None, log_interval=100):
        # Set the callback
        self._callback = callback
        # Call the parent class's learn method
        return super().learn(total_timesteps=total_timesteps, callback=callback, log_interval=log_interval)

    def train(self):
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        if self.use_sde:
            self.policy.reset_noise(self.n_steps)

        # -- FIX: compute last_values here --
        with th.no_grad():
            obs_tensor = th.as_tensor(self._last_obs).to(self.device)
            last_values = self.policy.predict_values(obs_tensor)
        
        # Initialize _last_dones if not already done
        if self._last_dones is None:
            self._last_dones = np.zeros((self.env.num_envs,), dtype=bool)

        self.rollout_buffer.compute_returns_and_advantage(
            last_values=last_values,
            dones=self._last_dones
        )

        # If a callback is provided:
        callback = self._callback

        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            for rollout_data in self.rollout_buffer.get(self.batch_size):
                # Evaluate the policy
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, rollout_data.actions
                )

                # Surrogate losses
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                advantages = rollout_data.advantages

                # Check if clip_range is a function and call it
                if callable(self.clip_range):
                    clip_range_value = self.clip_range(self._current_progress_remaining)
                else:
                    clip_range_value = self.clip_range


                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio, 1 - clip_range_value, 1 + clip_range_value
                )
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                value_loss = F.mse_loss(rollout_data.returns, values.flatten())
                entropy_loss = -th.mean(entropy)

                loss = (policy_loss
                        + self.vf_coef * value_loss
                        + self.ent_coef * entropy_loss)

                self.policy.optimizer.zero_grad()
                loss.backward()

                # Log gradient norm before clipping
                grad_norm_before = 0.0
                for param in self.policy.parameters():
                    if param.grad is not None:
                        grad_norm_before += param.grad.data.norm(2).item() ** 2
                grad_norm_before = grad_norm_before ** 0.5
                print(f"before {grad_norm_before}")

                # Clip grads
                if self.max_grad_norm is not None:
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

                # Log gradient norm after clipping
                grad_norm_after = 0.0
                for param in self.policy.parameters():
                    if param.grad is not None:
                        grad_norm_after += param.grad.data.norm(2).item() ** 2
                grad_norm_after = grad_norm_after ** 0.5
                print(f"after {grad_norm_after}")

                # Step optimizer
                self.policy.optimizer.step()

                approx_kl_div = th.mean(rollout_data.old_log_prob - log_prob).detach().cpu().numpy()
                approx_kl_divs.append(approx_kl_div)

                # Call our custom callback
                if callback is not None and isinstance(callback, GradLoggingCallback):
                    callback.on_training_batch(
                        grad_norm_before=grad_norm_before,
                        grad_norm_after=grad_norm_after,
                        ratio_array=ratio.detach().cpu().numpy(),
                        approx_kl_div=approx_kl_div,
                        epoch=epoch,
                        total_epochs=self.n_epochs
                    )

            if self.verbose > 0:
                print(f"[Epoch {epoch+1}/{self.n_epochs}] mean KL: {np.mean(approx_kl_divs):.6f}")

        return None


if __name__ == "__main__":
    scale = 10
    main(scale)
    
    
