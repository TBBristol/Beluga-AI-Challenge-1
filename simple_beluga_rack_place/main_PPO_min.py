import os
from stable_baselines3 import PPO
from simple_beluga_racking_env import BelugaRackingEnv
from stable_baselines3.common.monitor import Monitor
import random 
import torch as th



def train(instance_folder_path, max_steps=100, total_timesteps=20000, model_save_path='models'):
    """
    Train a PPO model on a single scale, using random container instances.
    We'll generate a new instance each time we reset by re-creating the env. 
    
    
    :param max_steps: optional time limit
    :param total_timesteps: how many steps to train
    :param model_save_path: directory to save the model
    :return: the trained model
    """
    # Create a log directory
    log_dir = f"logs/instances"
    os.makedirs(log_dir, exist_ok=True)

    
    
    # Create env
    env = BelugaRackingEnv(instance_folder_path=instance_folder_path)
    # Save logs in `log_dir`
    env = Monitor(env, log_dir)

    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose = 1,
        learning_rate=3e-4,
        batch_size=256,
        gamma=0.99,
        tensorboard_log="tensorboard_log",
        policy_kwargs={
            "optimizer_class": th.optim.Adam,
            "optimizer_kwargs": dict(
                eps=1e-5,
                weight_decay=1e-5
            ),
        }
    )
    
    
    # Train with monitoring
    model.learn(
        total_timesteps=total_timesteps,
        log_interval=100,
        
    )
    
    
    # Save model
    save_name = f"PPO_VANILLA_instances.zip"
    save_path = os.path.join(model_save_path, save_name)
    os.makedirs(model_save_path, exist_ok=True)
    model.save(save_path)
    
    env.close()
    return model, log_dir

def main(instance_folder_path):
    model_save_dir = "models"
    model, log_dir = train(
            instance_folder_path,
            total_timesteps=100,  
            model_save_path=model_save_dir

        )
   

 


if __name__ == "__main__":
    instance_folder_path = "/Users/ha24583/Documents/GitHub/Beluga-AI-Challenge/instances"
    main(instance_folder_path)