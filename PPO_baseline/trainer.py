from env import Gym2OpEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

import os
import torch

logdir = "PPO_improved/logs"

def train_ppo_agent(env, total_timesteps=1000000):

    env = DummyVecEnv([lambda: env])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=logdir
        )
    
    model.learn(total_timesteps=total_timesteps)
    
    model.save("PPO_iteration_one\masked_ppo_simple.zip")

    return model

env = Gym2OpEnv()
env = Monitor(env, logdir)

train_ppo_agent(env, total_timesteps=1000000)