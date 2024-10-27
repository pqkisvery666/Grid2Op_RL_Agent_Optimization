from env import Gym2OpEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import A2C
from stable_baselines3.common.monitor import Monitor

import os
import torch

logdir = "ASC_baseline/logs"

def train_ppo_agent(env, total_timesteps=200000):

    env = DummyVecEnv([lambda: env])
    
    model = A2C(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=logdir
        )
    
    model.learn(total_timesteps=total_timesteps)
    
    model.save("A2C_iteration1")

    return model

env = Gym2OpEnv()
env = Monitor(env, logdir)

train_ppo_agent(env, total_timesteps=200000)

