from stable_baselines3 import SAC

from env import Gym2OpEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor


from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import torch as th
from torch import nn
from stable_baselines3.sac.policies import SACPolicy
from typing import Dict, List, Tuple, Type, Union

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

import numpy as np

class CustomFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom features extractor for power grid control
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
        # Initialize the superclass
        super().__init__(observation_space, features_dim)
        
        n_input_features = int(np.prod(observation_space.shape))
        
        # Shared feature extractor
        self.shared_net = nn.Sequential(
            nn.Linear(n_input_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        
        # Power flow branch
        self.power_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        # Topology branch
        self.topo_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        
        # Combine features
        self.combine = nn.Sequential(
            nn.Linear(128, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        shared_features = self.shared_net(observations)
        
        # Process through separate branches
        power_features = self.power_net(shared_features)
        topo_features = self.topo_net(shared_features)
        
        # Combine features
        combined = th.cat([power_features, topo_features], dim=1)
        return self.combine(combined)

def train_sac_agent(env, total_timesteps=200000, logdir="logs"):
    # Wrap the environment
    env = DummyVecEnv([lambda: env])
    
    # Create policy kwargs with our custom feature extractor
    policy_kwargs = dict(
        features_extractor_class=CustomFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        net_arch=dict(pi=[256, 128], qf=[256, 128])
    )
    
    # Initialize SAC agent with our custom policy
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=logdir,
        policy_kwargs=policy_kwargs
    )
    
    # Train the agent with callbacks
    model.learn(
        total_timesteps=total_timesteps,
    )
    
    # Save the final model
    model.save("SAC_iter2")
    
    return model

# Usage
if __name__ == "__main__":
    logdir = "SAC_logs"
    env = Gym2OpEnv()
    env = Monitor(env, logdir)
    sac_agent = train_sac_agent(env, logdir=logdir)