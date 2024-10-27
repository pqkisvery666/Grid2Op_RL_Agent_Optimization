import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
import torch
import torch.nn as nn
from gymnasium.spaces import Dict

from env import Gym2OpEnv
from stable_baselines3.common.callbacks import EvalCallback
import torch
import torch.nn as nn
from gymnasium.spaces import Dict
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import get_schedule_fn

class DictFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Dict, features_dim: int = 256):  # Increased features_dim
        super().__init__(observation_space, features_dim)
        
        self.total_input_size = 0
        for space in observation_space.spaces.values():
            if len(space.shape) == 1:
                self.total_input_size += space.shape[0]
            else:
                self.total_input_size += np.prod(space.shape)

        # Deeper architecture with residual connections
        self.layer1 = nn.Sequential(
            nn.Linear(self.total_input_size, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1)  # Added dropout for regularization
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1)
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(512, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim)
        )

    def forward(self, observations):
        tensors = []
        for key, value in observations.items():
            if isinstance(value, np.ndarray):
                tensor = torch.FloatTensor(value).to(self.layer1[0].weight.device)
            else:
                tensor = value.to(self.layer1[0].weight.device)

            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            tensors.append(tensor.view(tensor.shape[0], -1))

        combined = torch.cat(tensors, dim=1)
        
        # Residual connections
        x1 = self.layer1(combined)
        x2 = self.layer2(x1) + x1  # Residual connection
        x3 = self.layer3(x2)
        
        return x3

class CustomGridPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        # Deeper network architecture
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]),  # Deeper architecture
            features_extractor_class=DictFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256),
            *args,
            **kwargs
        )

def create_learning_rate_schedule(initial_lr=3e-4, final_lr=1e-4):
    def schedule(progress):
        return final_lr + (initial_lr - final_lr) * (1 - progress)
    return schedule

def train_improved_ppo_agent(env, total_timesteps=1000000):
    env = DummyVecEnv([lambda: env])
    
    # Create evaluation environment
    eval_env = DummyVecEnv([lambda: Gym2OpEnv()])
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model",
        log_path="./logs",
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./checkpoints/",
        name_prefix="ppo_grid_model"
    )

    model = PPO(
        policy=CustomGridPolicy,
        env=env,
        learning_rate=create_learning_rate_schedule(),
        verbose=1,
        tensorboard_log="PPO_improved/logs"
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        tb_log_name="PPO_improved_stable"
    )

    model.save("ppo_grid_improved_stable")
    return model

if __name__ == "__main__":
    logdir = "PPO_improved/logs"
    env = Gym2OpEnv()
    env = Monitor(env, logdir)
    
    ppo_agent = train_improved_ppo_agent(env)