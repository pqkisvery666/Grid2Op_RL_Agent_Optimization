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


class DictFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Dict, features_dim: int = 128):
        # Initialize the parent class
        super().__init__(observation_space, features_dim)

        # Calculate total input size from all observation components
        self.total_input_size = 0
        for space in observation_space.spaces.values():
            if len(space.shape) == 1:  # 1D observations
                self.total_input_size += space.shape[0]
            else:  # Handle multi-dimensional observations
                self.total_input_size += np.prod(space.shape)

        # Define network architecture
        self.shared_net = nn.Sequential(
            nn.Linear(self.total_input_size, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim)
        )

    def forward(self, observations):
        # Convert dictionary observations to a single tensor
        tensors = []
        for key, value in observations.items():
            if isinstance(value, np.ndarray):
                tensor = torch.FloatTensor(value).to(self.shared_net[0].weight.device)
            else:
                tensor = value.to(self.shared_net[0].weight.device)

            # Handle both batched and unbatched inputs
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            tensors.append(tensor.view(tensor.shape[0], -1))

        # Concatenate all tensors
        combined = torch.cat(tensors, dim=1)
        return self.shared_net(combined)


class CustomGridPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        # Initialize with proper network architecture
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=dict(pi=[64, 64], vf=[64, 64]),  # Simplified architecture
            features_extractor_class=DictFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=128),
            *args,
            **kwargs
        )


def train_improved_ppo_agent(env, total_timesteps=1000000):
    # Create vectorized environment first
    env = DummyVecEnv([lambda: env])

    model = PPO(
        policy=CustomGridPolicy,
        env=env,
        verbose=1,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        tensorboard_log="logs"
    )

    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name="PPO_improved1mil"
    )

    model.save("ppo_grid_improved1mil")
    return model
# Example usage:
if __name__ == "__main__":

    logdir = "PPO_logs\logs"
    env = Gym2OpEnv()
    env = Monitor(env, logdir)

    # Train the improved agent
    ppo_agent = train_improved_ppo_agent(env)