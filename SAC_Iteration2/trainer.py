from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from env import Gym2OpEnv
from stable_baselines3.common.buffers import ReplayBuffer
import numpy as np
from collections import deque
import torch
import os


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Custom replay buffer that prioritizes important transitions
    """

    def __init__(self, buffer_size: int, observation_space, action_space,
                 device='auto', n_envs=1, alpha=0.6, beta=0.4):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001
        self.priorities = np.ones((buffer_size,), dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done, infos):
        """Add a new experience to memory."""
        # Calculate priority for the new experience
        priority = self._get_priority(obs, reward, done, infos)

        # Add experience to buffer using parent class method
        super().add(obs, next_obs, action, reward, done, infos)

        # Store priority
        self.priorities[self.pos - 1] = priority

    def _get_priority(self, obs, reward, done, infos):
        """Calculate priority score for a transition."""
        priority = 1.0

        # Handle vectorized observations
        if isinstance(obs, np.ndarray):
            max_val = np.max(np.abs(obs))
            if max_val > 0.95:
                priority += 1.0

        # Handle vectorized rewards and dones
        if isinstance(reward, (list, np.ndarray)):
            reward = reward[0]
        if isinstance(done, (list, np.ndarray)):
            done = done[0]

        # Add reward magnitude to priority
        priority += abs(reward)

        # Higher priority for terminal states
        if done:
            priority += 3.0

        # Handle infos
        if isinstance(infos, list):
            info = infos[0]
            if isinstance(info, dict):
                if info.get('is_illegal', False) or info.get('is_ambiguous', False):
                    priority += 1.0

        return max(priority, 1e-6)

    def sample(self, batch_size: int, env=None):
        """Sample a batch of experiences."""
        if self.full:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]

        # Convert priorities to probabilities
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices and calculate importance weights
        indices = np.random.choice(len(probs), size=batch_size, p=probs)
        weights = (len(probs) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)

        # Get experiences using parent class method
        experiences = self._get_samples(indices, env)

        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        return experiences

    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions."""
        priorities = np.abs(priorities) + 1e-6
        self.priorities[indices] = priorities


class CustomSAC(SAC):
    """
    Modified SAC to use prioritized experience replay
    """

    def __init__(self, policy, env, **kwargs):
        super().__init__(policy, env, **kwargs)

        # Replace default replay buffer with prioritized version
        self.replay_buffer = PrioritizedReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            self.device,
            n_envs=self.n_envs
        )


def make_env():
    """Create and wrap the environment"""
    env = Gym2OpEnv()
    env = Monitor(env, logdir)
    return env


def train_sac_agent(env, total_timesteps=200000):
    # Create vectorized environment
    env = DummyVecEnv([lambda: env])

    # Initialize SAC with custom replay buffer
    model = CustomSAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=logdir,
        learning_starts=1000,
        batch_size=256,  # Increased batch size for more stable learning
        buffer_size=100000,
        ent_coef="auto"
    )

    # Train the agent
    model.learn(total_timesteps=total_timesteps)

    # Save the model
    model.save("sac_prioritized_replay_model_new")

    return model


logdir = "C:\\Users\\karis\\PycharmProjects\\RLPROJECT\\SAC\\SAC"



env = Gym2OpEnv()
env = Monitor(env, logdir)
ppo_agent = train_sac_agent(env)