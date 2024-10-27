# Power Grid Management with Reinforcement Learning

This project implements and compares different reinforcement learning approaches for power grid management using the Grid2Op environment. We explore both Soft Actor-Critic (SAC) and Proximal Policy Optimization (PPO) algorithms with various improvements.

## Project Structure

```
RL_project/
├── PPO_baseline/
│   ├── env.py             # Grid2Op environment wrapper
│   ├── agent.py           # PPO agent implementation
│   ├── trainer.py         # Training script
│   └── ppo_baseline.zip   # Trained model weights
├── PPO_iteration1/
│   └── ...                # Similar structure with first improvement
├── PPO_iteration2/
│   └── ...                # Similar structure with second improvement
├── SAC_baseline/
│   ├── env.py             # Grid2Op environment wrapper
│   ├── agent.py           # SAC agent implementation
│   ├── trainer.py         # Training script
│   └── sac_baseline.zip   # Trained model weights
├── SAC_iteration1/
│   └── ...                # Similar structure with first improvement
├── SAC_iteration2/
│   └── ...                # Similar structure with second improvement
├── PPO_logs/              # Tensorboard logging files for PPO
└── SAC_logs/              # Tensorboard logging files for SAC
```

## Usage

### Training an Agent
Each algorithm variant can be trained using its respective trainer script:

```bash
cd PPO_baseline
python trainer.py
```

### Evaluating an Agent
To evaluate a trained agent:

```python
from env import Gym2OpEnv
from stable_baselines3 import PPO, SAC
from evaluation import evaluate_agent

# Load environment and agent
env = Gym2OpEnv()
agent = PPO.load("ppo_baseline.zip")  # or SAC.load() for SAC agents

# Run evaluation
stats = evaluate_agent(env, agent, num_episodes=100)
```

### Viewing Training Logs
Training progress can be visualized using Tensorboard:

```bash
tensorboard --logdir=PPO_logs  # or SAC_logs for SAC
```

## Implementation Details

### Environment Wrapper
- Custom Grid2Op environment wrapper implementing the Gymnasium interface
- Configurable observation and action spaces
- Normalized observations and discretized actions

### Algorithms
1. **SAC Implementation**
   - Prioritized Experience Replay
   - Custom replay buffer with priority-based sampling
   - Automatic entropy tuning

2. **PPO Implementation**
   - Custom neural network architecture
   - Shared feature extractor
   - Separate policy and value networks

## Results
Results and analysis can be found in the evaluation plots saved in each algorithm's directory.

## Authors
- Shailyn Ramsamy
- Karishma Lakhoo
- Salmaan Ebrahim

