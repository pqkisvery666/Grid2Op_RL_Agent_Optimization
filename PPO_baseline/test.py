from env import Gym2OpEnv 
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import numpy as np

def run_episode(env, agent, max_steps):
    obs, info = env.reset()
    step_rewards = []
    for step in range(max_steps):
        action, _ = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        step_rewards.append(reward)
        if terminated or truncated:
            break
    return step_rewards

def run_multiple_episodes(env, agent, num_episodes=20, max_steps=10000):
    all_episode_rewards = []
    for episode in range(num_episodes):
        rewards = run_episode(env, agent, max_steps)
        all_episode_rewards.append(rewards)
        print(f"Episode {episode + 1}/{num_episodes} completed")
    return all_episode_rewards

env = Gym2OpEnv()
ppo_agent = PPO.load("PPO_iteration_one\masked_ppo_simple.zip")

num_episodes = 20
all_episode_rewards = run_multiple_episodes(env, ppo_agent, num_episodes)

# Calculate average rewards for each step
max_steps = max(len(rewards) for rewards in all_episode_rewards)
avg_rewards = []
std_rewards = []

for step in range(max_steps):
    step_rewards = [rewards[step] if step < len(rewards) else 0 for rewards in all_episode_rewards]
    avg_rewards.append(np.mean(step_rewards))
    std_rewards.append(np.std(step_rewards))

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(range(1, max_steps + 1), avg_rewards, label='Average Reward')
plt.fill_between(range(1, max_steps + 1), 
                 np.array(avg_rewards) - np.array(std_rewards), 
                 np.array(avg_rewards) + np.array(std_rewards), 
                 alpha=0.2, label='Standard Deviation')
plt.title('Average Reward per Step over 100 Episodes')
plt.xlabel('Step')
plt.ylabel('Average Reward')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Calculate and print statistics
print("\nStatistics:")
print(f"Average Total Reward: {np.mean([sum(rewards) for rewards in all_episode_rewards]):.2f} ± {np.std([sum(rewards) for rewards in all_episode_rewards]):.2f}")
print(f"Average Episode Length: {np.mean([len(rewards) for rewards in all_episode_rewards]):.2f} ± {np.std([len(rewards) for rewards in all_episode_rewards]):.2f}")
print(f"Max Total Reward: {max(sum(rewards) for rewards in all_episode_rewards):.2f}")
print(f"Min Total Reward: {min(sum(rewards) for rewards in all_episode_rewards):.2f}")