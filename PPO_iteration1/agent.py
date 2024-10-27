from env import Gym2OpEnv 
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pandas as pd
import seaborn as sns

def evaluate_agent(env, agent, num_episodes=10, max_steps=10000):
    """
    Evaluate the agent over multiple episodes and return statistics
    """
    episode_returns = []
    episode_lengths = []
    invalid_actions_per_episode = []
    termination_reasons = []
    all_rewards = []
    
    for episode in tqdm(range(num_episodes), desc="Evaluating episodes"):
        curr_step = 0
        curr_return = 0
        invalid_actions = 0
        
        is_done = False
        obs, info = env.reset()
        
        while not is_done and curr_step < max_steps:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            curr_step += 1
            curr_return += reward
            all_rewards.append(reward)
            
            is_action_valid = not (info["is_illegal"] or info["is_ambiguous"])
            if not is_action_valid:
                invalid_actions += 1
            
            if terminated or truncated:
                is_done = True
                if terminated:
                    termination_reasons.append("terminated")
                elif truncated:
                    termination_reasons.append("truncated")
                break
        
        if not is_done:
            termination_reasons.append("max_steps")
            
        episode_returns.append(curr_return)
        episode_lengths.append(curr_step)
        invalid_actions_per_episode.append(invalid_actions)
    
    stats = {
        "returns": episode_returns,
        "lengths": episode_lengths,
        "invalid_actions": invalid_actions_per_episode,
        "termination_reasons": termination_reasons,
        "all_rewards": all_rewards
    }
    
    return stats

def plot_evaluation_results(stats):
    """
    Create visualization of returns over episodes
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.lineplot(x=range(len(stats["returns"])), y=stats["returns"], ax=ax, label='Returns', color='blue')
    
    ax.set_title("Returns Over Episodes", fontsize=12)
    ax.set_xlabel("Episode", fontsize=10)
    ax.set_ylabel("Return", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def print_summary_statistics(stats):
    """
    Print comprehensive summary statistics
    """
    print("\n###################")
    print("#     SUMMARY     #")
    print("###################")
    print(f"Number of episodes: {len(stats['returns'])}")
    print("\nReturns:")
    print(f"  Mean: {np.mean(stats['returns']):.2f}")
    print(f"  Std: {np.std(stats['returns']):.2f}")
    print(f"  Min: {np.min(stats['returns']):.2f}")
    print(f"  Max: {np.max(stats['returns']):.2f}")
    
    print("\nEpisode Lengths:")
    print(f"  Mean: {np.mean(stats['lengths']):.2f}")
    print(f"  Std: {np.std(stats['lengths']):.2f}")
    print(f"  Min: {np.min(stats['lengths'])}")
    print(f"  Max: {np.max(stats['lengths'])}")
    
    print("\nInvalid Actions:")
    print(f"  Total: {sum(stats['invalid_actions'])}")
    print(f"  Mean per episode: {np.mean(stats['invalid_actions']):.2f}")
    print(f"  Std per episode: {np.std(stats['invalid_actions']):.2f}")
    
    print("\nReward Statistics:")
    print(f"  Mean reward per step: {np.mean(stats['all_rewards']):.4f}")
    print(f"  Reward std: {np.std(stats['all_rewards']):.4f}")
    
    print("\nTermination Reasons:")
    for reason, count in pd.Series(stats['termination_reasons']).value_counts().items():
        print(f"  {reason}: {count}")
    print("###################")

if __name__ == "__main__":
  
    
    env = Gym2OpEnv()
    
    ppo_agent = PPO.load("PPO_iteration1\ppo_Iteration1.zip")
    
    print("Starting evaluation...")
    stats = evaluate_agent(env, ppo_agent, num_episodes=100, max_steps=10000)
    
    print_summary_statistics(stats)
    
    fig = plot_evaluation_results(stats)
    plt.savefig('evaluation_results_ppo_Iteration1.png')
    plt.close()