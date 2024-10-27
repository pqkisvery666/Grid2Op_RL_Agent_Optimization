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
            
            # Track invalid actions
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
    Create visualizations of the evaluation results
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Episode Returns Distribution
    sns.histplot(stats["returns"], ax=ax1)
    ax1.set_title("Distribution of Episode Returns")
    ax1.set_xlabel("Return")
    ax1.set_ylabel("Count")
    
    # Episode Lengths Distribution
    sns.histplot(stats["lengths"], ax=ax2)
    ax2.set_title("Distribution of Episode Lengths")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Count")
    
    # Invalid Actions per Episode
    sns.barplot(x=list(range(len(stats["invalid_actions"]))), 
                y=stats["invalid_actions"], ax=ax3)
    ax3.set_title("Invalid Actions per Episode")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Number of Invalid Actions")
    
    # Termination Reasons
    reason_counts = pd.Series(stats["termination_reasons"]).value_counts()
    reason_counts.plot(kind='bar', ax=ax4)
    ax4.set_title("Episode Termination Reasons")
    ax4.set_xlabel("Reason")
    ax4.set_ylabel("Count")
    
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

# Main execution
if __name__ == "__main__":
    # Load the agent
    ppo_agent = PPO.load("PPO_baseline\ppo_Iteration1.zip")
    
    # Create environment
    env = Gym2OpEnv()
    
    # Run evaluation
    print("Starting evaluation...")
    stats = evaluate_agent(env, ppo_agent, num_episodes=100, max_steps=10000)
    
    # Print statistics
    print_summary_statistics(stats)
    
    # Create and save plots
    fig = plot_evaluation_results(stats)
    plt.savefig('evaluation_results.png')
    plt.close()