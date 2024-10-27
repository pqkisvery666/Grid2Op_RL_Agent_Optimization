from env import Gym2OpEnv 
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
from IPython import display

env = Gym2OpEnv()

ppo_agent = PPO.load("PPO_baseline\ppo_baseline_model_50000.zip")

max_steps = 10000

curr_step = 0
curr_return = 0

is_done = False
obs, info = env.reset()
print(f"step = {curr_step} (reset):")
print(f"\t obs = {obs}")
print(f"\t info = {info}\n\n")

while not is_done and curr_step < max_steps:
    # Use the PPO agent to select an action
    action, _ = ppo_agent.predict(obs, deterministic=True)
    
    obs, reward, terminated, truncated, info = env.step(action)
    curr_step += 1
    curr_return += reward
    is_done = terminated or truncated

    print(f"step = {curr_step}: ")
    print(f"\t obs = {obs}")
    print(f"\t reward = {reward}")
    print(f"\t terminated = {terminated}")
    print(f"\t truncated = {truncated}")
    print(f"\t info = {info}")

    # Some actions are invalid (see: https://grid2op.readthedocs.io/en/latest/action.html#illegal-vs-ambiguous)
    # Invalid actions are replaced with 'do nothing' action
    is_action_valid = not (info["is_illegal"] or info["is_ambiguous"])
    print(f"\t is action valid = {is_action_valid}")
    if not is_action_valid:
        print(f"\t\t reason = {info['exception']}")
    print("\n")
    
# plt.imshow(env.render())
# plt.show()
print("###########")
print("# SUMMARY #")
print("###########")
print(f"return = {curr_return}")
print(f"total steps = {curr_step}")
print("###########")