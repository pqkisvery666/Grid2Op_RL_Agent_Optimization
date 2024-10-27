import matplotlib.pyplot as plt
from IPython import display
import imageio
import os
from env import Gym2OpEnv
from stable_baselines3 import PPO

env = Gym2OpEnv()

max_steps = 10000

ppo_agent = PPO.load("PPO_baseline\ppo_baseline_model.zip")

def run_simulation_and_create_gif(env, ppo_agent, max_steps, output_gif_path):
    images = []
    curr_step = 0
    curr_return = 0
    is_done = False
    obs, info = env.reset()

    while not is_done and curr_step < max_steps:
        action, _ = ppo_agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        fig = env.render()
        img = fig2img(plt.gcf())
        images.append(img)
        
        curr_step += 1
        curr_return += reward
        is_done = terminated or truncated

        print(f"step = {curr_step}: ")
    # Create GIF

    imageio.mimsave(output_gif_path, images, fps=10)
    
    plt.imshow(images[-2])
    
    print(f"GIF saved to {output_gif_path}")

def fig2img(fig):
    import io
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img.convert('RGB')

output_gif_path = "grid_state_simulation.gif"
run_simulation_and_create_gif(env, ppo_agent, max_steps, output_gif_path)