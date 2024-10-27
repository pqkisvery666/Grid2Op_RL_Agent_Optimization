from env import Gym2OpEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

logdir = "PPO_logs/logs"

def train_ppo_agent(env, total_timesteps=1000000):

    env = DummyVecEnv([lambda: env])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=logdir
        )
    
    model.learn(total_timesteps=total_timesteps)
    
    model.save("PPO_iteration_one\ppo_baseline.zip")

    return model

env = Gym2OpEnv()
env = Monitor(env, logdir)

train_ppo_agent(env, total_timesteps=1000000)