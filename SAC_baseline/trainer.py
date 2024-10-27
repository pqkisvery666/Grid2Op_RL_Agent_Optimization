from stable_baselines3 import SAC
from env import Gym2OpEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor


logdir = "SAC_logs"
def train_sac_agent(env, total_timesteps=200000):
    env = DummyVecEnv([lambda: env])

    model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
    model.learn(total_timesteps=total_timesteps)

    model.save("sac_baseline_model")

    return model


env = Gym2OpEnv()
env = Monitor(env, logdir)
sac_agent = train_sac_agent(env)