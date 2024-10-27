import gymnasium as gym
from grid2op.PlotGrid import PlotMatplot
import grid2op
from grid2op import gym_compat
from grid2op.Parameters import Parameters
from grid2op.Action import PlayableAction
from grid2op.Observation import CompleteObservation
from grid2op.Reward import L2RPNReward, N1Reward, CombinedScaledReward
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from lightsim2grid import LightSimBackend
from grid2op.gym_compat import ContinuousToDiscreteConverter, BoxGymActSpace, BoxGymObsSpace
import numpy as np
import matplotlib.pyplot as plt
from grid2op.gym_compat import ScalerAttrConverter
from gymnasium.spaces import Box, Discrete


# Gymnasium environment wrapper around Grid2Op environment
class Gym2OpEnv(gym.Env):
    def __init__(
            self
    ):
        super().__init__()

        self._backend = LightSimBackend()
        self._env_name = "l2rpn_case14_sandbox"  # DO NOT CHANGE

        action_class = PlayableAction
        observation_class = CompleteObservation
        reward_class = CombinedScaledReward  # Setup further below

        # DO NOT CHANGE Parameters
        # See https://grid2op.readthedocs.io/en/latest/parameters.html
        p = Parameters()
        p.MAX_SUB_CHANGED = 4  # Up to 4 substations can be reconfigured each timestep
        p.MAX_LINE_STATUS_CHANGED = 4  # Up to 4 powerline statuses can be changed each timestep

        # Make grid2op env
        self._g2op_env = grid2op.make(
            self._env_name, backend=self._backend, test=False,
            action_class=action_class, observation_class=observation_class,
            reward_class=reward_class, param=p
        )

        self.max_possible = self._g2op_env.chronics_handler.max_timestep()

        ##########
        # REWARD #
        ##########
        # NOTE: This reward should not be modified when evaluating RL agent
        # See https://grid2op.readthedocs.io/en/latest/reward.html
        cr = self._g2op_env.get_reward_instance()
        cr.addReward("N1", N1Reward(), 1.0)
        cr.addReward("L2RPN", L2RPNReward(), 1.0)
        # reward = N1 + L2RPN
        cr.initialize(self._g2op_env)
        ##########

        self._gym_env = gym_compat.GymEnv(self._g2op_env)
        self.obs_gym, _ = self._gym_env.reset()

        self.setup_observations()
        self.setup_actions()

        self.observation_space = self._gym_env.observation_space
        self.action_space = self._gym_env.action_space

    def setup_observations(self):
        dim_obs_space = np.sum([np.sum(self._gym_env.observation_space[el].shape).astype(int)
                                for el in self._gym_env.observation_space.spaces])

        print(f"The size of the observation space is : "f"{dim_obs_space}")

        self._gym_env.observation_space = self._gym_env.observation_space.keep_only_attr(
            ["rho", "gen_p", "load_p", "topo_vect",
             "actual_dispatch"])
        new_dim_obs_space = np.sum([np.sum(self._gym_env.observation_space[el].shape).astype(int)
                                    for el in self._gym_env.observation_space.spaces])

        print(f"The new size of the observation space is : "f"{new_dim_obs_space} (it was {dim_obs_space} before!)")

        ob_space = self._gym_env.observation_space
        ob_space = ob_space.reencode_space("actual_dispatch",
                                           ScalerAttrConverter(substract=0.,
                                                               divide=self._g2op_env.gen_pmax
                                                               )
                                           )
        ob_space = ob_space.reencode_space("gen_p",
                                           ScalerAttrConverter(substract=0.,
                                                               divide=self._g2op_env.gen_pmax
                                                               )
                                           )
        ob_space = ob_space.reencode_space("load_p",
                                           ScalerAttrConverter(substract=self.obs_gym["load_p"],
                                                               divide=0.5 * self.obs_gym["load_p"]
                                                               )
                                           )

        shape_ = (self._g2op_env.dim_topo, self._g2op_env.dim_topo)
        self._gym_env.observation_space.add_key("connectivity_matrix",
                                                lambda obs: obs.connectivity_matrix(),
                                                Box(shape=shape_,
                                                    low=np.zeros(shape_),
                                                    high=np.ones(shape_),
                                                    )
                                                )
        self._gym_env.observation_space = ob_space

        self._gym_env.observation_space = BoxGymObsSpace(self._g2op_env.observation_space)

    def setup_actions(self):
        dim_act_space = np.sum(
            [np.sum(self._gym_env.action_space[el].shape) for el in self._gym_env.action_space.spaces])
        print(f"The size of the action space is : "f"{dim_act_space}")

        self._gym_env.action_space = self._gym_env.action_space.ignore_attr("set_bus").ignore_attr("set_line_status")

        new_dim_act_space = np.sum(
            [np.sum(self._gym_env.action_space[el].shape) for el in self._gym_env.action_space.spaces])
        print(f"The new size of the action space is : {new_dim_act_space}")

        self._gym_env.action_space = self._gym_env.action_space.reencode_space("redispatch",
                                                                               ContinuousToDiscreteConverter(nb_bins=8))

        self._gym_env.action_space = BoxGymActSpace(
            self._g2op_env.action_space,
        )

    def reset(self, seed=None, options=None):
        obs, info = self._gym_env.reset(seed=seed, options=options)

        return obs, info

    def step(self, action):
        return self._gym_env.step(action)

    def render(self):
        return self._gym_env.render()


env = Gym2OpEnv()
print(f"The \"env_gym\" is a gym environment: {isinstance(env, gym.Env)}")

print(env.action_space)
# Print observation space
print(env.observation_space)