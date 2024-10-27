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
from grid2op.gym_compat import BoxGymObsSpace, DiscreteActSpace, BoxGymActSpace
import numpy as np
import matplotlib.pyplot as plt


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

        self.setup_observations()
        self.setup_actions()

        self.observation_space = self._gym_env.observation_space
        self.action_space = self._gym_env.action_space

    def parse_gym_observation(self, obs): #ignore these were for understanding of the spaces
        """
        Parse the flattened Gym observation array into its constituent parts.

        Args:
            obs (numpy.ndarray): The flattened Gym observation array

        Returns:
            dict: Dictionary containing parsed observation components
        """
        idx = 0

        line_status = obs[idx:idx + 20]
        idx += 20

        load_p = obs[idx:idx + 11]
        idx += 11

        prod_p = obs[idx:idx + 6]
        idx += 6

        rho = obs[idx:idx + 20]
        idx += 20

        topo_vect = obs[idx:idx + 57]

        return {
            'line_status': line_status,
            'load_p': load_p,
            'prod_p': prod_p,
            'rho': rho,
            'topo_vect': topo_vect
        }

    def print_observation_details(self, obs):
        """
        Print detailed information about each component of the observation.

        Args:
            obs (numpy.ndarray): The flattened Gym observation array
        """
        parsed_obs = self.parse_gym_observation(obs)

        
        return parsed_obs

    def setup_observations(self):
        self._gym_env.observation_space = BoxGymObsSpace(
            self._g2op_env.observation_space
        )

    def setup_actions(self):
        self._gym_env.action_space = BoxGymActSpace(
            self._g2op_env.action_space,
        )

    def reset(self, seed=None, options=None):
        obs, info = self._gym_env.reset(seed=seed, options=options)
        parsed_obs = self.print_observation_details(obs)
        # rho = parsed_obs['rho']
        # print("rho:", rho)

        return obs, info

    def step(self, action):
        return self._gym_env.step(action)

    def render(self):
        return self._gym_env.render()