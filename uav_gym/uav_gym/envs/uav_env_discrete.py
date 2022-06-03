from typing import Tuple

import gym
from gym.utils import seeding
from uav_gym.envs.env_settings import Settings
import numpy as np
from sklearn.datasets import make_blobs
import uav_gym.utils as gym_utils
import matplotlib.pyplot as plt


class UAVCoverage(gym.Env):
    def __init__(self):
        self.sg = Settings()

        self.seed()

        # ----
        # SIMULATION
        self.sim_size = self.sg.V['SIM_SIZE']
        self.scale = self.sg.V['SCALE']

        self.home_loc = self.sg.V['HOME_LOC']

        # ----
        # UAV
        self.n_uavs = self.sg.V['NUM_UAV']
        self.cov_range = self.sg.V['COV_RANGE']
        self.comm_range = self.sg.V['COMM_RANGE']

        self.dist = self.sg.V['DIST']

        # ----
        # USERS
        self.n_users = self.sg.V['NUM_USERS']

        self.b_factor = self.sg.V['BOUNDARY_FACTOR']
        self.n_clusters = 2

        self.cov_scores = None
        self.pref_users = None
        self.pref_factor = 2

        # ----
        # SPACES
        self.action_space = self._action_space_0()

        # locs are the locations of each UAV or user in the form [x1, y1, x2, y2]
        self.observation_space = self._observation_space_0()

        self.state = None
        self.timestep = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = {
            'uav_locs': np.array([self.sg.V['INIT_POSITION'] for _ in range(self.n_uavs)]).flatten(),
            'user_locs': gym_utils.scale(self._gen_user_locs_1(), s=self.scale, d='down').flatten()
        }

        self.cov_scores = np.array([0] * self.n_users)
        self.pref_users = self.np_random.choice([0, 1], size=(self.n_users,), p=[4. / 5, 1. / 5])

        self.timestep = 0

        return self.state

    def step(self, action: np.ndarray):
        done = False
        self.timestep += 1

        # unpack state and convert units to metres.
        uav_locs_ = gym_utils.scale(self.state['uav_locs'], self.scale, 'up')
        user_locs_ = gym_utils.scale(self.state['user_locs'], self.scale, 'up')

        # convert the lists of locs of the form [x1, y1, x2, y2] to [[x1, y1], [x2, y2]]
        uav_locs = gym_utils.conv_locs(uav_locs_)
        user_locs = gym_utils.conv_locs(user_locs_)

        distances = np.array([self.dist] * len(action))
        # ---
        # move UAVs
        moves = np.array(list(map(gym_utils.get_move, action, distances)))

        # maybe_locs is a list of the new positions regardless of whether they are out of bounds
        maybe_locs = uav_locs + moves

        new_locs = np.array(
            [maybe_locs[i]
             if gym_utils.inbounds(maybe_locs[i], self.sim_size, self.sim_size)
             else
             uav_locs[i]
             for i in range(len(moves))],
            dtype=np.float32)

        # update state
        self.state['uav_locs'] = gym_utils.scale(np.array(new_locs.flatten(), dtype=np.float32), self.scale, 'down')

        # update cov_scores
        self.cov_scores += gym_utils.get_coverage_state(new_locs, user_locs, self.cov_range)

        # ---
        # NOTE: reward calc needs to come after self.cov_scores update.
        # calculate reward = sum of all scores
        reward = self.reward_1(new_locs, user_locs)

        # TODO: Check this is the correct value and that it agrees with animation.
        # stop after 30 minutes where each timestep is 1 second.
        if self.timestep >= 1800:
            done = True

        info = {}

        return self.state, reward, done, info

    def render(self, mode="human"):
        uav_locs_ = gym_utils.scale(self.state['uav_locs'], self.scale, 'up')
        user_locs_ = gym_utils.scale(self.state['user_locs'], self.scale, 'up')

        uav_locs = [uav_locs_[::2], uav_locs_[1::2]]
        user_locs = [user_locs_[::2], user_locs_[1::2]]

        # Render the environment to the screen
        plt.xlim([0, self.sim_size])
        plt.ylim([0, self.sim_size])

        plt.scatter(uav_locs[0], uav_locs[1], s=20000, color='blue', alpha=0.3)
        plt.scatter(uav_locs[0], uav_locs[1], color='red')

        plt.scatter(user_locs[0], user_locs[1], color='grey', s=2)
        plt.xlabel("X cordinate")
        plt.ylabel("Y cordinate")
        plt.pause(0.001)
        plt.show()
        plt.clf()

    def _gen_user_locs_0(self):
        # TODO: Random cluster_stds?
        # TODO: How to do random_state? Using sim_size doesn't really make sense.
        std = float(0.05 * self.sim_size)

        stds = tuple([std] * self.n_clusters)

        centers = self.get_centers(stds)

        ul_init, blob_ids = make_blobs(n_samples=self.n_users, centers=centers, cluster_std=stds,
                                       random_state=self.np_random.randint(2 ** 32 - 1))

        # TODO: Should I sort the user_locs? Would it make it easier to find a solution?
        return gym_utils.constrain_user_locs(ul_init, blob_ids, centers, stds, self.np_random)

    def _gen_user_locs_1(self):
        """
        User locations are sorted
        :return:
        """
        ul_locs = self._gen_user_locs_0()

        return np.array(sorted(ul_locs.tolist()))

    def get_centers(self, stds: Tuple[float]):
        assert all([self.b_factor * std < self.sim_size for std in stds])

        centers = [
            [self.np_random.uniform(0 + 3 * std, self.sim_size - 3 * std)
             for std in stds]
            for _ in range(self.n_clusters)
        ]

        return centers

    def _action_space_0(self):
        return gym.spaces.MultiDiscrete(
            np.array([5] * self.n_uavs, dtype=np.int32)
        )

    def _observation_space_0(self):
        return gym.spaces.Dict({
            'uav_locs': gym.spaces.MultiDiscrete(
                np.array([self.sim_size // self.scale + 1] * 2 * self.n_uavs, dtype=np.int32)),
            'user_locs': gym.spaces.MultiDiscrete(
                np.array([self.sim_size // self.scale + 1] * 2 * self.n_users, dtype=np.int32)),
        })

    def reward_0(self, uav_locs, user_locs):
        """
        Basic
        """
        total_score = sum(
            gym_utils.get_scores(
                uav_locs,
                user_locs,
                self.cov_range,
                p_factor=self.sg.V['P_OUTSIDE_COV']
            )
        )

        return total_score / self.n_users

    def reward_1(self, uav_locs, user_locs):
        """
        Basic + punishment for disconnection
        """
        total_score = sum(
            gym_utils.get_scores(
                uav_locs,
                user_locs,
                self.cov_range,
                p_factor=self.sg.V['P_OUTSIDE_COV']
            )
        )
        reward = total_score / self.n_users

        graph = gym_utils.make_graph_from_locs(uav_locs, self.home_loc, self.comm_range)
        discon_count = gym_utils.get_disconnected_count(graph)

        return reward - 1000 * reward * discon_count

    def reward_2(self, uav_locs, user_locs):
        f_idx = gym_utils.fairness_idx(self.cov_scores / self.timestep)

        return f_idx * self.reward_0(uav_locs, user_locs)

    def reward_3(self, uav_locs, user_locs):
        scores = gym_utils.get_scores(
            uav_locs,
            user_locs,
            self.cov_range,
            p_factor=self.sg.V['P_OUTSIDE_COV']
        )

        # increase the scores of the preferred users by a factor of self.pref_factor.
        scaled_scores = scores + (self.pref_factor - 1) * self.pref_users * scores

        total_score = sum(scaled_scores)

        return total_score / self.n_users


if __name__ == '__main__':
    env = UAVCoverage()

    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env

    check_env(env)

    obs = env.reset()
    n_steps = 100
    for _ in range(n_steps):
        # Random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(obs)
        print(reward)
        if done:
            obs = env.reset()
        env.render()

    # model = PPO('MultiInputPolicy', env, verbose=1)
    # model.learn(total_timesteps=10**5)
    #
    # obs = env.reset()
    # env.seed(0)
    # locs = []
    # for _ in range(200):
    #     action, _states = model.predict(obs)
    #     obs, rewards, done, info = env.step(action)
    #     env.render()
