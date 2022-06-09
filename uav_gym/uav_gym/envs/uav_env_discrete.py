from typing import Tuple

import gym
from gym.utils import seeding
# from uav_gym.uav_gym.envs.env_settings import Settings
# import uav_gym.uav_gym.utils as gym_utils

from uav_gym.envs.env_settings import Settings
import uav_gym.utils as gym_utils

import numpy as np
from sklearn.datasets import make_blobs
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
        self.n_clusters = self.np_random.randint(1, 4)

        self.pref_users = None
        self.pref_factor = 2

        # ----
        # SPACES
        self.action_space = self._action_space_0()

        # locs are the locations of each UAV or user in the form [x1, y1, x2, y2]
        self.observation_space = self._observation_space_0()

        self.cov_scores = [0] * self.n_users

        self.state = None
        self.timestep = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = {
            'uav_locs': gym_utils.conv_locs([self.sg.V['INIT_POSITION'] for _ in range(self.n_uavs)],
                                            s=self.scale, from_state=False),
            'user_locs': gym_utils.conv_locs(self._gen_user_locs().tolist(), s=self.scale, from_state=False),
        }

        self.pref_users = self.np_random.choice([0, 1], size=(self.n_users,), p=[4. / 5, 1. / 5])

        self.timestep = 0

        return self.state

    def step(self, action: np.ndarray):
        done = False
        self.timestep += 1

        # convert the lists of locs of the form [x1, y1, x2, y2] to [[x1, y1], [x2, y2]]
        # unpack state and convert units to metres.
        uav_locs = gym_utils.conv_locs(self.state['uav_locs'].tolist(), self.scale, from_state=True)
        user_locs = gym_utils.conv_locs(self.state['user_locs'].tolist(), self.scale, from_state=True)

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
            dtype=int)

        # update state
        self.state['uav_locs'] = gym_utils.conv_locs(new_locs.tolist(), self.scale, from_state=False)
        self.cov_scores += gym_utils.get_coverage_state(new_locs.tolist(), user_locs.tolist(), self.cov_range)

        # ---
        # NOTE: reward calc needs to come after self.cov_scores update because of fairness calculation.
        reward = self.reward_2(maybe_locs)

        # stop after 30 minutes where each timestep is 1 second.
        if self.timestep >= self.sg.V['MAX_TIMESTEPS']:
            done = True

        info = {}

        return self.state, reward, done, info

    def render(self, mode="human"):
        uav_locs_ = gym_utils.scale(self.state['uav_locs'].tolist(), self.scale, from_state=True)
        user_locs_ = gym_utils.scale(self.state['user_locs'].tolist(), self.scale, from_state=True)

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

    def _gen_user_locs(self):
        stds = self.get_stds()  # TODO: Should this be uniformly sampled? Why these values?

        centers = self.get_centers(stds)

        densities = self.np_random.uniform(low=0.5, high=1.0, size=self.n_clusters)

        n_samples = (self.n_users * (stds * densities) / sum(stds * densities))

        # the following rounds n_samples to integers while ensuring the sum equals self.n_users.
        ints = n_samples.astype(int)
        rems = n_samples - n_samples.astype(int)
        diff = self.n_users - sum(ints)

        inds = np.argpartition(rems, -diff)[-diff:]
        m = np.zeros(self.n_clusters).astype(int)
        m[inds] = 1

        if diff > 0:
            n_samples = ints + m
        else:
            n_samples = ints

        ul_init, blob_ids = make_blobs(n_samples=n_samples, centers=centers, cluster_std=stds,
                                       random_state=self.np_random.randint(2 ** 32 - 1))

        ul_locs = gym_utils.constrain_user_locs(ul_init.tolist(), blob_ids, centers, stds, self.b_factor, self.np_random)\
            .round(0).astype(int).tolist()

        return np.array(sorted(ul_locs))

    def get_stds(self):
        samples = self.np_random.poisson(6, size=10000).astype(float)
        samples *= self.sim_size / 2 / 15 / self.b_factor
        samples += 50

        stds = self.np_random.choice(samples, size=self.n_clusters)

        while any(stds * self.b_factor > self.sim_size / 2) or any(stds == 0):
            stds = self.np_random.choice(samples, size=self.n_clusters)

        return stds

    def get_centers(self, stds: Tuple[float]):
        assert all([self.b_factor * std <= self.sim_size / 2 for std in stds])

        centers = [
            [self.np_random.uniform(self.b_factor * std, self.sim_size - self.b_factor * std)
             for _ in range(2)]
            for std in stds
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

    def _observation_space_1(self):
        return gym.spaces.Dict({
            'uav_locs': gym.spaces.MultiDiscrete(
                np.array([self.sim_size // self.scale + 1] * 2 * self.n_uavs, dtype=np.int32)),
            'user_locs': gym.spaces.MultiDiscrete(
                np.array([self.sim_size // self.scale + 1] * 2 * self.n_users, dtype=np.int32)),
            'cov_scores': gym.spaces.MultiDiscrete(
                np.array([self.sg.V['MAX_TIMESTEPS']] * self.n_users, dtype=np.int32)),
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

    def reward_1(self):
        """
        Includes user scores, fairness, and user prioritisation
        """
        uav_locs = gym_utils.conv_locs(self.state['uav_locs'].tolist(), s=self.scale, from_state=True)
        user_locs = gym_utils.conv_locs(self.state['user_locs'].tolist(), s=self.scale, from_state=True)

        scores = gym_utils.get_scores(
            uav_locs.tolist(),
            user_locs.tolist(),
            self.cov_range,
            p_factor=self.sg.V['P_OUTSIDE_COV']
        )

        f_idx = gym_utils.fairness_idx(self.cov_scores / self.timestep)

        # increase the scores of the preferred users by a factor of self.pref_factor.
        scaled_scores = scores + (self.pref_factor - 1) * self.pref_users * scores

        mean_score = sum(scaled_scores) / self.n_users

        return f_idx * mean_score

    def reward_2(self, maybe_uav_locs):
        """
        Include constant penalty for disconnecting or going out of bounds
        :param maybe_uav_locs: positions the UAVs tried to move to.
        """
        uav_locs = gym_utils.conv_locs(self.state['uav_locs'].tolist(), s=self.scale, from_state=True)

        reward = self.reward_1()

        graph = gym_utils.make_graph_from_locs(uav_locs.tolist(), self.home_loc, self.comm_range)
        dconnect_count = gym_utils.get_disconnected_count(graph)

        p_dconnect = self.sg.V['P_DISCONNECT'] * dconnect_count

        outside_count = self.n_uavs - sum(
            [gym_utils.inbounds(loc, x_ubound=self.sim_size, y_ubound=self.sim_size)
             for loc in maybe_uav_locs
             ]
        )

        p_outside = self.sg.V['P_OUT_BOUNDS'] * outside_count

        return reward - p_dconnect - p_outside


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
