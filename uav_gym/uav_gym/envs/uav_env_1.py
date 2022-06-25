from typing import Tuple

import gym
from gym.utils import seeding
# from uav_gym.uav_gym.envs.env_settings import Settings
# import uav_gym.uav_gym.utils as gym_utils

from uav_gym.envs.env_settings import Settings
import uav_gym.utils as gym_utils

import numpy as np
import math
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


class UAVCoverage_old_reward(gym.Env):
    def __init__(self, n_uavs: int = None, demonstration: bool = False):
        self.sg = Settings()

        self.seed()

        self.dem = demonstration
        # ----
        # SIMULATION
        self.sim_size = self.sg.V['SIM_SIZE']
        self.home_loc = self.sg.V['HOME_LOC']

        # ----
        # UAV
        if n_uavs is not None:
            self.n_uavs = n_uavs
        else:
            self.n_uavs = self.sg.V['NUM_UAV']
        self.cov_range = self.sg.V['COV_RANGE']
        self.comm_range = self.sg.V['COMM_RANGE']

        self.dist = self.sg.V['DIST']

        # ----
        # USERS
        self.n_users = self.sg.V['NUM_USERS']

        self.b_factor = self.sg.V['BOUNDARY_FACTOR']
        self.n_clusters = None

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
        if self.dem:
            user_locs = np.array(self.sg.V['USER_LOCS'])
        else:
            user_locs = np.array(self._gen_user_locs(), dtype=np.float64)

        self.state = self.normalize_obs(
            {
                'uav_locs': np.array([self.sg.V['INIT_POSITION'] for _ in range(self.n_uavs)], dtype=np.float64),
                'user_locs': user_locs,
                'pref_users': self.np_random.choice([0, 1], size=(self.n_users,), p=[4. / 5, 1. / 5]).astype(np.int32),
                'cov_scores': np.array([0] * self.n_users, dtype=np.float64)
            }
        )

        self.timestep = 0

        return self.state

    def step(self, action: np.ndarray):
        # gym wrapper will make done = True after 1800 timesteps
        done = False
        self.timestep += 1

        state = self.denormalize_obs(self.state)

        # unpack state
        uav_locs = state['uav_locs']
        user_locs = state['user_locs']

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
            dtype=np.float64)

        cov_state = gym_utils.get_coverage_state(new_locs.tolist(), user_locs.tolist(), self.cov_range)
        prev_cov_scores = state['cov_scores'].copy()
        new_cov_scores = state['cov_scores'] * (self.timestep - 1) / self.timestep + cov_state / self.timestep
        # update state
        self.state = self.normalize_obs(
            {
                'uav_locs': new_locs,
                'user_locs': user_locs,
                'pref_users': state['pref_users'],
                'cov_scores': new_cov_scores
            }
        )

        # ---
        # NOTE: reward calc needs to come after self.cov_scores update because of fairness calculation.
        reward = self.reward_2(prev_cov_scores, maybe_locs)

        info = {}

        return self.state, reward, done, info

    def render(self, mode="human"):
        state = self.denormalize_obs(self.state)

        # unpack state
        uav_locs_ = state['uav_locs']
        user_locs_ = state['user_locs']

        uav_locs = list(zip(*uav_locs_))
        user_locs = list(zip(*user_locs_))

        # set up figures and axes
        # ---
        fig, ax = plt.subplots()

        # set figure size
        fig.set_size_inches(6, 6)

        # set axes to always be equal
        plt.axis('scaled')

        # Render the environment to the screen
        plt.xlim([0, self.sim_size])
        plt.ylim([0, self.sim_size])

        circles = [plt.Circle((0, 0), self.cov_range, fc='b', alpha=0.3) for _ in range(self.n_uavs)]
        # add coverage of UAVs
        for circle in circles:
            ax.add_patch(circle)

        # Update UAV coverage positions
        for circle, x, y in zip(circles, uav_locs[0], uav_locs[1]):
            circle.center = (x, y)

        plt.scatter(uav_locs[0], uav_locs[1], color='red')

        plt.scatter(user_locs[0], user_locs[1], color='grey', s=2)
        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.show()

    def _gen_user_locs(self):
        self.n_clusters = self.np_random.randint(1, 4)

        stds = self.get_stds()

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

        ul_locs = gym_utils.constrain_user_locs(ul_init.tolist(),
                                                blob_ids, centers, stds, self.b_factor, self.np_random).tolist()

        return np.array(sorted(ul_locs), dtype=np.float64)

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

        return np.array(centers)

    def _action_space_0(self):
        return gym.spaces.MultiDiscrete(
            np.array([5] * self.n_uavs, dtype=np.int32)
        )

    def _observation_space_0(self):
        return gym.spaces.Dict({
            'uav_locs': gym.spaces.Box(low=-1, high=1, shape=(self.n_uavs, 2), dtype=np.float64),
            'user_locs': gym.spaces.Box(low=-1, high=1, shape=(self.n_users, 2), dtype=np.float64),
            'pref_users': gym.spaces.MultiBinary(self.n_users),
            'cov_scores': gym.spaces.Box(low=-1, high=1, shape=(self.n_users,), dtype=np.float64)
        })

    def normalize_obs(self, obs):
        sig_figs = math.floor(math.log(self.sim_size, 10))
        return {
            'uav_locs': (obs['uav_locs'] / (self.sim_size / 2) - 1).round(sig_figs),
            'user_locs': obs['user_locs'] / (self.sim_size / 2) - 1,
            'pref_users': obs['pref_users'],
            'cov_scores': obs['cov_scores'] * 2 - 1
        }

    def denormalize_obs(self, obs):
        return {
            'uav_locs': ((obs['uav_locs'] + 1) * (self.sim_size / 2)).round(0),
            'user_locs': (obs['user_locs'] + 1) * (self.sim_size / 2),
            'pref_users': obs['pref_users'],
            'cov_scores': (obs['cov_scores'] + 1) / 2
        }

    def reward_1(self, prev_cov_score):
        """
        Includes user scores, fairness, and user prioritisation
        """
        state = self.denormalize_obs(self.state)

        # unpack state
        uav_locs = state['uav_locs']
        user_locs = state['user_locs']
        pref_users = state['pref_users']
        cov_scores = state['cov_scores']

        # # calculate fairness
        # f_idx = gym_utils.fairness_idx(cov_scores)
        #
        # # calculate improvement in total coverage
        # delta_cov_score = cov_scores - prev_cov_score
        #
        # # calculate distance from each user to the closest UAV.
        # dist = gym_utils.dist_to_users(uav_locs.tolist(), user_locs.tolist())
        # dist_scores = 1 / (1 + dist)
        #
        # def scale_scores(scores):
        #     return scores + (self.pref_factor - 1) * pref_users * scores
        #
        # # both parts have a max values of n_users and a min value of 0.
        # cov_part = f_idx * sum(scale_scores(delta_cov_score)) / self.n_users
        # dist_part = sum(scale_scores(dist_scores)) / self.n_users
        #
        # return cov_part + self.sg.V['P_OUTSIDE_COV'] * dist_part

        scores = gym_utils.get_scores(
            uav_locs.tolist(),
            user_locs.tolist(),
            self.cov_range,
            p_factor=self.sg.V['P_OUTSIDE_COV']
        )

        f_idx = gym_utils.fairness_idx(cov_scores / self.timestep)

        # increase the scores of the preferred users by a factor of self.pref_factor.
        scaled_scores = scores + (self.pref_factor - 1) * pref_users * scores

        mean_score = sum(scaled_scores) / self.n_users

        return f_idx * mean_score

    def reward_2(self, prev_cov_score, maybe_uav_locs):
        """
        Include constant penalty for disconnecting or going out of bounds
        :param prev_cov_score: the coverage score from the previous timestep.
        :param maybe_uav_locs: positions the UAVs tried to move to.
        """
        uav_locs = self.denormalize_obs(self.state)['uav_locs']

        reward = self.reward_1(prev_cov_score)
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
    env = UAVCoverage(demonstration=True)

    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env

    check_env(env)

    obs = env.reset()
    print(env.denormalize_obs(obs)['user_locs'].tolist())
    n_steps = 100
    for _ in range(n_steps):
        # Random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(reward)
        # print(env.denormalize_obs(obs))
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
    #     print(rewards)
    #     env.render()
