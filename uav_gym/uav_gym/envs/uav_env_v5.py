import gym
from gym.utils import seeding
import numpy as np
from sklearn.datasets import make_blobs
import uav_gym.utils as gym_utils
import matplotlib.pyplot as plt


class UAVCoverage(gym.Env):
    n_users = 15

    # 1 unit = 100 m

    # ----
    # UAV SETTINGS
    # radius of the coverage range on the ground in units
    cov_range = 2  # units

    # uav velocity (assumes constant velocity)
    uav_vel = 1 / 9  # units / s
    dist = 1

    # battery capacity
    uav_bat_cap = 180

    # ----
    # Simulation settings
    time_per_epoch = 1  # seconds

    sim_size = 10
    sim_size_ = sim_size / dist

    def __init__(self, n_uavs: int = 5):
        self.seed()

        self.user_centres = []
        self.user_locs = []
        # TODO: Random number of UAVs?
        self.n_uavs = n_uavs

        self.action_space = gym.spaces.MultiDiscrete(
            np.array([5] * self.n_uavs, dtype=np.int32)
        )

        # uav_locs is the locations of each UAV in the form [x1, y1, x2, y2]
        # self.observation_space = gym.spaces.MultiDiscrete(
        #     np.array([self.sim_size * 1 / self.dist + 1] * 2 * self.n_uavs,
        #              dtype=np.int32)
        # )

        self.observation_space = gym.spaces.Dict({
            'uav_locs': gym.spaces.MultiDiscrete(
                np.array([self.sim_size + 1] * 2 * self.n_uavs, dtype=np.int32)),
            'user_locs': gym.spaces.MultiDiscrete(
                np.array([self.sim_size + 1] * 2 * self.n_users, dtype=np.int32)),
        })

        self.energy_used = None
        self.state = None
        self.timestep = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _gen_user_locs(self):
        # TODO: Random cluster_stds?
        # TODO: How to do random_state? Using sim_size doesn't really make sense.
        std = 0.5
        # centers must be within one std of the border.
        center = [self.np_random.uniform(0 + 3 * std, self.sim_size - 3 * std) for _ in range(2)]
        ul_init, _ = make_blobs(n_samples=self.n_users, centers=[center], cluster_std=[std])

        # constrain users to be with 3 stds of the centre and round locations to a whole number.
        ul_constr = np.array([gym_utils.constrain_user_loc(user_loc, center, std, self.np_random)
                              for user_loc in ul_init]).round(0)

        return ul_constr

    def reset(self):
        self.state = {
            'uav_locs': np.array([0] * 2 * self.n_uavs),
            'user_locs': self._gen_user_locs().flatten()
        }

        self.timestep = 0

        self.energy_used = np.array([0] * self.n_uavs, dtype=np.float32)

        return self.state

    def step(self, action: np.ndarray):
        done = False
        self.timestep += 1

        # unpack state
        uav_locs_ = self.state['uav_locs']
        user_locs_ = self.state['user_locs']

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
            [maybe_locs[i] if gym_utils.inbounds(maybe_locs[i]) else uav_locs[i] for i in range(len(moves))],
            dtype=np.float32).flatten()

        # ---
        # calculate energy usage
        # energy used moving + energy used hovering
        time_moving = distances / self.uav_vel
        time_hovering = self.time_per_epoch - time_moving
        energy_usage = np.array(list(map(gym_utils.energy_move, time_moving))) + np.array(
            list(map(gym_utils.energy_hover, time_hovering)))

        # update total energy used.
        self.energy_used += energy_usage

        # ---
        # calculate reward = sum of all scores

        total_score = sum(
            gym_utils.get_scores(
                gym_utils.conv_locs(new_locs),
                user_locs,
                self.cov_range,
                p_factor=0.8
            )
        )

        reward = total_score / self.n_users

        # update state
        self.state['uav_locs'] = np.array(new_locs, dtype=np.float32)

        # end episode if UAV runs out of battery
        if any(self.energy_used > self.uav_bat_cap):
            done = True

        info = {}

        return self.state, reward, done, info

    def render(self, mode="human"):
        uav_locs = [self.state['uav_locs'][::2], self.state['uav_locs'][1::2]]
        user_locs = [self.state['user_locs'][::2], self.state['user_locs'][1::2]]
        # Render the environment to the screen
        plt.xlim([0, 10])
        plt.ylim([0, 10])

        plt.scatter(uav_locs[0], uav_locs[1], s=6000, color='blue', alpha=0.3)
        plt.scatter(uav_locs[0], uav_locs[1], color='red')

        plt.scatter(user_locs[0], user_locs[1], color='grey', s=2)
        plt.xlabel("X cordinate")
        plt.ylabel("Y cordinate")
        plt.pause(0.001)
        plt.show()
        plt.clf()
        # return list(map(list, zip(*gym_utils.conv_uav_locs(self.state['uav_locs'].tolist()))))


if __name__ == '__main__':
    env = UAVCoverage(n_uavs=1)

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
