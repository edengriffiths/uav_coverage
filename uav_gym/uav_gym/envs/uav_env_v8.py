import gym
from gym.utils import seeding
import numpy as np
from sklearn.datasets import make_blobs
import uav_gym.utils as gym_utils
import matplotlib.pyplot as plt


class UAVCoverage(gym.Env):
    n_users = 10
    max_demand = 100
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

    def __init__(self, n_uavs: int = 5):
        self.seed()

        self.user_locs = None
        self.user_demand = None

        self.n_uavs = n_uavs

        self.action_space = gym.spaces.Discrete(2)

        self.observation_space = gym.spaces.Discrete(int(self.sim_size / self.dist + 1))

        self.energy_used = None
        self.state = None
        self.timestep = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=0):
        self.state = 0

        self.user_locs = np.array([0.2, 1, 1.6, 2, 6, 6.8, 7, 8, 9, 9.2])
        self.user_demand = self.np_random.randint(0, self.max_demand, size=self.n_users)

        self.timestep = 0

        self.energy_used = 0

        return self.state

    def step(self, action: np.ndarray):
        done = False
        self.timestep += 1

        # ---
        # move UAVs
        new_pos = self.state + self.dist * get_move(action)

        # maybe_locs is a list of the new positions regardless of whether they are out of bounds
        maybe_pos = self.state + self.dist * get_move(action)

        new_pos = maybe_pos if 0 <= maybe_pos <= self.sim_size else self.state

        # ---
        # calculate reward = sum of all scores

        try:
            reward = 1 / sum(self.user_demand)
        except ZeroDivisionError:
            reward = 1000

        demand = [None] * len(self.user_locs)

        for i in range(len(self.user_locs)):
            dist = abs(self.user_locs[i] - new_pos)

            if dist <= self.cov_range:
                demand[i] = self.user_demand[i] - 5/self.dist
            else:
                demand[i] = self.user_demand[i]# - 1 / dist

            if demand[i] <= 0:
                demand[i] = 0

        self.user_demand = demand

        # update state
        self.state = new_pos

        info = {}

        return self.state, reward, done, info

    def render(self, mode="human"):
        positions = [self.state, 5]
        # Render the environment to the screen
        plt.xlim([0, 10])
        plt.ylim([0, 10])

        plt.scatter(positions[0], positions[1], s=6000, color='blue', alpha=0.3)
        plt.scatter(positions[0], positions[1], color='red')

        plt.scatter(self.user_locs, [5] * len(self.user_locs), color='grey', s=2)
        plt.xlabel("X cordinate")
        plt.ylabel("Y cordinate")
        plt.pause(0.001)
        plt.show()
        plt.clf()

        # print(self.user_locs.tolist())
        # print(gym_utils.conv_uav_locs(self.state['uav_locs']).tolist())
        # return list(map(list, zip(*gym_utils.conv_uav_locs(self.state['uav_locs'].tolist()))))



def get_move(action):
    if action == 0:
        return 1
    elif action == 1:
        return -1


if __name__ == '__main__':
    env = UAVCoverage()

    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env

    # check_env(env)
    #
    # obs = env.reset()
    # n_steps = 100
    # for _ in range(n_steps):
    #     # Random action
    #     action = env.action_space.sample()
    #     obs, reward, done, info = env.step(action)
    #     print(reward)
    #     print(env.user_demand)
    #     if done:
    #         obs = env.reset()
    #     env.render()

    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10**5)

    obs = env.reset()
    locs = []
    for _ in range(200):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        print(env.user_demand)
        env.render()
