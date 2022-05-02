import copy
import gym
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt


import copy
import gym
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt


def get_move(action):
    if action == 0:
        return [0, 1]
    elif action == 1:
        return [1, 0]
    elif action == 2:
        return [0, -1]
    elif action == 3:
        return [-1, 0]


def inbounds(loc, x_l_bound=0, x_u_bound=10, y_l_bound=0, y_u_bound=10):
    return x_l_bound <= loc[0] <= x_u_bound and y_l_bound <= loc[1] <= y_u_bound


def get_covered_users(uav_loc: np.array, user_locs: np.array, cov_range: int):
    dist_to_users = distance.cdist([uav_loc], user_locs, 'euclidean').flatten()
    return (dist_to_users <= cov_range).astype(int)


class SimpleUAVEnv(gym.Env):
    n_users = 5

    # 1 unit = 100 m

    # ----
    # UAV SETTINGS
    # radius of the coverage range on the ground in units
    cov_range = 2  # units


    # ----
    # Simulation settings
    time_per_epoch = 10  # seconds

    def __init__(self):
        self.action_space = gym.spaces.Discrete(4)

        # observation space is the x, y coordinate of the UAV.
        self.observation_space = gym.spaces.MultiDiscrete(np.array([11, 11]))

        self.user_locs = np.array([[i + 1, i + 1] for i in range(5)])

        self.state = None

        self.reset()

    def step(self, action):
        done = False

        uav_loc = self.state

        # move UAV
        move = np.array(get_move(action))
        if inbounds(uav_loc + move):
            uav_loc = uav_loc + move

        cov_state = get_covered_users(uav_loc, self.user_locs, self.cov_range)
        reward = float(sum(cov_state))

        # update state
        self.state = uav_loc

        info = {}

        return self.state, reward, done, info

    def reset(self):
        self.state = np.array([0, 0])

        return self.state

    def render(self, mode="human"):
        position = self.state
        # Render the environment to the screen
        plt.xlim([0, 10])
        plt.ylim([0, 10])

        plt.scatter(position[0], position[1], s=6000, color='blue', alpha=0.3)
        plt.scatter(position[0], position[1], color='red')

        plt.scatter(*zip(*self.user_locs), color='grey', s=2)
        plt.xlabel("X cordinate")
        plt.ylabel("Y cordinate")
        plt.pause(0.001)
        plt.show()


if __name__ == '__main__':
    env = SimpleUAVEnv()

    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env

    check_env(env)

    obs = env.reset()
    n_steps = 10
    for _ in range(n_steps):
        # Random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
        env.render()

    # model = PPO('MultiInputPolicy', env, verbose=1)
    # model.learn(total_timesteps=10000)
    #
    # obs = env.reset()
    # for _ in range(50):
    #     action, _states = model.predict(obs)
    #     obs, rewards, done, info = env.step(action)
    #     env.render()