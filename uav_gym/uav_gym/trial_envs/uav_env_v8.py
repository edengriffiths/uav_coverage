import gym
from gym.utils import seeding
import numpy as np
from sklearn.datasets import make_blobs
import uav_gym.utils as gym_utils
import matplotlib.pyplot as plt


class UAVCoverage(gym.Env):
    n_users = 6
    max_demand = 10
    # 1 unit = 100 m

    # ----
    # UAV SETTINGS
    # radius of the coverage range on the ground in units
    cov_range = 1  # units

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

        self.n_uavs = n_uavs

        self.action_space = gym.spaces.Discrete(3)

        # self.observation_space = gym.spaces.Discrete(int(self.sim_size / self.dist + 1))
        self.observation_space = gym.spaces.Dict({
            'uav_locs': gym.spaces.Discrete(int(self.sim_size / self.dist + 1)),  # TODO: Check that using int is okay
            'user_locs': gym.spaces.MultiDiscrete(np.array([self.sim_size + 1] * self.n_users)),
            'user_demand': gym.spaces.MultiDiscrete(np.array([self.max_demand] * self.n_users)),
        })

        self.energy_used = None
        self.state = None
        self.timestep = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=0):
        # user_demand = np.array([self.max_demand / self.n_users * i for i in range(self.n_users)])
        # self.user_demand = self.np_random.randint(0, self.max_demand, size=self.n_users)

        self.state = {
            'uav_locs': 0,
            'user_locs': np.array([1, 2, 3, 8, 9, 10]),
            'user_demand': self.np_random.randint(0, self.max_demand, size=self.n_users)
        }

        self.timestep = 0

        self.energy_used = 0

        return self.state

    def step(self, action: np.ndarray):
        done = False
        self.timestep += 1

        # unpack state
        uav_locs = self.state['uav_locs']
        user_locs = self.state['user_locs']
        user_demand = self.state['user_demand']

        # ---
        # move UAVs

        # maybe_locs is a list of the new positions regardless of whether they are out of bounds
        maybe_pos = uav_locs + self.dist * get_move(action)

        new_pos = maybe_pos if 0 <= maybe_pos <= self.sim_size else uav_locs

        # ---
        # calculate reward = sum of all scores

        d = np.array(abs(user_locs - new_pos))

        # TODO: This can stop before all user demand gone. Case: d = [5, 4, 0], ud = [0, 0, 1]
        if sum(d * user_demand) <= 0:
            reward = 10 / self.timestep
            done = True
        else:
            reward = 10 / sum(d * user_demand) / self.timestep

        demand = [None] * len(user_locs)

        for i in range(len(user_locs)):
            dist = abs(user_locs[i] - new_pos)

            if dist <= self.cov_range:
                demand[i] = user_demand[i] - 1/self.dist
            else:
                demand[i] = user_demand[i]# - 1 / dist #FIXME

            if demand[i] <= 0:
                demand[i] = 0


        # update state
        self.state = {
            'uav_locs': new_pos,
            'user_locs': user_locs,
            'user_demand': np.array(demand, dtype=np.float32)
        }

        info = {}

        return self.state, reward, done, info

    def render(self, mode="human"):
        uav_locs = [self.state['uav_locs'], 5]
        user_locs = self.state['user_locs']
        # Render the environment to the screen
        plt.xlim([0, 10])
        plt.ylim([0, 10])

        plt.scatter(uav_locs[0], uav_locs[1], s=6000, color='blue', alpha=0.3)
        plt.scatter(uav_locs[0], uav_locs[1], color='red')

        plt.scatter(user_locs, [5] * len(user_locs), color='grey', s=2)
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
        return 0
    elif action == 1:
        return 1
    elif action == 2:
        return -1


if __name__ == '__main__':
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env

    env = UAVCoverage()
    env.reset()
    # check_env(env)
    #
    # obs = env.reset()
    # n_steps = 100
    # for _ in range(n_steps):
    #     # Random action
    #     action = env.action_space.sample()
    #     obs, reward, done, info = env.step(action)
    #     print(reward)
    #     if done:
    #         obs = env.reset()
    #     env.render()

    model = PPO('MultiInputPolicy', env, verbose=1)
    model.learn(total_timesteps=10**5)
    model.save('./t')

    # model = PPO.load('./t.zip')
    obs = env.reset()
    locs = []
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        print(obs['user_demand'])
        env.render()
