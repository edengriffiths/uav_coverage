import copy
import gym
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from functools import reduce


def conv_uav_locs(uav_locs):
    return np.array([uav_locs[x:x + 2] for x in range(0, len(uav_locs), 2)])

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


def get_coverage_state_from_uav(uav_loc: np.array, user_locs: np.array, cov_range: int):
    dist_to_users = distance.cdist([uav_loc], user_locs, 'euclidean').flatten()
    return dist_to_users <= cov_range


def get_coverage_state(uav_locs: np.array, user_locs: np.array, cov_range: int):
    coverage_states = list(map(get_coverage_state_from_uav,
                                   uav_locs,
                                   [user_locs] * len(uav_locs),
                                   [cov_range] * len(uav_locs)))

    return reduce(lambda acc, x: acc | x, coverage_states)


# TODO: Work out better values for these
# energy used hovering
def energy_hover(t): return t / 10


# energy used flying
def energy_move(t): return t / 12.5


class SimpleUAVEnv(gym.Env):
    n_users = 100
    n_uavs = 5

    # 1 unit = 100 m

    # ----
    # UAV SETTINGS
    # radius of the coverage range on the ground in units
    cov_range = 2  # units

    # uav velocity (assumes constant velocity)
    uav_vel = 1 / 9  # units / s

    # battery capacity
    uav_bat_cap = 63

    def energy_move_dist(self, d): return energy_move(d / self.uav_vel)

    # ----
    # Simulation settings
    time_per_epoch = 10  # seconds

    def __init__(self):
        self.action_space = gym.spaces.MultiDiscrete(np.array([4] * self.n_uavs, dtype=np.int32))

        # uav_locs is the locations of each UAV in the form [x1, y1, x2, y2]
        self.observation_space = gym.spaces.Dict({
            'uav_locs': gym.spaces.MultiDiscrete(np.array([11] * 2 * self.n_uavs, dtype=np.int32)),
            'cov_score': gym.spaces.Box(low=0, high=1, shape=(self.n_users,), dtype=np.float32),
            'energy_used': gym.spaces.Box(low=0, high=self.uav_bat_cap, shape=(self.n_uavs,), dtype=np.float32)
        })

        self.user_locs, _ = make_blobs(n_samples=self.n_users, centers=[[8, 3], [3, 8]], cluster_std=[0.5, 0.7],
                                       random_state=1)

        self.state = None
        self.timestep = 0

        self.reset()

    def step(self, action: np.ndarray):
        done = False
        self.timestep += 1

        # convert the list of uav locs of the form [x1, y1, x2, y2] to [[x1, y1], [x2, y2]]
        uav_locs = conv_uav_locs(self.state['uav_locs'])

        # ---
        # move UAVs
        moves = np.array(list(map(get_move, action)))

        # maybe_locs is a list of the new positions regardless of whether they are out of bounds
        maybe_locs = uav_locs + moves
        inbounds_arr = np.array(list(map(inbounds, maybe_locs)))
        outside_count = self.n_uavs - sum(inbounds_arr)

        new_locs = np.array([maybe_locs[i] if inbounds_arr[i] else uav_locs[i] for i in range(len(moves))],
                            dtype=np.float32).flatten()

        # ---
        # calculate energy usage
        # energy used moving + energy used hovering
        dist = np.array([1] * self.n_uavs)  # each movement is 1 block so dist = 1
        time_moving = dist / self.uav_vel
        time_hovering = self.time_per_epoch - time_moving
        energy_usage = np.array(list(map(energy_move, time_moving))) + np.array(list(map(energy_hover, time_hovering)))

        # ---
        # calculate the coverage score
        cov_state = get_coverage_state(uav_locs, self.user_locs, self.cov_range)
        prev_cov_score = self.state['cov_score']
        new_cov_score = (prev_cov_score * (self.timestep - 1) + cov_state) / self.timestep

        # ---
        # calculate reward = change in coverage score / change in total energy consumption
        reward = sum(new_cov_score - prev_cov_score) / sum(energy_usage)

        # end episode if UAV runs out of battery
        if any(energy_usage > self.uav_bat_cap):
            done = True

        # update state
        self.state = {
            'uav_locs': np.array(new_locs, dtype=np.float32),
            'cov_score': np.array(new_cov_score, dtype=np.float32),
            'energy_used': (self.state['energy_used'] + energy_usage).astype(np.float32)
        }

        info = {}

        return self.state, reward - 100 * outside_count * reward, done, info

    def reset(self):
        self.state = {
            'uav_locs': np.array([0] * 2 * self.n_uavs),
            'cov_score': np.array([0] * self.n_users, dtype=np.float32),
            'energy_used': np.array([0] * self.n_uavs, dtype=np.float32),
        }

        return self.state

    def render(self, mode="human"):
        # positions = self.state['uav_locs']
        # x = [positions[i] for i in range(0, len(positions), 2)]
        # y = [positions[i] for i in range(1, len(positions), 2)]
        # # Render the environment to the screen
        # plt.xlim([0, 10])
        # plt.ylim([0, 10])
        #
        # plt.scatter(x, y, s=6000, color='blue', alpha=0.3)
        # plt.scatter(x, y, color='red')
        #
        # plt.scatter(*zip(*self.user_locs), color='grey', s=2)
        # plt.xlabel("X cordinate")
        # plt.ylabel("Y cordinate")
        # plt.pause(0.001)
        # plt.show()

        return list(map(list, zip(*conv_uav_locs(self.state['uav_locs'].tolist()))))


if __name__ == '__main__':
    env = SimpleUAVEnv()

    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env

    # check_env(env)
    #
    # obs = env.reset()
    # n_steps = 5
    # for _ in range(n_steps):
    #     # Random action
    #     action = env.action_space.sample()
    #     obs, reward, done, info = env.step(action)
    #     if done:
    #         obs = env.reset()
    #     env.render()

    model = PPO('MultiInputPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save("./v6")

    obs = env.reset()
    locs = []
    for _ in range(200):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        print(rewards)
        locs.append(env.render())

    print(env.user_locs)
    print(locs)
