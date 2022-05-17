import gym
from gym.utils import seeding
import numpy as np
from sklearn.datasets import make_blobs
import utils as gym_utils
import matplotlib.pyplot as plt


class UAVCoverage(gym.Env):
    n_users = 1000

    # 1 unit = 100 m

    # ----
    # UAV SETTINGS
    # radius of the coverage range on the ground in units
    cov_range = 2  # units

    # uav velocity (assumes constant velocity)
    uav_vel = 1 / 9  # units / s

    # battery capacity
    uav_bat_cap = 180

    # ----
    # Simulation settings
    time_per_epoch = 1  # seconds

    sim_size = 10

    def __init__(self, n_uavs: int = 5):
        self.seed()

        self.user_centres = []
        self.user_locs = []
        # TODO: Random number of UAVs?
        self.n_uavs = n_uavs

        self.action_space = gym.spaces.MultiDiscrete(np.array([4] * self.n_uavs, dtype=np.int32))

        # uav_locs is the locations of each UAV in the form [x1, y1, x2, y2]
        self.observation_space = gym.spaces.Dict({
            'uav_locs': gym.spaces.MultiDiscrete(np.array([11] * 2 * self.n_uavs, dtype=np.int32)),
            'cov_score': gym.spaces.Box(low=0, high=1, shape=(self.n_users,), dtype=np.float32),
            'energy_used': gym.spaces.Box(low=0, high=self.uav_bat_cap, shape=(self.n_uavs,), dtype=np.float32)
        })

        self.state = None
        self.timestep = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=0):
        self.state = {
            'uav_locs': np.array([0] * 2 * self.n_uavs),
            'cov_score': np.array([0] * self.n_users, dtype=np.float32),
            'energy_used': np.array([0] * self.n_uavs, dtype=np.float32),
        }

        # self.user_locs = np.array(self.np_random.uniform(0, self.sim_size, size=[self.n_users, 2]))
        self.user_centres = self.np_random.randint(0, self.sim_size, size=[1, 2])
        # # TODO: Random cluster_stds?
        # self.user_locs, _ = make_blobs(n_samples=self.n_users, centers=self.user_centres, cluster_std=[0.5, 0.7],
        #                                random_state=1)
        self.user_locs, _ = make_blobs(n_samples=self.n_users, centers=self.user_centres, cluster_std=[2.5],
                                       random_state=1)

        self.timestep = 0

        return self.state

    def step(self, action: np.ndarray):
        done = False
        self.timestep += 1

        # convert the list of uav locs of the form [x1, y1, x2, y2] to [[x1, y1], [x2, y2]]
        uav_locs = gym_utils.conv_uav_locs(self.state['uav_locs'])

        distances = np.array([0.1] * len(action))
        # ---
        # move UAVs
        moves = np.array(list(map(gym_utils.get_move, action, distances)))

        # maybe_locs is a list of the new positions regardless of whether they are out of bounds
        maybe_locs = uav_locs + moves

        new_locs = np.array([maybe_locs[i] if gym_utils.inbounds(maybe_locs[i]) else uav_locs[i] for i in range(len(moves))],
                            dtype=np.float32).flatten()

        # ---
        # calculate energy usage
        # energy used moving + energy used hovering
        time_moving = distances / self.uav_vel
        time_hovering = self.time_per_epoch - time_moving
        energy_usage = np.array(list(map(gym_utils.energy_move, time_moving))) + np.array(list(map(gym_utils.energy_hover, time_hovering)))

        # ---
        # calculate the coverage score
        cov_state = gym_utils.get_coverage_state(gym_utils.conv_uav_locs(new_locs), self.user_locs, self.cov_range)
        prev_cov_score = self.state['cov_score']
        new_cov_score = (prev_cov_score * (self.timestep - 1) + cov_state) / self.timestep

        fair_ind = 0 if sum(new_cov_score) == 0 else sum(new_cov_score)**2 / (self.n_users * sum(new_cov_score**2))
        # FIXME: reward function and clusters
        # ---
        # calculate distance to user cluster centres
        # dist_to_clusters = distance.cdist(gym_utils.conv_uav_locs(new_locs), self.user_centres, 'euclidean')

        # ---
        # calculate reward = change in coverage score / change in total energy consumption
        # reward = -0.005 * sum(dist_to_clusters.min(axis=1)) + sum(new_cov_score - prev_cov_score) / sum(energy_usage)
        reward = fair_ind * sum(new_cov_score - prev_cov_score) / sum(energy_usage)

        # update state
        self.state = {
            'uav_locs': np.array(new_locs, dtype=np.float32),
            'cov_score': np.array(new_cov_score, dtype=np.float32),
            'energy_used': (self.state['energy_used'] + energy_usage).astype(np.float32)
        }

        # end episode if UAV runs out of battery
        if any(self.state['energy_used'] > self.uav_bat_cap):
            done = True

        info = {}

        return self.state, reward, done, info

    def render(self, mode="human"):
        positions = self.state['uav_locs']
        # Render the environment to the screen
        plt.xlim([0, 10])
        plt.ylim([0, 10])

        plt.scatter(positions[0], positions[1], s=6000, color='blue', alpha=0.3)
        plt.scatter(positions[0], positions[1], color='red')

        plt.scatter(*zip(*self.user_locs), color='grey', s=2)
        plt.xlabel("X cordinate")
        plt.ylabel("Y cordinate")
        plt.pause(0.001)
        plt.show()
        plt.clf()

        # print(self.user_locs.tolist())
        # print(gym_utils.conv_uav_locs(self.state['uav_locs']).tolist())
        # return list(map(list, zip(*gym_utils.conv_uav_locs(self.state['uav_locs'].tolist()))))


if __name__ == '__main__':
    env = UAVCoverage()

    from stable_baselines3 import PPO

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
    model.learn(total_timesteps=1000)

    obs = env.reset()
    locs = []
    for _ in range(200):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()

