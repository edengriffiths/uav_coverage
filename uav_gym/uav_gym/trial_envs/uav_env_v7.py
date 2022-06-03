import copy
import math
import gym
import numpy as np
import networkx as nx
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from functools import reduce


def conv_uav_locs(uav_locs):
    return np.array([uav_locs[x:x + 2] for x in range(0, len(uav_locs), 2)])


def move_uav(uav_loc, distance, direction):
    """
    :param uav_loc: [x, y] constrained by the edge of the area
    :param distance: floating point number [0, d_max]
    :param direction: floating point number [0, 2π)
    :return:
    """
    x = uav_loc[0] + distance * math.cos(direction)
    y = uav_loc[1] + distance * math.sin(direction)

    return [x, y]


def inbounds(loc, x_l_bound=0, x_u_bound=10, y_l_bound=0, y_u_bound=10):
    return x_l_bound <= loc[0] <= x_u_bound and y_l_bound <= loc[1] <= y_u_bound


def make_graph_from_locs(uav_locs, home_loc, comm_range):
    """
    :param uav_locs: np.ndarray of uav location [[x1, y1], [x2, y2],..., [xn, yn]
    :param home_loc: location of the home base [x, y]
    :param comm_range: an integer that represents the communication range of a UAV.
    :return: a networkx graph of the UAVs and home with edges between UAVs that are within the communication range.
    """
    # TODO: Can the home base have a larger range? Can it get data from UAV?
    all_locs = [home_loc] + uav_locs

    # convert to matrix of distances between all UAVs
    # convert to boolean matrix where True if distance < comm_range
    M = distance.cdist(all_locs, all_locs) < comm_range

    # make graph and remove self loops
    G = nx.from_numpy_matrix(M)
    G.remove_edges_from(nx.selfloop_edges(G))

    return G


def get_disconnected_count(G):
    """
    :param G: networkx graph
    :return: the number of nodes that are unreachable from home.
    """
    home = 0

    # Get the subgraph of all the nodes connected to home
    s = G.subgraph(nx.shortest_path(G, home))

    # Return the number of nodes not connected to home
    return G.number_of_nodes() - s.number_of_nodes()


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
    # ----
    # SIMULATION SETTINGS
    time_per_epoch = 10  # seconds
    env_width = 10
    env_length = 10

    n_users = 100
    n_uavs = 5

    # 1 unit = 100 m

    # ----
    # UAV SETTINGS
    # radius of the coverage range on the ground in units
    cov_range = 2  # units

    # radius of the communication range (distance UAVs can be from each other and still communicate) in units
    comm_range = 5  # units

    # uav velocity (assumes constant velocity)
    uav_vel = 1 / 9  # units / s

    max_dist = uav_vel * time_per_epoch

    # battery capacity
    uav_bat_cap = 63

    def energy_move_dist(self, d): return energy_move(d / self.uav_vel)


    def __init__(self):
        # TODO: Action space has to be a vector so can't have shape (2, ...)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2, self.n_uavs), dtype=np.float32)

        # TODO: Normalise observation spaces
        # TODO: Change UAV_locs to be a 1D vector
        # uav_locs is the locations of each UAV in the form [x1, y1, x2, y2]
        self.observation_space = gym.spaces.Dict({
            'uav_locs': gym.spaces.Box(low=0, high=1, shape=(2, self.n_uavs), dtype=np.float32),
            'cov_score': gym.spaces.Box(low=0, high=1, shape=(self.n_users,), dtype=np.float32),
            'energy_used': gym.spaces.Box(low=0, high=self.uav_bat_cap, shape=(self.n_uavs,), dtype=np.float32)
        })

        self.user_locs, _ = make_blobs(n_samples=self.n_users, centers=[[8, 3], [3, 8]], cluster_std=[0.5, 0.7],
                                       random_state=1)

        self.home_loc = np.array([0, 0])
        self.state = None
        self.timestep = 0

        self.reset()

    def step(self, action: np.ndarray):
        done = False
        self.timestep += 1

        scaled_uav_locs = self.state['uav_locs'] * [[self.env_width], [self.env_length]]
        scaled_distances = action[0] * self.max_dist
        scaled_directions = action[1] * math.pi

        # convert the list of uav locs to the correct shape
        # [[x1, x2, ..., xn], [y1, y2, ..., yn]] to [[x1, y1], [x2, y2], ... [xn, yn]]
        uav_locs = np.array(list(zip(*scaled_uav_locs)))

        # ---
        # move UAVs

        # maybe_locs is a list of the new positions regardless of whether they are out of bounds
        maybe_locs = np.array(list(map(move_uav, uav_locs, scaled_distances, scaled_directions)))
        inbounds_arr = np.array(list(map(inbounds, maybe_locs)))
        outside_count = self.n_uavs - sum(inbounds_arr)

        new_locs = np.array(list(zip(*[maybe_locs[i] if inbounds_arr[i] else uav_locs[i] for i in range(self.n_uavs)])),
                            dtype=np.float32)

        disconnected_count = get_disconnected_count(make_graph_from_locs(uav_locs, self.home_loc, self.comm_range))

        norm_new_locs = new_locs / [[self.env_width], [self.env_length]]

        # TODO: Recalculate energy usage given continuous action space
        # ---
        # calculate energy usage
        # distance if successfully moved, otherwise 0
        dist = scaled_distances * inbounds_arr
        time_moving = dist / self.uav_vel
        time_hovering = self.time_per_epoch - time_moving
        # energy used moving + energy used hovering
        energy_usage = np.array(list(map(energy_move, time_moving))) + np.array(list(map(energy_hover, time_hovering)))

        # ---
        # calculate the coverage score
        cov_state = get_coverage_state(np.array(list(zip(*new_locs))), self.user_locs, self.cov_range)
        prev_cov_score = self.state['cov_score']
        new_cov_score = (prev_cov_score * (self.timestep - 1) + cov_state) / self.timestep

        # ---
        # calculate reward = change in coverage score / change in total energy consumption
        reward = sum(new_cov_score - prev_cov_score) / sum(energy_usage)
        penalty = 100 * reward
        penalised_reward = reward - penalty * (outside_count + disconnected_count)

        # TODO: What to do when UAVs run out of battery?
        # end episode if UAV runs out of battery
        if any(energy_usage > self.uav_bat_cap):
            done = True

        # update state
        self.state = {
            'uav_locs': np.array(norm_new_locs, dtype=np.float32),
            'cov_score': np.array(new_cov_score, dtype=np.float32),
            'energy_used': (self.state['energy_used'] + energy_usage).astype(np.float32)
        }

        info = {}

        print(self.state['uav_locs'])

        print(self.observation_space['uav_locs'])

        return self.state, penalised_reward, done, info

    def reset(self):
        self.state = {
            'uav_locs': np.array([[0] * self.n_uavs, [0] * self.n_uavs], dtype=np.float32),
            'cov_score': np.array([0] * self.n_users, dtype=np.float32),
            'energy_used': np.array([0] * self.n_uavs, dtype=np.float32),
        }

        return self.state

    def render(self, mode="human"):
        # positions = self.state['uav_locs']
        # # Render the environment to the screen
        # plt.xlim([0, 10])
        # plt.ylim([0, 10])
        #
        # plt.scatter(positions[0], positions[1], s=6000, color='blue', alpha=0.3)
        # plt.scatter(positions[0], positions[1], color='red')
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

    # while True:
    #     check_env(env)

    # obs = env.reset()
    # n_steps = 5
    # for _ in range(n_steps):
    #     # Random action
    #     action = env.action_space.sample()
    #     print(action)
    #     print(obs)
    #     obs, reward, done, info = env.step(action)
    #     if done:
    #         obs = env.reset()
    #     env.render()

    model = PPO('MultiInputPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)
    # model.save("./v6")
    # model = PPO.load('./v6')

    obs = env.reset()
    locs = []
    for _ in range(200):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        print(rewards)
        locs.append(env.render())

    print(env.user_locs)
    print(locs)
