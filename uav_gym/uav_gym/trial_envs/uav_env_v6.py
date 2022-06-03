import copy
import gym
import numpy as np
import networkx as nx
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from functools import reduce

from typing import List


def conv_uav_locs(uav_locs):
    return np.array([uav_locs[x:x + 2] for x in range(0, len(uav_locs), 2)])


def get_move(action):
    dist = 0.2
    if action == 0:
        return [0, 0]
    elif action == 1:
        return [0, dist]
    elif action == 2:
        return [dist, 0]
    elif action == 3:
        return [0, -dist]
    elif action == 4:
        return [-dist, 0]


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

    nodes = dict(enumerate(all_locs))
    g = nx.Graph()

    for n, pos in nodes.items():
        g.add_node(n, pos=pos)

    edges = nx.generators.geometric.geometric_edges(g, radius=comm_range, p=2)
    g.add_edges_from(edges)

    return g


def get_disconnected_count(g):
    """
    :param G: networkx graph
    :return: the number of nodes that are unreachable from home.
    """
    home = 0

    # the total number of nodes in g (-1 for home) - the number of nodes reachable from home.
    return len(g) - len(nx.descendants(g, home)) - 1

    # # Get the subgraph of all the nodes connected to home
    # s = g.subgraph(nx.shortest_path(g, home))
    #
    # # Return the number of nodes not connected to home
    # return g.number_of_nodes() - s.number_of_nodes()


def get_coverage_state_from_uav(uav_loc: np.array, user_locs: np.array, cov_range: float):
    dist_to_users = distance.cdist([uav_loc], user_locs, 'euclidean').flatten()
    return dist_to_users <= cov_range


def get_coverage_state(uav_locs: np.array, user_locs: np.array, cov_range: float):
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
    # 1 unit = 100 m

    # ----
    # UAV SETTINGS

    # uav velocity (assumes constant velocity)
    uav_vel = 1 / 9  # units / s

    # battery capacity
    uav_bat_cap = 180

    def energy_move_dist(self, d): return energy_move(d / self.uav_vel)

    # ----
    # Simulation settings
    time_per_epoch = 10  # seconds

    def __init__(self, n_uavs: int = 5, n_users: int = 100, cov_range: float = 2.0, comm_range: float = 5.0):
        """
        :param n_uavs: number of uavs
        :param cov_range: radius of the coverage range on the ground (in units)
        :param comm_range: distance UAVs can be from each other and still communicate (in units)
        :param n_users: number of users
        """
        self.n_uavs = n_uavs
        self.cov_range = cov_range
        self.comm_range = comm_range
        self.n_users = n_users

        self.action_space = gym.spaces.MultiDiscrete(np.array([5] * self.n_uavs, dtype=np.int32))

        # uav_locs is the locations of each UAV in the form [x1, y1, x2, y2]
        self.observation_space = gym.spaces.Dict({
            'uav_locs': gym.spaces.MultiDiscrete(np.array([11] * 2 * self.n_uavs, dtype=np.int32)),
            'cov_score': gym.spaces.Box(low=0, high=1, shape=(self.n_users,), dtype=np.float32),
            'energy_used': gym.spaces.Box(low=0, high=self.uav_bat_cap, shape=(self.n_uavs,), dtype=np.float32)
        })

        self.user_locs, _ = make_blobs(n_samples=self.n_users, centers=[[1, 3], [3, 1]], cluster_std=[0.5, 0.7],
                                       random_state=1)

        self.home_loc = np.array([0, 0])
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

        disconnected_count = get_disconnected_count(make_graph_from_locs(uav_locs, self.home_loc, self.comm_range))

        # ---
        # calculate energy usage
        # energy used moving + energy used hovering
        dist = np.array([1] * self.n_uavs)  # each movement is 1 block so dist = 1
        time_moving = dist / self.uav_vel
        time_hovering = self.time_per_epoch - time_moving
        energy_usage = np.array(list(map(energy_move, time_moving))) + np.array(list(map(energy_hover, time_hovering)))

        # ---
        # calculate the coverage score
        cov_state = get_coverage_state(conv_uav_locs(new_locs), self.user_locs, self.cov_range)
        prev_cov_score = self.state['cov_score']
        new_cov_score = (prev_cov_score * (self.timestep - 1) + cov_state) / self.timestep

        # ---
        # calculate reward = change in coverage score / change in total energy consumption
        reward = sum(new_cov_score - prev_cov_score) / sum(energy_usage)
        penalty = 100 * reward
        penalised_reward = reward - penalty * (outside_count + disconnected_count)

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

        return self.state, penalised_reward, done, info

    def reset(self):
        self.state = {
            'uav_locs': np.array([0] * 2 * self.n_uavs),
            'cov_score': np.array([0] * self.n_users, dtype=np.float32),
            'energy_used': np.array([0] * self.n_uavs, dtype=np.float32),
        }

        self.timestep = 0

        return self.state

    def render(self, mode="human"):
        print(self.state['uav_locs'])
        return self.state


if __name__ == '__main__':
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3 import PPO

    env = SimpleUAVEnv(5)

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
    positions = []
    for _ in range(100):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        locs = obs['uav_locs']
        positions.append(np.array([locs[::2], locs[1::2]]).tolist())

    print(env.user_locs.tolist())
    print(positions)
