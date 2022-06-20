import gym
import uav_gym
import uav_gym.utils as gym_utils
import itertools
import math
from copy import deepcopy


def get_real_greedy_action(env):
    n_uavs = len(env.action_space.nvec)
    n_actions = env.action_space.nvec[0]
    actions = list(itertools.product(range(n_actions), repeat=n_uavs))

    best_action = None
    best_reward = -math.inf

    for action in actions:
        new_env = deepcopy(env)
        obs, reward, done, info = new_env.step(action)
        if reward > best_reward:
            print(action)
            print(reward)
            best_action = action
            best_reward = reward

    return best_action


def get_fake_greedy_action(env):
    n_uavs = len(env.action_space.nvec)
    n_actions = env.action_space.nvec[0]

    # log of previous uav actions
    best_action = [0] * n_uavs

    for uav in range(n_uavs):
        best_reward = -math.inf
        for action_i in range(n_actions):
            action = best_action[:]
            action[uav] = action_i

            new_env = deepcopy(env)
            obs, reward, _, _ = new_env.step(action)

            if reward > best_reward:
                best_action = action
                best_reward = reward

    # check if the action would take out of bounds or disconnect. If it would, none move.
    new_env = deepcopy(env)
    obs, _, _, _ = new_env.step(best_action)
    uav_locs = env.denormalize_obs(obs)['uav_locs']

    graph = gym_utils.make_graph_from_locs(uav_locs.tolist(), env.home_loc, env.comm_range)
    dconnect_count = gym_utils.get_disconnected_count(graph)

    if not all(gym_utils.inbounds(uav_locs, env.sim_size, env.sim_size)) or dconnect_count > 0:
        return [0] * n_uavs

    return best_action


env = gym.make('uav-v0', n_uavs=4)
env.seed(0)
obs = env.reset()
done = False

while not done:
    action = get_fake_greedy_action(env)
    obs, reward, done, info = env.step(action)
    print(action)
    print(reward)
    env.render()

# actions = [
#     [0, 0, 0, 0],
#     [2, 1, 0, 0],
#     [2, 1, 0, 0],
#
# ]
#
# import numpy as np
# env.state['uav_locs'] = np.array([
#     [-0.2, -0.4],
#     [-0.8,  0.4],
#     [-1.,  -1.],
#     [-1.,  -1.]
# ])
#
# for action in actions:
#     obs, reward, done, info = env.step(action)
#     print(reward)
#     env.render()
