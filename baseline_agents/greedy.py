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
            obs, reward, done, info = new_env.step(action)

            if reward > best_reward:
                best_action = action
                best_reward = reward

    return best_action


env = gym.make('uav-v0', n_uavs=4)
env.seed(0)
obs = env.reset()
# print(env.denormalize_obs(obs)['uav_locs'])

done = False

# while not done:
#     action = get_fake_greedy_action(env)
#     obs, reward, done, info = env.step(action)
#     print(action)
#     env.render()

actions = [
    [1, 2, 0, 0],
    [2, 2, 0, 0],
    [2, 2, 0, 0],
    [2, 2, 1, 0],
    [2, 2, 2, 0],
    [2, 2, 2, 0],
    [0, 2, 0, 0],
    [0, 2, 0, 0],
    [0, 1, 0, 0],
    [0, 2, 0, 0],
    [0, 2, 0, 0],
    [0, 0, 0, 0],
    # [0, 1, 0, 0],
    # [0, 1, 0, 0],
    # [0, 1, 0, 0],
    # [0, 1, 0, 0],
]

for action in actions:
    obs, reward, done, info = env.step(action)
    # print(reward)
    # print(obs['cov_scores'])
    env.render()
