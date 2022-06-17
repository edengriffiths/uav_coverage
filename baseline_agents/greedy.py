import gym
import uav_gym
import uav_gym.utils as gym_utils
import itertools
import math
from copy import deepcopy


def get_greedy_action(env):
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


env = gym.make('uav-v0', n_uavs=4)

obs = env.reset()
# print(env.denormalize_obs(obs)['uav_locs'])

done = False

while not done:
    action = get_greedy_action(env)
    obs, reward, done, info = env.step(action)
    print(action)
    env.render()

print(env.denormalize_obs(obs)['cov_scores'])