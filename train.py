from uav_gym.uav_gym.envs.uav_env_v5 import UAVCoverage as Env_v5
from uav_gym.uav_gym.envs.uav_env_v6 import SimpleUAVEnv as Env_v6
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
import numpy as np
import matplotlib.pyplot as plt
import os
import gym


# def make_env(rank: int = 0, seed: int = 0):
#     def _init():
#         env = Env_v5
#         return env
#
#     return _init()
#
#
# env = make_vec_env(Env_v5, n_envs=4, seed=0)
#
# # env = SubprocVecEnv([make_env(i, i) for i in range(4)])
# # env = Env_v5(n_uavs=2)
#
#
# model = PPO('MultiInputPolicy', env, verbose=1)
# model.learn(total_timesteps=100000)
# model.save('./t')
#
# # model = PPO.load('./t')
#
# locs = []
#
# env = Env_v5()
#
# obs = env.reset()
#
# # done = False
# # while not done:
# for _ in range(180):
#     action, _states = model.predict(obs)
#     obs, reward, done, info = env.step(action)
#     l = obs['uav_locs'].tolist()
#     l = [l[x:x + 2] for x in range(0, len(l), 2)]
#     locs.append(l)
#
#
# print(env.user_locs.tolist())
# print(locs)
# # env.render()
# # print(env.user_locs.tolist())
# # print(locs)


models_dir = "models/PPO"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)
#
env = Env_v5()
env.reset()

model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=logdir)


# TODO: Work out if this is just learning on the same UAV loc positions or new ones.
TIMESTEPS = 10000
iters = 0
for i in range(30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")


# model = PPO.load(f"{models_dir}/190000.zip", env=env)
#
# locs = []
#
# obs = env.reset()
# done = False
# while not done:
#     action, _states = model.predict(obs)
#     obs, rewards, done, info = env.step(action)
#     l = obs['uav_locs'].tolist()
#     l = [l[x:x + 2] for x in range(0, len(l), 2)]
#     locs.append(l)
#
#
# import animate
#
# user_locs = np.array(list(zip(*env.user_locs)))
# list_uav_locs = np.array(locs)
#
# a = animate.AnimatedScatter(user_locs, list_uav_locs)
# plt.show()




