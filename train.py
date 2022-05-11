from uav_gym.uav_gym.envs.uav_env_v5 import SimpleUAVEnv as Env_v5
from uav_gym.uav_gym.envs.uav_env_v6 import SimpleUAVEnv as Env_v6
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import numpy as np
import matplotlib.pyplot as plt
import os


env = Env_v5(n_uavs=2)


model = PPO('MultiInputPolicy', env, verbose=1)
model.learn(total_timesteps=100000)
# model.save('./t')

locs = []

obs = env.reset()

for _ in range(180):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    l = obs['uav_locs'].tolist()
    locs.append([l[::2], l[1::2]])

print(env.user_locs.tolist())
print(locs)