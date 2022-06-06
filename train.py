from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import numpy as np
import matplotlib.pyplot as plt
import os
import gym
import uav_gym


env_v = 'v5'
models_dir = f"models/{env_v}/PPO"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)
#
env = gym.make('uav-v0')
env.reset()

model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=logdir)


TIMESTEPS = 10000
iters = 0
i = 0
while True:
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")
    i += 1




