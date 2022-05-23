import gym
import uav_gym
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt

model_dir = models_dir = "rl-baselines3-zoo/logs/ppo/uav-v0_10/"

env = gym.make('uav-v0')
env.reset()

model = PPO.load(f"{models_dir}/best_model.zip", env=env)

locs = []

obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    l = obs.tolist()
    l = [l[x:x + 2] for x in range(0, len(l), 2)]
    print(l)
    locs.append(l)


import animate

user_locs = np.array(list(zip(*env.user_locs)))
list_uav_locs = np.array(locs)

a = animate.AnimatedScatter(user_locs, list_uav_locs)
plt.show()
