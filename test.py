from uav_gym.uav_gym.envs.uav_env_v5 import UAVCoverage as Env_v5
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt

model_dir = models_dir = "models/PPO"

env = Env_v5()
env.reset()

model = PPO.load(f"{models_dir}/70000.zip", env=env)

locs = []

obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    l = obs['uav_locs'].tolist()
    l = [l[x:x + 2] for x in range(0, len(l), 2)]
    locs.append(l)


import animate

user_locs = np.array(list(zip(*env.user_locs)))
list_uav_locs = np.array(locs)

a = animate.AnimatedScatter(user_locs, list_uav_locs)
plt.show()
