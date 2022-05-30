import gym
import uav_gym
from uav_gym import utils as gym_utils
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt

# models_dir = "rl-baselines3-zoo/logs/ppo"
#
# env = gym.make('uav-v0')
# env.reset()
#
# model = PPO.load(f"{models_dir}/uav-v0_13/best_model.zip", env=env)

env_v = 'v5'
models_dir = f"models/{env_v}/PPO"

env = gym.make('uav-v0', n_uavs=1)
env.seed(0)
env.reset()


model = PPO.load(f"{models_dir}/1110000.zip", env=env)

locs = []

obs = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    l = gym_utils.scale(obs['uav_locs'], s=env.scale, d='up').tolist()
    l = [l[::2], l[1::2]]
    locs.append(l)

c_scores = env.cov_scores / env.timestep
avg_cov_score = c_scores.mean()
f_ind = sum(c_scores) ** 2 / (env.n_users * sum(c_scores ** 2))

pref = env.pref_users * c_scores
pref = pref[pref != 0]
avg_pref_score = pref.mean()

reg = (1 - env.pref_users) * c_scores
reg = reg[reg != 0]
avg_reg_score = reg.mean()

print(c_scores)
print(avg_cov_score)
print(f_ind)
print(avg_pref_score)
print(avg_reg_score)


import animate

user_locs = gym_utils.scale(obs['user_locs'], s=env.scale, d='up')
user_locs = [user_locs[x:x + 2] for x in range(0, len(user_locs), 2)]
user_locs = np.array(list(zip(*user_locs)))
list_uav_locs = np.array(locs)


# a = animate.AnimatedScatter(user_locs, list_uav_locs, env.sim_size)
# plt.show()
