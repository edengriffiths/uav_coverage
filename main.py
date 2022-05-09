from uav_gym.envs import uav_env_v6
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt
import animate


env = uav_env_v6.SimpleUAVEnv(5)
model = PPO('MultiInputPolicy', env, verbose=1)
model.learn(total_timesteps=10000)
model.save("./v6")
#
#
# model = PPO.load('./uav_gym/uav_gym/envs/v6')

obs = env.reset()


uav_locs_all = []
rewards = []
cov_score = None
energy_used = None
time = 0

for _ in range(200):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)

    state = env.render()
    uav_locs = state['uav_locs'].tolist()
    cov_score = state['cov_score']
    energy_used = state['energy_used']
    time = env.timestep * env.time_per_epoch

    uav_locs_all.append([uav_locs[::2], uav_locs[1::2]])
    rewards.append(reward)


avg_cov_score = np.mean(cov_score)
n_users_covered = len(cov_score > 0)
avg_energy_use = np.mean(energy_used)
avg_reward = np.mean(rewards)

print(avg_cov_score)
print(n_users_covered)
print(avg_energy_use)
print(avg_reward)
print(uav_locs_all)

# user_locs = np.array(list(zip(*env.user_locs)))
#
# a = animate.AnimatedScatter(user_locs, locs)
# plt.show()