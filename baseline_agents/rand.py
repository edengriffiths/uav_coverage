import gym
import uav_gym
import uav_gym.utils as gym_utils


"""Random: At each timestep, a random action is selected for each UAV i from the action space. If these actions 
result in any of the UAVs going out of bounds or disconnecting, then all the actions are abandoned and the UAVs stay 
where they were. """
env = gym.make('uav-v0')

obs = env.reset()
# print(env.denormalize_obs(obs)['uav_locs'])
n_steps = 100
done = False

while not done:
    # Random action
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    # print(reward)
    # print(env.denormalize_obs(obs))

print(env.denormalize_obs(obs)['cov_scores'])