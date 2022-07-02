from gym.envs.registration import register


register(
    id='uav-v0',
    entry_point='uav_gym.envs:UAVCoverage',
    max_episode_steps=600  # each step is 3 seconds. 1800 seconds is the maximum flight time of all the UAVs
)
