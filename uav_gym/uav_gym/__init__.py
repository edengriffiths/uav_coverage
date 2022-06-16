from gym.envs.registration import register


register(
    id='uav-v0',
    entry_point='uav_gym.envs:UAVCoverage',
    max_episode_steps=1800  # this is the maximum flight time of all the UAVs
)