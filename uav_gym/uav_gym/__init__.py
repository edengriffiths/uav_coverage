from gym.envs.registration import register


register(
    id='uav-v0',
    entry_point='uav_gym.envs:UAVCoverage0',
    max_episode_steps=600  # each step is 3 seconds. 1800 seconds is the maximum flight time of all the UAVs
)

register(
    id='uav-v1',
    entry_point='uav_gym.envs:UAVCoverage1',
    max_episode_steps=600  # each step is 3 seconds. 1800 seconds is the maximum flight time of all the UAVs
)

register(
    id='uav-v2',
    entry_point='uav_gym.envs:UAVCoverage2',
    max_episode_steps=600  # each step is 3 seconds. 1800 seconds is the maximum flight time of all the UAVs
)

register(
    id='uav-v3',
    entry_point='uav_gym.envs:UAVCoverage3',
    max_episode_steps=600  # each step is 3 seconds. 1800 seconds is the maximum flight time of all the UAVs
)
