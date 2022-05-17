from gym.envs.registration import register


register(
    id='uav-v0',
    entry_point='uav_gym.envs:UAVCoverage',
)