from gym.envs.registration import register

register(id='uav_env-v0', entry_point='gym_basic.envs:UAVENV_v0',)
register(id='uav_env-v1', entry_point='gym_basic.envs:UAVENV_v1',)
register(id='uav_env-v2', entry_point='gym_basic.envs:UAVENV_v2',)
register(id='uav_env-v3', entry_point='gym_basic.envs:UAVENV_v3',)