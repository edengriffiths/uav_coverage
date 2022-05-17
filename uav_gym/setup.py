from setuptools import setup

setup(
    name='uav_gym',
    version='0.0.1',
    description='Package of minimalistic UAV coverage envs',
    packages=[
        'uav_gym',
        'uav_gym.envs'
    ],
    requires=[
        'gym'
    ]
)
