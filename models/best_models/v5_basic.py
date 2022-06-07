import gym
from gym.utils import seeding
import numpy as np
from sklearn.datasets import make_blobs
import uav_gym.utils as gym_utils
import matplotlib.pyplot as plt


class UAVCoverage(gym.Env):
    # ----
    # Simulation settings
    n_users = 15

    # metres per unit
    scale = 50  # keep sim_size divisible by scale

    time_per_epoch = 1  # seconds

    sim_size = 1000

    # ----
    # UAV SETTINGS
    # radius of the coverage range on the ground in units
    cov_range = 200  # metres

    # uav velocity (assumes constant velocity)
    uav_vel = 1 / 9  # units / s  # TODO: Change
    dist = scale  # TODO: Change to smaller distance

    # battery capacity
    uav_bat_cap = 180

    def __init__(self, n_uavs: int = 5):
        self.seed()

        self.user_centres = []
        self.user_locs = []
        # TODO: Random number of UAVs?
        self.n_uavs = n_uavs

        self.action_space = gym.spaces.MultiDiscrete(
            np.array([5] * self.n_uavs, dtype=np.int32)
        )

        # uav_locs is the locations of each UAV in the form [x1, y1, x2, y2]
        self.observation_space = gym.spaces.Dict({
            'uav_locs': gym.spaces.MultiDiscrete(
                np.array([self.sim_size // self.scale + 1] * 2 * self.n_uavs, dtype=np.int32)),
            'user_locs': gym.spaces.MultiDiscrete(
                np.array([self.sim_size // self.scale + 1] * 2 * self.n_users, dtype=np.int32)),
        })

        self.cov_scores = np.array([0] * self.n_users)
        self.pref_users = self.np_random.choice([0, 1], size=(self.n_users,), p=[4./5, 1./5])

        self.state = None
        self.timestep = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _gen_user_locs(self):
        # TODO: Random cluster_stds?
        # TODO: How to do random_state? Using sim_size doesn't really make sense.
        std = 0.05 * self.sim_size
        # centers must be within one std of the border.
        center = [self.np_random.uniform(0 + 3 * std, self.sim_size - 3 * std) for _ in range(2)]
        ul_init, _ = make_blobs(n_samples=self.n_users, centers=[center], cluster_std=[std],
                                random_state=self.np_random.randint(2**32 - 1))  # TODO: This is a lot of possible states but it's still bounded. would be better if unbounded.

        # constrain users to be with 3 stds of the centre and round locations to a whole number.
        ul_constr = np.array([gym_utils._constrain_user_loc(user_loc, center, std, self.np_random)
                              for user_loc in ul_init])

        return ul_constr

    def reset(self):
        self.state = {
            'uav_locs': np.array([0] * 2 * self.n_uavs),
            'user_locs': gym_utils.scale(self._gen_user_locs().flatten(), s=self.scale, d='down')
        }
        self.timestep = 0

        return self.state

    def step(self, action: np.ndarray):
        done = False
        self.timestep += 1

        # unpack state and convert units to metres.
        uav_locs_ = gym_utils.scale(self.state['uav_locs'], self.scale, 'up')
        user_locs_ = gym_utils.scale(self.state['user_locs'], self.scale, 'up')

        # convert the lists of locs of the form [x1, y1, x2, y2] to [[x1, y1], [x2, y2]]
        uav_locs = gym_utils.conv_locs(uav_locs_)
        user_locs = gym_utils.conv_locs(user_locs_)

        distances = np.array([self.dist] * len(action))
        # ---
        # move UAVs
        moves = np.array(list(map(gym_utils.get_move, action, distances)))

        # maybe_locs is a list of the new positions regardless of whether they are out of bounds
        maybe_locs = uav_locs + moves

        new_locs = np.array(
            [maybe_locs[i]
             if gym_utils.inbounds(maybe_locs[i], self.sim_size, self.sim_size)
             else
             uav_locs[i]
             for i in range(len(moves))],
            dtype=np.float32)

        # ---
        # calculate reward = sum of all scores

        total_score = sum(
            gym_utils.get_scores(
                new_locs,
                user_locs,
                self.cov_range,
                p_factor=0.8
            )
        )

        reward = total_score / self.n_users

        # update state
        self.state['uav_locs'] = gym_utils.scale(np.array(new_locs.flatten(), dtype=np.float32), self.scale, 'down')

        # update cov_scores
        self.cov_scores += gym_utils.get_coverage_state(new_locs, user_locs, self.cov_range)

        # TODO: Check this is the correct value and that it agrees with animation.
        # stop after 30 minutes where each timestep is 1 second.
        if self.timestep >= 1800:
            done = True

        info = {}

        return self.state, reward, done, info

    def render(self, mode="human"):
        uav_locs_ = gym_utils.scale(self.state['uav_locs'], self.scale, 'up')
        user_locs_ = gym_utils.scale(self.state['user_locs'], self.scale, 'up')

        uav_locs = [uav_locs_[::2], uav_locs_[1::2]]
        user_locs = [user_locs_[::2], user_locs_[1::2]]

        # Render the environment to the screen
        plt.xlim([0, self.sim_size])
        plt.ylim([0, self.sim_size])

        plt.scatter(uav_locs[0], uav_locs[1], s=6000, color='blue', alpha=0.3)
        plt.scatter(uav_locs[0], uav_locs[1], color='red')

        plt.scatter(user_locs[0], user_locs[1], color='grey', s=2)
        plt.xlabel("X cordinate")
        plt.ylabel("Y cordinate")
        plt.pause(0.001)
        plt.show()
        plt.clf()


if __name__ == '__main__':
    env = UAVCoverage(n_uavs=1)

    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env

    check_env(env)

    obs = env.reset()
    n_steps = 100
    for _ in range(n_steps):
        # Random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(obs)
        print(reward)
        if done:
            obs = env.reset()
        env.render()

    # model = PPO('MultiInputPolicy', env, verbose=1)
    # model.learn(total_timesteps=10**5)
    #
    # obs = env.reset()
    # env.seed(0)
    # locs = []
    # for _ in range(200):
    #     action, _states = model.predict(obs)
    #     obs, rewards, done, info = env.step(action)
    #     env.render()
