from uav_gym.envs.uav_env import UAVCoverage
from uav_gym import utils as gym_utils


class UAVCoverage16(UAVCoverage):

    def reward_2(self, prev_cov_score, maybe_uav_locs):
        """
        Include constant penalty for disconnecting or going out of bounds
        :param prev_cov_score: the coverage score from the previous timestep.
        :param maybe_uav_locs: positions the UAVs tried to move to.
        """
        uav_locs = self.denormalize_obs(self.state)['uav_locs']

        reward = self.reward_1(prev_cov_score)
        graph = gym_utils.make_graph_from_locs(uav_locs.tolist(), self.home_loc, self.comm_range)
        dconnect_count = gym_utils.get_disconnected_count(graph)

        p_dconnect = self.sg.V['P_DISCONNECT'] * dconnect_count

        outside_count = self.n_uavs - sum(
            [gym_utils.inbounds(loc, x_ubound=self.sim_size, y_ubound=self.sim_size)
             for loc in maybe_uav_locs
             ]
        )

        p_outside = self.sg.V['P_OUT_BOUNDS'] * outside_count

        self.disconnect_count += dconnect_count

        return reward - 100 * p_dconnect * reward - 100 * p_outside * reward


if __name__ == '__main__':
    env = UAVCoverage(demonstration=True)

    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env

    check_env(env)

    obs = env.reset()
    print(env.denormalize_obs(obs)['user_locs'].tolist())
    n_steps = 100
    for _ in range(n_steps):
        # Random action
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(reward)
        # print(env.denormalize_obs(obs))
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
    #     print(rewards)
    #     env.render()
