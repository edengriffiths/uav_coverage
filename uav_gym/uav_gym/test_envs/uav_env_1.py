from uav_gym.envs.uav_env import UAVCoverage
from uav_gym import utils as gym_utils

class UAVCoverage1(UAVCoverage):
    def reward_1(self, prev_cov_score):
        """
        Includes user scores, fairness, and user prioritisation
        """
        state = self.denormalize_obs(self.state)

        # unpack state
        uav_locs = state['uav_locs']
        user_locs = state['user_locs']
        pref_users = state['pref_users']
        cov_scores = state['cov_scores']

        # # calculate fairness
        # f_idx = gym_utils.fairness_idx(cov_scores)
        #
        # # calculate improvement in total coverage
        # delta_cov_score = cov_scores - prev_cov_score
        #
        # # calculate distance from each user to the closest UAV.
        # dist = gym_utils.dist_to_users(uav_locs.tolist(), user_locs.tolist())
        # dist_scores = 1 / (1 + dist)
        #
        # def scale_scores(scores):
        #     return scores + (self.pref_factor - 1) * pref_users * scores
        #
        # # both parts have a max values of n_users and a min value of 0.
        # cov_part = f_idx * sum(scale_scores(delta_cov_score)) / self.n_users
        # dist_part = sum(scale_scores(dist_scores)) / self.n_users
        #
        # return cov_part + self.sg.V['P_OUTSIDE_COV'] * dist_part

        scores = gym_utils.get_scores(
            uav_locs.tolist(),
            user_locs.tolist(),
            self.cov_range,
            p_factor=self.sg.V['P_OUTSIDE_COV']
        )

        f_idx = gym_utils.fairness_idx(cov_scores / self.timestep)

        # increase the scores of the preferred users by a factor of self.pref_factor.
        scaled_scores = scores + (self.pref_factor - 1) * pref_users * scores

        mean_score = sum(scaled_scores) / self.n_users

        return f_idx * mean_score


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
