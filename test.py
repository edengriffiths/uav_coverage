import gym
from typing import Tuple

import uav_gym
import animate
import uav_gym.utils as gym_utils

from stable_baselines3 import PPO
import numpy as np
from operator import add
import matplotlib.pyplot as plt
from matplotlib import animation
import json
import os

import multiprocessing as multip


class Environments:

    def __init__(self, env_id, environments_n, model):
        self.env_id = env_id
        self.environments_n = environments_n
        self.cores_count = multip.cpu_count()
        self.envs = []
        self.reset_all_environments()
        self.model = model

    def __enter__(self):
        self.pool = multip.Pool(self.cores_count)
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.pool.close()
        self.pool.join()

    def reset_all_environments(self):
        for env in self.envs:
            env.close()
        self.envs = [gym.make(self.env_id) for _ in range(self.environments_n)]

    @staticmethod
    def get_c_scores_single(env):
        obs = env.reset()
        done = False
        while not done:
            # model comes from global scope so that pickling still works
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, done, info = env.step(action)

        state = env.denormalize_obs(obs)

        return state['cov_scores'], state['pref_users']

    def get_c_scores_all(self):
        results = self.pool.map(self.get_c_scores_single, self.envs)
        return results


def get_graph_data(env, model):
    uav_locs = []
    c_scores_all = []
    c_scores_reg = []
    c_scores_pref = []
    fidx = []

    obs = env.reset()

    user_locs = env.denormalize_obs(obs)['user_locs']
    pref_users = obs['pref_users'].astype(bool)
    reg_user_locs = user_locs[~pref_users]
    pref_user_locs = user_locs[pref_users]

    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)

        c_scores = env.denormalize_obs(obs)['cov_scores']

        uav_locs.append(env.denormalize_obs(obs)['uav_locs'])
        c_scores_all.append(np.round(c_scores.mean(), 4))
        c_scores_reg.append(np.round(c_scores[~pref_users].mean(), 4))
        c_scores_pref.append(np.round(c_scores[pref_users].mean(), 4))
        fidx.append(np.round(gym_utils.fairness_idx(c_scores), 4))

    return reg_user_locs, pref_user_locs, np.array(uav_locs), c_scores_all, c_scores_reg, c_scores_pref, fidx


def mean_cov_score(c_scores: np.ndarray) -> float:
    return c_scores.mean()


def mean_pref_score(c_scores: np.ndarray, pref_ids: np.ndarray) -> float:
    """
    :param c_scores: a list of c_scores from one environment episode.
    :param pref_ids: a list of pref_ids from the same environment.
    :return: the mean score of prioritised users.
    """
    _c_scores = c_scores + 1
    pref = pref_ids * _c_scores
    return (pref[pref != 0] - 1).mean()


def mean_reg_score(c_scores: np.ndarray, pref_ids: np.ndarray) -> float:
    """
    :param c_scores: a list of c_scores from one environment episode.
    :param pref_ids: a list of pref_ids from the same environment.
    :return: the mean score of regular users.
    """
    # add 1 to coverage score and then subtract it to handle the case where the coverage score equals 0.
    _c_scores = c_scores + 1
    reg = (1 - pref_ids) * _c_scores
    return (reg[reg != 0] - 1).mean()


def get_interquartile_vals(l):
    """
    :param l: l must be greater than 3.
    :return:
    """
    l.sort()
    q1 = len(l) // 4
    q3 = len(l) - q1
    return l[q1:q3]


def get_ep_vals(c_scores, pref_ids):
    n_users = len(c_scores)

    return (
        mean_cov_score(c_scores),
        gym_utils.fairness_idx(c_scores),
        mean_reg_score(c_scores, pref_ids),
        mean_pref_score(c_scores, pref_ids)
    )


def get_data(env_id, model):
    l_c_scores = []
    l_fidx = []
    l_reg_scores = []
    l_pref_scores = []

    iq_means = [1, 1, 1, 1]
    stds = [0, 0, 0, 0]
    eps = 0.001

    i = 0
    # for the five previous iterations, was the change for all variables smaller than or equal to epsilon?
    sati = [False] * 5

    # if the past five iterations satisfied stopping condition and more than 100 iterations, stop.
    while not all(sati) or i < 100:
        print(f"iteration: {i}")

        # use multiprocessing to get coverage scores
        with Environments(env_id, multip.cpu_count(), model) as envs:
            results = envs.get_c_scores_all()
            l_c, l_p = list(zip(*results))

        # get a list of lists, where the inner lists are the scores for n_cpu episodes.
        vals = list(zip(*map(get_ep_vals, l_c, l_p)))

        l_c_scores += vals[0]
        l_fidx += vals[1]
        l_reg_scores += vals[2]
        l_pref_scores += vals[3]

        iq_c_scores = get_interquartile_vals(l_c_scores)
        iq_fidx = get_interquartile_vals(l_fidx)
        iq_reg_scores = get_interquartile_vals(l_reg_scores)
        iq_pref_scores = get_interquartile_vals(l_pref_scores)

        prev_means = iq_means.copy()

        iq_means = [
            np.mean(iq_c_scores),
            np.mean(iq_fidx),
            np.mean(iq_reg_scores),
            np.mean(iq_pref_scores)
        ]

        stds = [
            np.std(iq_c_scores),
            np.std(iq_fidx),
            np.std(iq_reg_scores),
            np.std(iq_pref_scores)
        ]

        change = np.array(iq_means) - np.array(prev_means)
        print(f"New metrics: {iq_means}")
        print(f"New stds: {stds}")
        print(f"Change in metrics: {change}")

        i += 1
        sati = sati[1:] + [not any(abs(change) > eps)]
        print(f"Prev stopping conditions: {sati}")

    return iq_means, stds


#
#
# def get_data(env_id, model):
#
#     totals = np.array([0, 0, 0, 0])
#
#     new_means = np.array([1, 1, 1, 1])
#     ep = 0.01
#
#     i = 0
#     # for the three previous iterations, was the change for all variables smaller than or equal to epsilon?
#     sati = [False] * 5
#
#     # if the past three iterations satisfied stopping condition, stop.
#     while not all(sati) or i < 100:
#         print(f"iteration: {i}")
#
#         totals = totals + np.array(get_metrics(env_id, model))
#         new_means, means = np.array(totals) / (i + 1), new_means
#
#         print(f"New metrics: {new_means}")
#         print(f"Change in metrics: {new_means - means}")
#
#         i += 1
#         sati = sati[1:] + [not any(abs(new_means - means) > ep)]
#         print(f"Prev stopping conditions: {sati}")
#
#     return new_means


def write_data(env_id, exp_num, model):
    iq_means, stds = get_data(env_id, model)

    directory = f"experiments/experiment #{exp_num}"

    with open(f"{directory}/data", 'w') as f:
        f.write(
            f"Mean coverage score: {iq_means[0]} ± {stds[0]} \n"
            f"Fairness index: {iq_means[1]} ± {stds[1]} \n"
            f"Mean preferred score: {iq_means[2]} ± {stds[2]} \n"
            f"Mean regular score: {iq_means[3]} ± {stds[3]} \n")

    with open(f"{directory}/settings", 'w') as f:
        settings = uav_gym.envs.env_settings.Settings()
        json.dump(settings.V, f)

    model.save(f"{directory}/model")


def make_mp4(exp_num, env, model):
    directory = f"experiments/experiment #{exp_num}"
    reg_user_locs, pref_user_locs, l_uav_locs, c_scores_all, c_scores_reg, c_scores_pref, fidx = get_graph_data(env, model)
    a = animate.AnimatedScatter(reg_user_locs, pref_user_locs, l_uav_locs.tolist(),
                                c_scores_all, c_scores_reg, c_scores_pref, fidx,
                                cov_range=env.cov_range, comm_range=env.comm_range, sim_size=env.sim_size)

    f = rf"{directory}/animation.mp4"
    writervideo = animation.FFMpegWriter(fps=10)
    a.ani.save(f, writer=writervideo)


def show_mp4(env, model):
    reg_user_locs, pref_user_locs, l_uav_locs, c_scores_all, c_scores_reg, c_scores_pref, fidx = get_graph_data(env, model)
    a = animate.AnimatedScatter(reg_user_locs, pref_user_locs, l_uav_locs.tolist(),
                                c_scores_all, c_scores_reg, c_scores_pref, fidx,
                                cov_range=env.cov_range, comm_range=env.comm_range, sim_size=env.sim_size)
    plt.show()


if __name__ == '__main__':
    env_id = 'uav-v0'
    models_dir = "rl-baselines3-zoo/logs"

    model = PPO.load(f"{models_dir}/old_reward/ppo/uav-v0_1/best_model")
    env_v = 'v5'
    # models_dir = f"models/{env_v}/PPO"
    # model = PPO.load(f"{models_dir}/1600000.zip")

    # env = gym.make('uav-v0', demonstration=False)
    # # env.seed(0)
    # env.reset()

    exp_num = 12

    directory = f"experiments/experiment #{exp_num}"

    if os.path.isdir(directory):
        if len(os.listdir(directory)) != 0:
            inp = input(f'Are you sure you want to overwrite experiment {exp_num}? y/n ')
            if inp == 'n':
                exp_num += 1
                directory = f"experiments/experiment #{exp_num}"
                os.makedirs(directory)
    else:
        os.makedirs(directory)


    write_data(env_id, exp_num, model)
    # make_mp4(exp_num, env, model)
    # show_mp4(env, model)
