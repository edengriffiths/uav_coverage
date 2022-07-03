import gym
from typing import Tuple

import uav_gym
import animate
import uav_gym.utils as gym_utils
from baseline_agents.greedy import get_fake_greedy_action

from stable_baselines3 import PPO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
import json

import os
import sys

import multiprocessing as multip


class Environments:

    def __init__(self, env_id, n_uavs, cov_range, pref_prop, pref_factor, environments_n, model):
        self.env_id = env_id
        self.n_uavs = n_uavs
        self.cov_range = cov_range
        self.pref_prop = pref_prop
        self.pref_factor = pref_factor
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
        self.envs = [gym.make(self.env_id, n_uavs=self.n_uavs, cov_range=self.cov_range, pref_prop=self.pref_prop, pref_factor=self.pref_factor) for _ in range(self.environments_n)]

    @staticmethod
    def get_data_single(env):
        obs = env.reset()
        done = False
        while not done:
            # model comes from global scope so that pickling still works
            # action, _states = model.predict(obs, deterministic=True)
            # action = get_fake_greedy_action(env)
            action = env.action_space.sample()
            obs, rewards, done, info = env.step(action)

        state = env.denormalize_obs(obs)

        return state['cov_scores'], state['pref_users'], env.disconnect_count

    def get_data_all(self):
        results = self.pool.map(self.get_data_single, self.envs)
        return results


def mean_cov_score(c_scores: np.ndarray) -> float:
    return c_scores.mean()


def mean_pref_score(c_scores: np.ndarray, pref_ids: np.ndarray) -> float:
    """
    :param c_scores: a list of c_scores from one environment episode.
    :param pref_ids: a list of pref_ids from the same environment.
    :return: the mean score of prioritised users.
    """
    filt_c_scores = c_scores[pref_ids.astype(bool)]
    return filt_c_scores.mean()


def mean_reg_score(c_scores: np.ndarray, pref_ids: np.ndarray) -> float:
    """
    :param c_scores: a list of c_scores from one environment episode.
    :param pref_ids: a list of pref_ids from the same environment.
    :return: the mean score of regular users.
    """
    filt_c_scores = c_scores[~pref_ids.astype(bool)]
    return filt_c_scores.mean()


def get_ep_vals(c_scores, pref_ids, dconnects):
    return (
        mean_cov_score(c_scores),
        gym_utils.fairness_idx(c_scores),
        mean_pref_score(c_scores, pref_ids),
        mean_reg_score(c_scores, pref_ids),
        dconnects
    )


def exclude_outliers(df_metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out all rows that have an outlier in any columns.
    """
    q1_c_all = df_metrics['cov (all)'].quantile(0.25)
    q3_c_all = df_metrics['cov (all)'].quantile(0.75)
    iqr_c_all = q3_c_all - q1_c_all

    q1_c_pref = df_metrics['cov (pref)'].quantile(0.25)
    q3_c_pref = df_metrics['cov (pref)'].quantile(0.75)
    iqr_c_pref = q3_c_pref - q1_c_pref

    q1_c_reg = df_metrics['cov (reg)'].quantile(0.25)
    q3_c_reg = df_metrics['cov (reg)'].quantile(0.75)
    iqr_c_reg = q3_c_reg - q1_c_reg

    q1_fidx = df_metrics['f idx'].quantile(0.25)
    q3_fidx = df_metrics['f idx'].quantile(0.75)
    iqr_fidx = q3_fidx - q1_fidx

    q1_dconn = df_metrics['dconnects'].quantile(0.25)
    q3_dconn = df_metrics['dconnects'].quantile(0.75)
    iqr_dconn = q3_dconn - q1_dconn

    df_filtered = df_metrics.query(
        '(@q1_c_all - 1.5 * @iqr_c_all) <= `cov (all)` <= (@q3_c_all + 1.5 * @iqr_c_all) &'
        '(@q1_c_pref - 1.5 * @iqr_c_pref) <= `cov (pref)` <= (@q3_c_pref + 1.5 * @iqr_c_pref) &'
        '(@q1_c_reg - 1.5 * @iqr_c_reg) <= `cov (reg)` <= (@q3_c_reg + 1.5 * @iqr_c_reg) &'
        '(@q1_fidx - 1.5 * @iqr_fidx) <= `f idx` <= (@q3_fidx + 1.5 * @iqr_fidx) &'
        '(@q1_dconn - 1.5 * @iqr_dconn) <= `dconnects` <= (@q3_dconn + 1.5 * @iqr_dconn)'
    )

    return df_filtered


def get_data(env_id, n_uavs, cov_range, pref_prop, pref_fac, model):
    metric_names = ['cov (all)', 'f idx', 'cov (pref)', 'cov (reg)', 'dconnects']

    df_metrics = pd.DataFrame(columns=metric_names)

    # summarised metrics including outliers
    df_summarised_all = pd.DataFrame(index=metric_names)

    # summarised metrics excluding outliers
    df_summarised_nout = pd.DataFrame(index=metric_names)

    # for the five previous iterations, was the change for all variables smaller than or equal to epsilon?
    eps = 0.001
    sati = [False] * 5

    i = 0
    # if the past five iterations satisfied stopping condition and more than 100 iterations, stop.
    while not all(sati) or i < 1000 // multip.cpu_count():
        print(f"iteration: {i}")

        # use multiprocessing to get coverage scores, pref_ids and disconnect counts.
        with Environments(env_id, n_uavs, cov_range, pref_prop, pref_fac, multip.cpu_count(), model) as envs:
            results = envs.get_data_all()
            l_c, l_p, l_dc = list(zip(*results))

        # append the new metrics to the df
        df_metrics = pd.concat(
            [df_metrics,
             pd.DataFrame(
                 list(map(get_ep_vals, l_c, l_p, l_dc)),
                 columns=metric_names
             )]
        )

        df_filtered = exclude_outliers(df_metrics)

        # if first iteration, don't calculate change in means.
        if i == 0:
            df_summarised_all = df_metrics.describe(include='all')
            df_summarised_nout = df_filtered.describe(include='all')

            print(df_summarised_all)
        else:
            prev_means = df_summarised_all.loc['mean'].copy()

            df_summarised_all = df_metrics.describe(include='all')

            df_summarised_nout = df_filtered.describe(include='all')

            change = df_summarised_all.loc['mean'] - prev_means

            sati = sati[1:] + [not any(abs(change) / prev_means > eps)]

            print(df_summarised_all)
            print(f"Change in means: {change.tolist()}")
            print(f"Prev stopping conditions: {sati}")

        i += 1

    return df_metrics, df_summarised_all, df_summarised_nout


def write_data(env_id, n_uavs, cov_range, pref_prop, pref_fac, model, directory):
    df_metrics, df_all, df_nout = get_data(env_id, n_uavs, cov_range, pref_prop, pref_fac, model)

    with open(f"{directory}/data_raw.csv", 'w') as f:
        f.write(
            df_metrics.to_csv())

    with open(f"{directory}/data_all.csv", 'w') as f:
        f.write(
            df_all.to_csv())

    with open(f"{directory}/data_nout.csv", 'w') as f:
        f.write(
            df_nout.to_csv())

    with open(f"{directory}/settings", 'w') as f:
        settings = uav_gym.envs.env_settings.Settings()
        json.dump(settings.V, f)

    model.save(f"{directory}/model")


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


def make_mp4(exp_num, env, model):
    directory = f"experiments/experiment #{exp_num}"
    reg_user_locs, pref_user_locs, l_uav_locs, c_scores_all, c_scores_reg, c_scores_pref, fidx = get_graph_data(env,
                                                                                                                model)
    a = animate.AnimatedScatter(reg_user_locs, pref_user_locs, l_uav_locs.tolist(),
                                c_scores_all, c_scores_reg, c_scores_pref, fidx,
                                cov_range=env.cov_range, comm_range=env.comm_range, sim_size=env.sim_size)

    f = rf"{directory}/animation.mp4"
    writervideo = animation.FFMpegWriter(fps=10)
    a.ani.save(f, writer=writervideo)


def show_mp4(env, model):
    reg_user_locs, pref_user_locs, l_uav_locs, c_scores_all, c_scores_reg, c_scores_pref, fidx = get_graph_data(env,
                                                                                                                model)
    a = animate.AnimatedScatter(reg_user_locs, pref_user_locs, l_uav_locs.tolist(),
                                c_scores_all, c_scores_reg, c_scores_pref, fidx,
                                cov_range=env.cov_range, comm_range=env.comm_range, sim_size=env.sim_size)
    plt.show()


if __name__ == '__main__':

    if len(sys.argv) == 7:
        env_id = sys.argv[1]
        n_uavs = int(sys.argv[2])
        cov_range = int(sys.argv[3])
        pref_prop = int(sys.argv[4])
        pref_fac = int(sys.argv[5])
        exp_name = sys.argv[6]

    else:
        # env_id = 'uav-v8'
        raise TypeError(f"test.py requires one argument, env_id: str, {len(sys.argv) - 1} given")

    exp_vals = f"{n_uavs}_{cov_range}_{pref_prop}_{pref_fac}"
    # models_dir = f"rl-baselines3-zoo/logs/{exp_name}/{exp_vals}"
    # model_id = f"{env_id}_1"

    # model = PPO.load(f"{models_dir}/ppo/{model_id}/best_model")

    # env = gym.make(env_id, demonstration=False)
    # # env.seed(0)
    # env.reset()

    model_id = ''
    model = None
    directory = f"experiments/{exp_name}/experiment #{exp_vals}"

    if os.path.isdir(directory):
        if len(os.listdir(directory)) != 0:
            inp = input(f'Are you sure you want to overwrite experiment {model_id}? y/n ')
            if inp == 'n':
                exp_vals += "_t"
                directory = f"experiments/{exp_name}/experiment #{exp_vals}"
                os.makedirs(directory)
    else:
        os.makedirs(directory)

    write_data(env_id, n_uavs, cov_range, pref_prop, pref_fac, model, directory)
    # make_mp4(exp_num, env, model)
    # show_mp4(gym.make(env_id, alpha=alpha, beta=beta, gamma=gamma, delta=delta), model)
