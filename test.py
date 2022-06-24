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
    gym_environment = 'uav-v0'

    def __init__(self, environments_n, model):
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
        self.envs = [gym.make(self.gym_environment) for _ in range(self.environments_n)]

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


def mean_cov_score(l_c_scores: np.array([[float]])) -> float:
    return l_c_scores.mean()


def get_fair_ind(c_scores: np.array([float]), n_users: int) -> float:
    # if all the coverage scores are zero, the fairness index is 1.
    if any(c_scores):
        return sum(c_scores) ** 2 / (n_users * sum(c_scores ** 2))
    else:
        return 1.0


def mean_fair_ind(l_c_scores: np.array([[float]]), n_users: int) -> float:
    return np.array([get_fair_ind(c_scores, n_users) for c_scores in l_c_scores]).mean()


def get_pref_scores(c_scores: np.array([float]), pref_ids: np.array([0 | 1])) -> np.array([float]):
    # add 1 to coverage score and then subtract it to handle the case where the coverage score equals 0.
    _c_scores = c_scores + 1
    pref = pref_ids * _c_scores
    return pref[pref != 0] - 1


def get_reg_scores(c_scores: np.array([float]), pref_ids: np.array([0 | 1])) -> np.array([float]):
    # add 1 to coverage score and then subtract it to handle the case where the coverage score equals 0.
    _c_scores = c_scores + 1
    reg = (1 - pref_ids) * _c_scores
    return reg[reg != 0] - 1


def mean_pref_scores(l_c_scores: np.array([[float]]), pref_ids: np.array([[0 | 1]])) -> float:
    return \
        np.concatenate(
            np.array(
                list(
                    map(get_pref_scores, l_c_scores, pref_ids)
                ),
                dtype=object
            )
        ).mean()


def mean_reg_scores(l_c_scores: np.array([float]), pref_ids: np.array([0 | 1])) -> float:
    return \
        np.concatenate(
            np.array(
                list(
                    map(get_reg_scores, l_c_scores, pref_ids)
                ),
                dtype=object
            )
        ).mean()


def get_metrics(model) -> Tuple[float, float, float, float]:
    # use multiprocessing to
    with Environments(multip.cpu_count(), model) as envs:
        results = envs.get_c_scores_all()
        l_c_scores, l_pref_ids = np.array(list(zip(*results)))

    n_users = len(l_pref_ids[0])

    return (
        mean_cov_score(l_c_scores),
        mean_fair_ind(l_c_scores, n_users),
        mean_pref_scores(l_c_scores, l_pref_ids),
        mean_reg_scores(l_c_scores, l_pref_ids)
    )


def conv_locs(locs, sim_size):
    locs_ = locs * sim_size
    return list(list(loc) for loc in zip(*locs_))


def get_data(model):
    totals = [0, 0, 0, 0]

    new_means = np.array([1, 1, 1, 1])
    ep = 0.01

    i = 0
    # for the three previous iterations, was the change for all variables smaller than or equal to epsilon?
    sati = [False] * 5

    # if the past three iterations satisfied stopping condition, stop.
    while not all(sati):
        print(f"iteration: {i}")

        totals = list(map(add, totals, get_metrics(model)))
        new_means, means = np.array(totals) / (i + 1), new_means

        print(f"New metrics: {new_means}")
        print(f"Change in metrics: {abs(new_means - means)}")

        i += 1
        sati = sati[1:] + [not any(abs(new_means - means) > ep)]
        print(f"Prev stopping conditions: {sati}")

    return new_means


def write_data(exp_num, model):
    avg_cov_score, avg_fair_ind, avg_pref_score, avg_reg_score = get_data(model)

    directory = f"experiments/experiment #{exp_num}"

    with open(f"{directory}/data", 'w') as f:
        f.write(
            f"Mean coverage score: {avg_cov_score} \n"
            f"Fairness index: {avg_fair_ind} \n"
            f"Mean preferred score: {avg_pref_score} \n"
            f"Mean regular score: {avg_reg_score} \n")

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
    models_dir = "rl-baselines3-zoo/logs"

    model = PPO.load(f"{models_dir}/old_reward/ppo/uav-v0_1/best_model")
    env_v = 'v5'
    # models_dir = f"models/{env_v}/PPO"
    # model = PPO.load(f"{models_dir}/1600000.zip")

    env = gym.make('uav-v0', demonstration=False)
    # env.seed(0)
    env.reset()

    exp_num = 10

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

    write_data(exp_num, model)
    make_mp4(exp_num, env, model)
    # show_mp4(env, model)
