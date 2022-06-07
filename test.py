import gym
from typing import Tuple

import uav_gym
from uav_gym import utils as gym_utils
import animate

from stable_baselines3 import PPO
import numpy as np
from operator import add
import matplotlib.pyplot as plt
from matplotlib import animation
import json


def get_c_scores(i, env, model):
    print(f"trial: {i}")

    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)

    c_scores = env.cov_scores / env.timestep
    return c_scores, env.pref_users


def get_locs(env, model):
    uav_locs = []

    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        uav_locs.append(obs['uav_locs'])

    return obs['user_locs'], uav_locs


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


def get_metrics(env, model, n_trials=100) -> Tuple[float, float, float, float]:
    l_c_scores, l_pref_ids = map(np.array,
                                 map(list,
                                     zip(*[get_c_scores(i, env, model)
                                           for i in range(n_trials)])))
    n_users = len(l_pref_ids[0])

    return (
        mean_cov_score(l_c_scores),
        mean_fair_ind(l_c_scores, n_users),
        mean_pref_scores(l_c_scores, l_pref_ids),
        mean_reg_scores(l_c_scores, l_pref_ids)
    )


def conv_locs(locs):
    scaled_locs = gym_utils.scale(locs, s=env.scale, d='up')
    return [scaled_locs[::2], scaled_locs[1::2]]


def get_data():
    totals = [0, 0, 0, 0]

    means = np.array([0, 0, 0, 0])
    new_means = np.array([1, 1, 1, 1])
    ep = 0.01

    i = 0
    while True:
        if not any(abs(new_means - means) > ep): # and i > 10:
            break

        totals = list(map(add, totals, get_metrics(env, model, 10)))
        new_means, means = np.array(totals) / (i + 1), new_means

        print(f"iteration: {i}")
        print(abs(new_means - means))
        print(new_means)
        i += 1

    return new_means


def write_data(exp_num):
    avg_cov_score, avg_fair_ind, avg_pref_score, avg_reg_score = get_data()

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


def make_mp4(exp_num, env, model):
    directory = f"experiments/experiment #{exp_num}"
    user_locs, uav_locs = get_locs(env, model)

    user_locs = conv_locs(user_locs)
    l_uav_locs = list(map(conv_locs, uav_locs))

    a = animate.AnimatedScatter(user_locs, l_uav_locs, cov_range=env.cov_range, comm_range=env.comm_range,
                                sim_size=env.sim_size)
    # plt.show()

    f = rf"{directory}/animation.mp4"
    writervideo = animation.FFMpegWriter(fps=60)
    a.ani.save(f, writer=writervideo)


if __name__ == '__main__':
    # models_dir = "rl-baselines3-zoo/logs/ppo"
    #
    # model = PPO.load(f"{models_dir}/uav-v0_13/best_model.zip", env=env)

    env_v = 'v5'
    models_dir = f"models/{env_v}/PPO"

    env = gym.make('uav-v0')
    # env.seed(0)
    env.reset()

    model = PPO.load(f"{models_dir}/800000.zip", env=env)

    exp_num = 3
    write_data(exp_num)
    make_mp4(exp_num, env, model)
