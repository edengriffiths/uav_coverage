from uav_gym.uav_gym.envs.uav_env_v6 import SimpleUAVEnv as Env_v6
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt
import os


class Graph:
    """
    The graph class is designed to collect and plot data on how different environments and models perform and then plot.

    Collecting the data:
    1. Get the independent variable values.
    2. Get the relevant models (by either reading existing saved versions or making new models).
    3. Get the dependent variables values by running the model on the environment.

    Making changes:
    Adding an independent variable
        ~ Add it to the list of valid independent variables.
        ~ Add a function for making independent variables values to the _get_ind_vals function.
        ~ Add the independent variable to the Env class __init__ parameters.

    Adding a dependent variable
        ~ Add the data to be collected to the _get_dep_vals function along with a method for aggregating the data.
    """
    def __init__(self, ind_var: str, dep_var: str, n: int, env_v: str, new_models: bool = False):
        """
        :param ind_var: the independent variable
        :param dep_var: the dependent variable
        :param n: number of points
        :param env_v: the environment version
        """
        assert ind_var in ['n_uavs', 'n_users', 'cov_range', 'comm_range'], \
            f"invalid independent variable, {ind_var}"
        assert dep_var in ['cov_score', 'users_covered', 'energy_use', 'rewards'], \
            f"invalid dependent variable, {dep_var}"
        assert env_v in [f'v{i}' for i in range(7)]

        self.env_v = env_v

        self.ind_var = ind_var
        self.dep_var = dep_var
        self.n = n

        self.n_uavs = [5] * n
        self.n_users = [100] * n
        self.cov_range = [2] * n
        self.comm_range = [5] * n

        self.ind_vals = self._get_ind_vals()

        if new_models:
            self.models, self.envs = self._make_models()
        else:
            try:
                self.models, self.envs = self._get_models(), self._get_envs()
            except FileNotFoundError:
                self.models, self.envs = self._make_models()

        self.dep_vals = self._get_dep_vals()

    def _get_models(self):
        directory = f"./models/{self.env_v}/{self.ind_var}"

        models = []

        for ind_val in self.ind_vals:
            filename = f"{self.ind_var}={ind_val}.zip"
            f_path = directory + '/' + filename
            models.append(PPO.load(f_path))

        return models

    def _get_envs(self):
        return list(map(Env_v6, self.n_uavs, self.n_users, self.cov_range, self.comm_range))

    def _make_models(self):
        # TODO: Prompt user to confirm overwrite of model.
        envs = list(map(Env_v6, self.n_uavs, self.n_users, self.cov_range, self.comm_range))
        models = [PPO('MultiInputPolicy', env) for env in envs]
        for i in range(len(models)):
            models[i].learn(total_timesteps=10000)
            models[i].save(f"./models/{self.env_v}/{self.ind_var}/{self.ind_var}={self.ind_vals[i]}.zip")

            print(str((i + 1) / self.n * 100), '%')

        return models, envs

    def _get_ind_vals(self):
        if self.ind_var == 'n_uavs':
            self.n_uavs = [i + 1 for i in range(5)]
            ind_vals = self.n_uavs

        elif self.ind_var == 'n_users':
            self.n_users = [75 + 25 * i for i in range(5)]
            ind_vals = self.n_users

        elif self.ind_var == 'cov_range':
            self.cov_range = [0.5 * i + 1 for i in range(5)]
            ind_vals = self.cov_range

        elif self.ind_var == 'comm_range':
            self.comm_range = [i + 1 for i in range(5)]
            ind_vals = self.comm_range

        else:
            raise ValueError(f"Invalid independent variable, {self.ind_var}")

        return ind_vals

    def _get_dep_vals(self):
        assert len(self.models) == len(self.envs)

        avg_cov_scores = []
        n_users_covereds = []
        avg_energy_uses = []
        avg_rewards = []

        dep_vals = []

        for i in range(len(self.models)):
            env = self.envs[i]
            model = self.models[i]

            rewards = []
            cov_score = None
            energy_used = None

            obs = env.reset()

            for _ in range(100):
                action, _states = model.predict(obs)
                obs, reward, done, info = env.step(action)

                cov_score = env.state['cov_score']
                energy_used = env.state['energy_used']
                time = env.timestep * env.time_per_epoch

                rewards.append(reward)

            avg_cov_scores.append(np.mean(cov_score))
            n_users_covereds.append(len(cov_score > 0))
            avg_energy_uses.append(np.mean(energy_used))
            avg_rewards.append(np.mean(rewards))

        if self.dep_var == 'cov_score':
            dep_vals = avg_cov_scores.copy()
        elif self.dep_var == 'users_covered':
            dep_vals = n_users_covereds.copy()
        elif self.dep_var == 'energy_use':
            dep_vals = avg_energy_uses.copy()
        elif self.dep_var == 'rewards':
            dep_vals = avg_rewards.copy()

        return dep_vals

    def graph(self):
        plt.scatter(self.ind_vals, self.dep_vals)
        plt.xlabel(get_xaxis_label(self.ind_var))
        plt.ylabel(get_yaxis_label(self.dep_var))

        # TODO: Prompt user to confirm overwrite of graphs.

        directory = f"./graphs/{self.env_v}/{self.ind_var}"
        filename = f"{self.dep_var}.png"
        file_path = directory + '/' + filename

        if os.path.isfile(file_path):
            os.remove(file_path)

        if not os.path.isdir(directory):
            os.makedirs(directory)

        plt.savefig(file_path, bbox_inches='tight')
        plt.show()


def get_xaxis_label(ind_var):
    if ind_var == 'n_uavs':
        return "Number of UAVs"
    elif ind_var == 'n_users':
        return"Number of users"
    elif ind_var == 'cov_range':
        return "UAV coverage range (units)"
    elif ind_var == 'comm_range':
        return "UAV communication range (units)"
    else:
        return ValueError(f"Invalid independent variable, {ind_var}")


def get_yaxis_label(dep_var):
    if dep_var == 'cov_score':
        return "Average coverage score"
    elif dep_var == 'users_covered':
        return "Number of users provided with coverage"
    elif dep_var == 'energy_use':
        return "Average energy usage in units"
    elif dep_var == 'rewards':
        return "Average reward"
    else:
        return ValueError(f"Invalid dependent variable, {dep_var}")


ind_vars = ['n_uavs', 'n_users', 'cov_range', 'comm_range']
dep_vars = ['cov_score', 'users_covered', 'energy_use', 'rewards']


for ind_var in ind_vars:
    for dep_var in dep_vars:
        g = Graph(ind_var, dep_var, 5, 'v6')
        g.graph()