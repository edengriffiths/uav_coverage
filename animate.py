import time
import networkx as nx
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
import matplotlib.patches
from matplotlib import gridspec

from typing import Tuple, List, Union

from uav_gym.utils import make_graph_from_locs

Loc = List[Union[np.float32]]
Locs = List[Loc]


class AnimatedScatter(object):
    def __init__(self, ruser_locs: Locs, puser_locs: Locs,
                 l_ruavs_locs: List[Locs], l_puav_locs: List[Locs],
                 cov_scores_all: [float], r_cov_scores: [float], p_cov_scores: [float], fidx: [float],
                 cov_range: int, comm_range: int, sim_size: int):
        self.ruser_locs = ruser_locs
        self.p_user_locs = puser_locs

        self.l_ruav_locs = np.array(l_ruavs_locs)
        self.l_puav_locs = np.array(l_puav_locs)

        self.l_cov_scores_all = cov_scores_all
        self.l_cov_scores_r = r_cov_scores
        self.l_cov_scores_p = p_cov_scores
        self.l_fidx = fidx

        self.cov_range = cov_range
        self.comm_range = comm_range
        self.sim_size = sim_size

        self.n_timesteps = len(l_ruavs_locs)
        self.time_per_epoch = 3
        self.time = [self.time_per_epoch * i for i in range(self.n_timesteps)]

        self.n_ruavs = len(l_ruavs_locs[0])
        self.n_puavs = len(l_puav_locs[0])

        # set up figures and axes
        # ---
        self.fig = plt.figure()
        gs = gridspec.GridSpec(2, 2)
        self.ax_env = self.fig.add_subplot(gs[:, :-1])
        self.ax_cov = self.fig.add_subplot(gs[:-1, -1])
        self.ax_fidx = self.fig.add_subplot(gs[-1, -1])

        # set figure size
        self.fig.set_size_inches(12, 4)
        self.fig.subplots_adjust(right=0.7)

        # set env axes to always be equal
        self.ax_env.axis('scaled')

        # set ax_env limits
        self.ax_env.set_xlim(0, self.sim_size)
        self.ax_env.set_ylim(0, self.sim_size)

        # set plot limits
        self.ax_cov.set_ylim(0, 1)
        self.ax_cov.set_xlim(0, max(self.time))

        self.ax_cov.tick_params(labelbottom=False)  # remove ticks from bottom

        self.ax_fidx.set_ylim(0, 1)
        self.ax_fidx.set_xlim(0, max(self.time))

        self.ax_fidx.set_xlabel("Time (s)")



        # initialise patches
        # ---
        self.rect = matplotlib.patches.Rectangle((0, 0), 50, 20)

        self.rcircles = [plt.Circle((0, 0), self.cov_range, fc='b', alpha=0.3) for _ in range(self.n_ruavs)]
        self.pcircles = [plt.Circle((0, 0), self.cov_range, fc='g', alpha=0.3) for _ in range(self.n_puavs)]

        # set up animation
        self.ani = animation.FuncAnimation(self.fig, self.update,
                                           init_func=self.setup,
                                           frames=self.n_timesteps,
                                           interval=50,
                                           repeat=False)

    def setup(self):
        """
        Initialise the figure for the animation
        """

        # ----
        # set up env axes

        x_ruavs, y_ruavs = list(zip(*self.l_ruav_locs[0]))
        x_puavs, y_puavs = list(zip(*self.l_puav_locs[0]))
        ruser_locs = list(zip(*self.ruser_locs))
        puser_locs = list(zip(*self.p_user_locs))

        # add users
        self.rusers = self.ax_env.scatter(ruser_locs[0], ruser_locs[1], s=20, c='gray')
        if len(puser_locs) > 0:
            self.pusers = self.ax_env.scatter(puser_locs[0], puser_locs[1], s=20, c='red')
        else:
            self.pusers = []

        # add UAVs
        self.ruavs = self.ax_env.scatter(x_ruavs, y_ruavs, s=20, c='r')
        self.puavs = self.ax_env.scatter(x_puavs, y_puavs, s=20, c='pink')

        # add coverage of UAVs
        for circle in self.rcircles:
            self.ax_env.add_patch(circle)

        for circle in self.pcircles:
            self.ax_env.add_patch(circle)

        # add connection links between UAVs
        c = get_connections(self.l_ruav_locs[0], comm_range=self.comm_range)
        lc = LineCollection(c, linestyles=':')
        self.ruav_connection = self.ax_env.add_collection(lc)

        c = get_connections(self.l_puav_locs[0], comm_range=self.comm_range)
        lc = LineCollection(c, linestyles=':')
        self.puav_connection = self.ax_env.add_collection(lc)

        # ----
        # set up plot axes

        self.line_all, = self.ax_cov.plot([0], self.l_cov_scores_all[0], c='pink', label='Coverage Score (All)')
        self.line_reg, = self.ax_cov.plot([0], self.l_cov_scores_r[0], c='gray', label='Coverage Score (Regular)')
        self.line_pref, = self.ax_cov.plot([0], self.l_cov_scores_p[0], c='red', label='Coverage Score (Prioritised)')

        self.ax_cov.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

        self.line_fidx, = self.ax_fidx.plot([0], self.l_fidx[0], c='blue', label='Fairness Index')

        self.ax_fidx.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

        return (self.line_all, self.line_reg, self.line_pref, self.line_fidx,
                self.rusers, self.pusers, self.ruavs, self.puavs,
                self.ruav_connection, self.puav_connection,
                *self.rcircles, *self.pcircles)

    def update(self, i):

        # ----
        # Update env axes
        x_ruavs, y_ruavs = list(zip(*self.l_ruav_locs[i]))
        x_puavs, y_puavs = list(zip(*self.l_puav_locs[i]))

        # Update UAV positions
        self.ruavs.set_offsets(np.c_[x_ruavs, y_ruavs])
        self.puavs.set_offsets(np.c_[x_puavs, y_puavs])

        # Update UAV coverage positions
        for circle, x, y in zip(self.rcircles, x_ruavs, y_ruavs):
            circle.center = (x, y)

        for circle, x, y in zip(self.pcircles, x_puavs, y_puavs):
            circle.center = (x, y)

        # Update links between UAVs
        c = get_connections(self.l_ruav_locs[i], comm_range=self.comm_range)
        self.ruav_connection.set_segments(c)

        c = get_connections(self.l_puav_locs[i], comm_range=self.comm_range)
        self.puav_connection.set_segments(c)

        # ----
        # update plot axes
        self.line_all.set_data(self.time[:i+1], self.l_cov_scores_all[:i+1])
        self.line_reg.set_data(self.time[:i+1], self.l_cov_scores_r[:i + 1])
        self.line_pref.set_data(self.time[:i+1], self.l_cov_scores_p[:i + 1])
        self.line_fidx.set_data(self.time[:i+1], self.l_fidx[:i+1])

        return (self.line_all, self.line_reg, self.line_pref, self.line_fidx,
                self.rusers, self.pusers, self.ruavs, self.puavs,
                self.ruav_connection, self.puav_connection,
                *self.rcircles, *self.pcircles)


def edge_to_locs(g: nx.Graph, e: Tuple[int]):
    assert e in g.edges, "Edge, e, not in graph, g"

    return [g.nodes[e[0]]['pos'], g.nodes[e[1]]['pos']]


def get_locs_of_connected(g: nx.Graph):
    locs = []

    for e in g.edges:
        locs.append(edge_to_locs(g, e))

    return locs


def get_connections(uav_locs, comm_range):
    g = make_graph_from_locs(uav_locs.tolist(), home_loc=[0.0, 0.0], comm_range=comm_range)
    return get_locs_of_connected(g)


if __name__ == '__main__':
    # uav_locs = [
    #     [[0, 0], [0, 0]],
    #     [[100, 0], [100, 0]],
    #     [[200, 0], [200, 0]],
    # ]

    ruav_locs = [
        [
            [0, 0], [0, 0], [0, 0]
        ],
        [
            [50, 0], [0, 50], [0, 0]
        ],
        [
            [100, 50], [50, 100], [0, 0]
        ],
        [
            [200, 100], [100, 50], [0, 0]
        ]
    ]

    puav_locs = [
        [
            [0, 0]
        ],
        [
            [0, 50]
        ],
        [
            [0, 100]
        ],
        [
            [50, 100]
        ]
    ]

    reg_user_locs = [[100, 200], [100, 200]]
    pref_user_locs = [[200, 100], [300, 300]]


    c_scores_all = [0, 0.2, 0.3, 0.4]
    c_scores_reg = [0, 0.1, 0.2, 0.3]
    c_scores_pref = [0, 0.3, 0.4, 0.5]
    fidx = [0.5, 0.4, 0.4, 0.5]

    a = AnimatedScatter(reg_user_locs,
                        pref_user_locs,
                        ruav_locs,
                        puav_locs,
                        c_scores_all,
                        c_scores_reg,
                        c_scores_pref,
                        fidx,
                        cov_range=200,
                        comm_range=500,
                        sim_size=1000)
    plt.show()
