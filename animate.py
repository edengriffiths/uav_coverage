import time
import networkx as nx
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
import matplotlib.patches

from typing import Tuple

from uav_gym.utils import make_graph_from_locs


class AnimatedScatter(object):
    def __init__(self, user_locs, uav_locs, sim_size):
        self.uav_locs = uav_locs
        self.user_locs = user_locs
        self.sim_size = sim_size

        self.time_per_epoch = 1

        # set up figures and axes
        # ---
        self.fig, self.ax = plt.subplots()

        # set figure size
        self.fig.set_size_inches(6, 6)

        # set axes to always be equal
        plt.axis('scaled')

        # set axis limits
        self.ax.set_xlim(0, self.sim_size)
        self.ax.set_ylim(0, self.sim_size)

        self.title = self.ax.text(0.65 * self.sim_size, 1.05 * self.sim_size, "Time Elapsed: 00:00:00")
        # initialise patches
        # ---
        self.rect = matplotlib.patches.Rectangle((0, 0), 50, 20)

        self.circles = [plt.Circle((0, 0), 200, fc='b', alpha=0.3) for _ in range(len(self.uav_locs[0][0]))]

        # set up animation
        self.ani = animation.FuncAnimation(self.fig, self.update,
                                           init_func=self.setup,
                                           frames=len(self.uav_locs),
                                           interval=50,
                                           repeat=False)

    def setup(self):
        """
        Initialise the figure for the animation
        """
        x_uavs = self.uav_locs[0][0]
        y_uavs = self.uav_locs[0][1]

        # add users
        self.users = self.ax.scatter(self.user_locs[0], self.user_locs[1], s=20, c='gray')

        # add UAVs
        self.uavs = self.ax.scatter(x_uavs, y_uavs, s=20, c='r')

        # add coverage of UAVs
        for circle in self.circles:
            self.ax.add_patch(circle)

        # add connection links between UAVs
        c = get_connections(x_uavs, y_uavs)
        lc = LineCollection(c, linestyles=':')
        self.uav_connection = self.ax.add_collection(lc)

        return (self.title, self.users, self.uavs, self.uav_connection, *self.circles)

    def update(self, i):
        x_uavs = self.uav_locs[i][0]
        y_uavs = self.uav_locs[i][1]

        # Update time
        self.title.set_text(f"Time Elapsed: {time.strftime(('%H:%M:%S'), time.gmtime(i * self.time_per_epoch))}")

        # Update UAV positions
        self.uavs.set_offsets(np.c_[x_uavs, y_uavs])

        # Update UAV coverage positions
        for circle, x, y in zip(self.circles, x_uavs, y_uavs):
            circle.center = (x, y)

        # Update links between UAVs
        c = get_connections(x_uavs, y_uavs)
        self.uav_connection.set_segments(c)

        return (self.title, self.users, self.uavs, self.uav_connection, *self.circles)


def edge_to_locs(g: nx.Graph, e: Tuple[int]):
    assert e in g.edges, "Edge, e, not in graph, g"

    return [g.nodes[e[0]]['pos'], g.nodes[e[1]]['pos']]


def get_locs_of_connected(g: nx.Graph):
    locs = []

    for e in g.edges:
        locs.append(edge_to_locs(g, e))

    return locs


def get_connections(x_uavs, y_uavs):
    # TODO: Read these values from somewhere
    g = make_graph_from_locs(list(zip(x_uavs, y_uavs)), home_loc=[0, 0], comm_range=5)
    return get_locs_of_connected(g)


if __name__ == '__main__':
    pass

