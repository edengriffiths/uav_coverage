import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# TODO: Make callable function
# TODO: Show time elapsed


def update(i):
    ax.clear()

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    ax.scatter(list_uav_locs[i][0], list_uav_locs[i][1], s=6000, c='b', label='first', alpha=0.3)
    ax.scatter(user_locs[0], user_locs[1], s=20, c='gray', label='third')
    ax.scatter(list_uav_locs[i][0], list_uav_locs[i][1], s=20, c='r', label='second')
    #

    # # Adding Figure Labels
    # ax.set_title('Trajectory \nTime = ' + str(np.round(t[i],
    #                                                    decimals=2)) + ' sec')
    ax.set_xlabel('x')
    ax.set_ylabel('y')


def setup():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.scatter([0] * 3, [0] * 3, s=6000, c='b', alpha=0.3)
    ax.scatter(user_locs[0], user_locs[1], s=20, c='gray')
    ax.scatter([0] * 3, [0] * 3, s=20, c='r')

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    return fig, ax


if __name__ == '__main__':
    user_locs = [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]]
    user_locs = np.array(list(zip(*user_locs)))
    # list_uav_locs = np.array([[[1, 2, 3], [1, 2, 3]], [[2, 1, 4], [1, 2, 3]], [[2, 1, 4], [0, 3, 2]]])
    list_uav_locs = np.array([[[0], [0]], [[1], [0]], [[1], [0]], [[2], [0]], [[3], [0]], [[2], [0]], [[3], [0]], [[2], [0]], [[1], [0]], [[1], [0]], [[0], [0]]])

    fig, ax = setup()

    line_ani = animation.FuncAnimation(fig, update, interval=1000,
                                       frames=len(list_uav_locs), repeat=False)
    plt.show()
