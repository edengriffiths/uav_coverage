
import numpy as np
from functools import reduce
from scipy.spatial import distance


def conv_uav_locs(uav_locs):
    return np.array([uav_locs[x:x + 2] for x in range(0, len(uav_locs), 2)])


def get_move(action, dist):
    if action == 0:
        return [0, dist]
    elif action == 1:
        return [dist, 0]
    elif action == 2:
        return [0, -dist]
    elif action == 3:
        return [-dist, 0]


def inbounds(loc, x_l_bound=0, x_u_bound=10, y_l_bound=0, y_u_bound=10):
    return x_l_bound <= loc[0] <= x_u_bound and y_l_bound <= loc[1] <= y_u_bound


def get_coverage_state_from_uav(uav_loc: np.array, user_locs: np.array, cov_range: int):
    dist_to_users = distance.cdist([uav_loc], user_locs, 'euclidean').flatten()
    return dist_to_users <= cov_range


def get_coverage_state(uav_locs: np.array, user_locs: np.array, cov_range: int):
    coverage_states = list(map(get_coverage_state_from_uav,
                               uav_locs,
                               [user_locs] * len(uav_locs),
                               [cov_range] * len(uav_locs)))

    return reduce(lambda acc, x: acc | x, coverage_states)


# energy used hovering
def energy_hover(t): return t / 10


# energy used flying
def energy_move(t): return t / 9  # FIXME: Energy used moving should have a larger denominator.