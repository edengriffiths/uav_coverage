
import numpy as np
from functools import reduce
from scipy.spatial import distance
import numpy as np
import networkx as nx


def conv_locs(locs):
    return np.array([locs[x:x + 2] for x in range(0, len(locs), 2)])


def constrain_user_loc(user_loc, center, std, rng):
    x_u, y_u = user_loc
    x_c, y_c = center
    if x_c - 3 * std > x_u > x_c + 3 * std:
        x_u = rng.random.uniform(x_c - 3 * std, x_c + 3 * std)

    if y_c - 3 * std > y_u > y_c + 3 * std:
        y_u = rng.random.uniform(y_c - 3 * std, y_c + 3 * std)

    return x_u, y_u

# TODO: Need to add not move action.
def get_move(action, dist):
    if action == 0:
        return [0, 0]
    elif action == 1:
        return [0, dist]
    elif action == 2:
        return [dist, 0]
    elif action == 3:
        return [0, -dist]
    elif action == 4:
        return [-dist, 0]


def inbounds(loc, x_u_bound, y_u_bound, x_l_bound=0, y_l_bound=0):
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


def get_score(dist: float, cov_range: float, p_factor: float) -> float:
    """
    user score = 1, if user is within coverage range of a UAV
                 p_factor * 1/(1 + distance) from the closest UAV, otherwise
    The score for users outside the coverage range incentivises the UAVs to move closer to the users, however, it will
    always be less than 1.

    :param dist: distance to closest uav
    :param cov_range: the coverage range of the UAVs
    :param p_factor: the punishment factor for being outside the coverage range.
    :return: the score as a float
    """
    if dist <= cov_range:
        return 1
    else:
        return p_factor * 1/(1 + dist)


def get_scores(uav_locs: np.array, user_locs: np.array, cov_range: float, p_factor: float) -> np.array([float]):
    # for each UAV get distance to users
    dist_to_users = distance.cdist(uav_locs, user_locs, 'euclidean')

    # get the minimum distance to a UAV for each user
    min_dist_to_users = dist_to_users.min(axis=0)

    # compute the score for each user
    user_scores = np.array([get_score(min_dist_to_users[i], cov_range, p_factor)
                            for i in range(len(min_dist_to_users))])

    return user_scores

def make_graph_from_locs(uav_locs, home_loc, comm_range):
    """
    :param uav_locs: np.ndarray of uav location [[x1, y1], [x2, y2],..., [xn, yn]
    :param home_loc: location of the home base [x, y]
    :param comm_range: an integer that represents the communication range of a UAV.
    :return: a networkx graph of the UAVs and home with edges between UAVs that are within the communication range.
    """
    # TODO: Can the home base have a larger range? Can it get data from UAV?
    all_locs = [home_loc] + uav_locs

    nodes = dict(enumerate(all_locs))
    g = nx.Graph()

    for n, pos in nodes.items():
        g.add_node(n, pos=pos)

    edges = nx.generators.geometric.geometric_edges(g, radius=comm_range, p=2)
    g.add_edges_from(edges)

    return g

# energy used hovering
def energy_hover(t): return t / 10


# energy used flying
def energy_move(t): return t / 9  # FIXME: Energy used moving should have a larger denominator.