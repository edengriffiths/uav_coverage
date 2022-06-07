from functools import reduce
from scipy.spatial import distance
import numpy as np
import networkx as nx

from typing import Union, List
from typeguard import check_type

StateLocs = List[int]
RegLocs = List[List[int]]
RegLocsFloat = List[List[float]]


def scale(locs_: Union[RegLocs, StateLocs], s: int, from_state: bool) -> np.array:
    """
    Rescale a list of locations
    :param locs_: 1D np.array of locations.
    :param s: scaling factor
    :param from_state: the direction of scale: true (increase the fidelity), false (decrease the fidelity).
    :return: scaled locs or raise a ValueError if incorrect direction using.
    """
    locs = np.copy(np.array(locs_))

    if from_state:
        return locs * s
    else:
        return (locs / s).round(0).astype(int)


def conv_locs(locs: Union[RegLocs, StateLocs], s: int, from_state: bool) -> np.array:
    """
    Convert the locations of users and UAVs as seen in the state, [x1, y1, x2, y2] to the form [[x1, y1], [x2, y2]]
    """

    if from_state:
        # assert locs.shape == (len(locs),), "Incorrect shape for locs"
        check_type("locs", locs, StateLocs)

        return scale([locs[x:x + 2] for x in range(0, len(locs), 2)],
                     s, from_state)
    else:
        # assert locs.shape == (len(locs), 2), "Incorrect shape for locs"
        try:
            check_type("locs", locs, RegLocs)
        except TypeError:
            print(locs)

        check_type("locs", locs, RegLocs)

        return scale(locs, s, from_state).flatten()


def _constrain_user_loc(user_loc: [float], center: np.array([float]),
                        std: np.array([float]), b_factor: int, rng):

    x_u, y_u = user_loc
    x_c, y_c = center
    if x_c - b_factor * std > x_u or x_u > x_c + b_factor * std:
        x_u = rng.uniform(x_c - b_factor * std, x_c + b_factor * std)

    if y_c - b_factor * std > y_u or y_u > y_c + b_factor * std:
        y_u = rng.uniform(y_c - b_factor * std, y_c + b_factor * std)

    return x_u, y_u


def constrain_user_locs(user_locs: RegLocsFloat, blob_ids: np.array,
                        centers: np.array([[float]]), stds: np.array([[float]]),
                        b_factor: int, rng):
    """
    Constrains a list of user locations to be within b_factor standard deviations of the centers
    :param user_locs: a list of user locations of the form [[x1, y1], [x2, y2], ..., [xn, yn]]
    :param blob_ids: the blob id of each point eg [0, 1, ..., 0]
    :param centers: the center of the blobs
    :param stds: the standard deviations of the blobs
    :param b_factor: the maximum number of standard deviations a user can be from the center.
    :param rng: a random number generator
    :return: return a list of user locations of the same form as user_locs
    """
    check_type("user_locs", user_locs, RegLocsFloat)

    user_locs = np.array(user_locs)

    ul1 = np.array(list(zip(blob_ids, user_locs)), dtype='object')
    ul2 = ul1[ul1[:, 0].argsort()]
    ul3 = np.array(np.split(ul2[:, 1], np.unique(ul2[:, 0], return_index=True)[1][1:]), dtype='object')

    ul_constrained = np.array([
        _constrain_user_loc(user_loc, centers[c_id], stds[c_id], b_factor, rng)
        for c_id in range(len(centers))
        for user_loc in ul3[c_id]
    ])

    return ul_constrained


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


def inbounds(loc: [int], x_ubound, y_ubound, x_lbound=0, y_lbound=0):
    return x_lbound <= loc[0] <= x_ubound and y_lbound <= loc[1] <= y_ubound


def get_coverage_state_from_uav(uav_loc: np.array, user_locs: np.array, cov_range: int):
    dist_to_users = distance.cdist([uav_loc], user_locs, 'euclidean').flatten()
    return dist_to_users <= cov_range


def get_coverage_state(uav_locs: RegLocs, user_locs: RegLocs, cov_range: int):
    check_type("uav_locs", user_locs, RegLocs)
    check_type("user_locs", user_locs, RegLocs)

    coverage_states = [get_coverage_state_from_uav(uav_loc, user_locs, cov_range) for uav_loc in uav_locs]

    return reduce(lambda acc, x: acc | x, coverage_states)


def _get_score(dist: float, cov_range: float, p_factor: float) -> float:
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
        return p_factor * 1 / (1 + dist)


def get_scores(uav_locs: RegLocs, user_locs: RegLocs, cov_range: float, p_factor: float) -> np.array([float]):
    check_type("uav_locs", user_locs, RegLocs)
    check_type("user_locs", user_locs, RegLocs)

    # for each UAV get distance to each user
    dist_to_users = distance.cdist(uav_locs, user_locs, 'euclidean')

    # get the minimum distance to a UAV for each user
    min_dist_to_users = dist_to_users.min(axis=0)

    # compute the score for each user
    user_scores = np.array([_get_score(min_dist_to_users[i], cov_range, p_factor)
                            for i in range(len(min_dist_to_users))])

    return user_scores


def fairness_idx(cov_scores: np.array([float])) -> float:
    """
    Calculates the Jain's Fairness index for the user coverage scores.
    :param cov_scores: an np array of floats between 0 and 1.
    :return: a float between 0 and 1 that represents the fairness of the coverage.
    """
    n_users = len(cov_scores)
    if any(cov_scores):
        return sum(cov_scores) ** 2 / (n_users * sum(cov_scores ** 2))
    else:
        return 1


def make_graph_from_locs(uav_locs: RegLocs, home_loc: List[int], comm_range: int):
    """
    :param uav_locs: np.ndarray of uav location [[x1, y1], [x2, y2],..., [xn, yn]
    :param home_loc: location of the home base [x, y]
    :param comm_range: an integer that represents the communication range of a UAV.
    :return: a networkx graph of the UAVs and home with edges between UAVs that are within the communication range.
    """
    check_type("uav_locs", uav_locs, RegLocs)

    # TODO: Can the home base have a larger range? Can it get data from UAV?
    all_locs = np.array([home_loc] + uav_locs)

    nodes = dict(enumerate(all_locs))
    g = nx.Graph()

    for n, pos in nodes.items():
        g.add_node(n, pos=pos)

    edges = nx.generators.geometric.geometric_edges(g, radius=comm_range, p=2)
    g.add_edges_from(edges)

    return g


def get_disconnected_count(G):
    """
    :param G: networkx graph
    :return: the number of nodes that are unreachable from home.
    """
    home = 0

    # Get the subgraph of all the nodes connected to home
    s = G.subgraph(nx.shortest_path(G, home))

    # Return the number of nodes not connected to home
    return G.number_of_nodes() - s.number_of_nodes()


# energy used hovering
def energy_hover(t): return t / 10


# energy used flying
def energy_move(t): return t / 9  # FIXME: Energy used moving should have a larger denominator.
