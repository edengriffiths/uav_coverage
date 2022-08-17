import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cityblock, cdist
from python_tsp.exact import solve_tsp_dynamic_programming


class UAV:
    def __init__(self, id):
        self.id = id

        self.pos = np.array([0, 0])
        self.vel = 50 / 3

        self.bat_lev = 100
        self.bat_dep_rate = 5

    def __str__(self):
        return f"---------------\n" \
               f"id: {self.id}\n" \
               f"battery level: {self.bat_lev}\n" \
               f"position: {self.pos}\n"


class Charger:
    """
    States:
        0 = Not deployed
        1 = Deployed
        2 = Charging
        3 = Returning
    """
    def __init__(self):
        self.pos = np.array([0, 0])
        self.vel = 50 / 3
        self.state = 0
        self.charge_time = 3

    def update_charge_time(self):
        if self.state == 0:
            self.charge_time = 3

        elif self.state == 2:
            self.charge_time -= 1


class Simulation:
    def __init__(self, l_uav_locs):
        self.n_uavs = len(l_uav_locs)

        self.charger = Charger()
        self.uavs = [UAV(i) for i in range(self.n_uavs)]

        self.l_uav_locs = l_uav_locs

        self.sim_size = 1000
        self.home_loc = [0, 0]
        self.time_per_step = 3

        self.timestep = 0

    def step(self):
        self.timestep += 1
        move_uavs(self.uavs, self.l_uav_locs[:, self.timestep])
        update_uavs_bat(self.uavs)

        if deploy_charger(self.uavs, 30):
            uav_order = get_charge_order(self.home_loc, self.uavs)


def move_uavs(uavs, new_poss):
    for uav, new_pos in zip(uavs, new_poss):
        uav.pos = new_pos


def update_uavs_bat(uavs):
    for uav in uavs:
        uav.bat_lev -= uav.bat_dep_rate


def deploy_charger(uavs: [UAV], min_bat_lev: int):
    if any([uav.bat_lev < min_bat_lev for uav in uavs]):
        return True
    else:
        return False


def get_charge_order(home_loc, uavs: [UAV]):
    coords = [home_loc] + [uav.pos for uav in uavs]
    dist_matrix = cdist(coords, coords, 'cityblock')

    return solve_tsp_dynamic_programming(dist_matrix)[0]

l_uav_locs = np.array([
    [[1, 0], [2, 0], [2, 0], [2, 1], [2, 1], [2, 1], [2, 2], [2, 2], [2, 1]] + [[2, 2]] * 10,
    [[1, 0], [1, 1], [1, 2], [1, 3], [2, 3], [1, 3], [1, 2]] + [[1, 2]] * 12
])

sim = Simulation(l_uav_locs)

for i in range(5):
    sim.step()
    for uav in sim.uavs:
        print(uav)

print(get_charge_order(sim.home_loc, sim.uavs))
#
#
# def get_charger_direction(charger_loc, target_loc):
#     charger_dir = target_loc - charger_loc
#
#     if np.array_equal(target_loc, charger_loc):
#         move = [0, 0]
#     elif np.abs(charger_dir)[0] >= np.abs(charger_dir)[1]:
#         if charger_dir[0] >= 0:
#             move = [1, 0]
#         else:
#             move = [-1, 0]
#     else:
#         if charger_dir[1] >= 0:
#             move = [0, 1]
#         else:
#             move = [0, -1]
#
#     return np.array(move)
#
#
# class ChargingSingle:
#     def __init__(self, uav, charger, l_uav_locs):
#         self.uav = uav
#         self.charger = charger
#
#         self.home_loc = np.array([0, 0])
#
#         self.vel = 1
#         self.bat_dep_rate = 1
#         self.sim_size = 10
#
#         self.i = 0
#         self.l_uav_locs = l_uav_locs
#
#         self.dep_time = 0
#
#     def get_target_loc(self, t_arrival):
#         if self.charger.state == 0:
#             return self.home_loc
#
#         elif self.charger.state == 1:
#             return self.l_uav_locs[t_arrival]
#
#         elif self.charger.state == 2:
#             return self.uav.pos
#
#         elif self.charger.state == 3:
#             return self.home_loc
#
#     def update_charger_state(self):
#         if self.charger.state == 0 and self.uav.bat_lev == 2 * self.sim_size // self.vel * self.bat_dep_rate:
#             self.charger.state = 1
#
#         elif self.charger.state == 1 and np.array_equal(self.uav.pos, self.charger.pos):
#             self.charger.state = 2
#
#         elif self.charger.state == 2 and self.uav.bat_lev == 100:
#             self.charger.state = 3
#
#         elif self.charger.state == 3 and np.array_equal(self.home_loc, self.charger.pos):
#             self.charger.state = 0
#
#     def move_charger(self, t_arrival):
#         goal_loc = self.get_target_loc(t_arrival)
#         self.charger.pos += get_charger_direction(self.charger.pos, goal_loc)
#
#     def update_uav_bat(self):
#         self.uav.bat_lev -= self.bat_dep_rate
#
#         if self.charger.state == 2:
#             if self.charger.charge_time == 0:
#                 self.uav.bat_lev = 100
#
#     def move_uav(self):
#         self.uav.pos = self.l_uav_locs[i]
#
#     def update_dep_time(self):
#         if self.charger.state == 0 and self.uav.bat_lev == 2 * self.sim_size // self.vel * self.bat_dep_rate:
#             t_arrival = 0
#             for step in range(2 * self.sim_size // self.vel):
#                 print(step)
#                 manhattan_dist = cityblock(self.charger.pos, self.l_uav_locs[step])
#                 if manhattan_dist <= step * self.vel:
#                     self.dep_time = self.i + step - manhattan_dist // self.vel
#                     return t_arrival + self.i
#
#     def step(self):
#         self.move_uav()
#         self.update_uav_bat()
#         t_arrival = self.update_dep_time()
#         self.update_charger_state()
#         self.charger.update_charge_time()
#         self.move_charger(t_arrival)
#
#         self.i += 1
#
#
# class ChargingMultiple:
#     def __init__(self, n_uavs, n_chargers, l_uav_locs):
#         """
#         :param l_uav_locs: a list of lists where the inner lists contain a given uav's locations at each timestep.
#         """
#         self.n_uavs = n_uavs
#         self.n_chargers = n_chargers
#         self.l_uav_locs = l_uav_locs
#
#         self.l_uavs: [UAV] = [UAV() for _ in range(self.n_uavs)]
#         self.l_chargers: [Charger] = [Charger() for _ in range(self.n_chargers)]
#
#         self.charger_locs: [[int]] = []
#
#     def match_uav_charger(self):
#         self.pairs = [ChargingSingle(self.l_uavs[i], self.l_chargers[i], self.l_uav_locs[i]) for _ in range(self.n_uavs)]
#
#     def step(self):
#         [pair.step() for pair in self.pairs]
#         self.charger_locs = [self.charger_locs[i].append(self.pairs[0].charger)]


# l_uav_locs1 = np.array([[1, 0], [2, 0], [2, 0], [2, 1], [2, 1], [2, 1], [2, 2], [2, 2], [2, 1]] + [[2, 2]] * 10)
# charging1 = ChargingSingle(UAV(), Charger(), l_uav_locs1)
#
# l_uav_locs2 = np.array([[1, 0], [1, 1], [1, 2], [1, 3], [2, 3], [1, 3], [1, 2]] + [[1, 2]] * 12)
# charging2 = ChargingSingle(UAV(), Charger(), l_uav_locs2)


# for i in range(len(charging1.l_uav_locs)):
#     fig = plt.figure()
#     ax1 = fig.add_subplot(111)
#     ax1.set_xlim(0, charging1.sim_size)
#     ax1.set_ylim(0, charging1.sim_size)
#
#     ax1.scatter([charging1.uav.pos[0]], [charging1.uav.pos[1]])
#     ax1.scatter([charging1.charger.pos[0]], [charging1.charger.pos[1]])
#
#     ax1.scatter([charging2.uav.pos[0]], [charging2.uav.pos[1]])
#     ax1.scatter([charging2.charger.pos[0]], [charging2.charger.pos[1]])
#
#     if charging1.charger.state == 2:
#         plt.annotate("charging", charging1.uav.pos)
#     else:
#         plt.annotate(charging1.uav.bat_lev, charging1.uav.pos)
#
#     if charging2.charger.state == 2:
#         plt.annotate("charging", charging2.uav.pos)
#     else:
#         plt.annotate(charging2.uav.bat_lev, charging2.uav.pos)
#
#     plt.show()
#
#     charging1.step()
#     charging2.step()
#
