import numpy as np
import matplotlib.pyplot as plt


class UAV:
    def __init__(self):
        self.bat_lev = 100
        self.pos = np.array([0, 0])


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
        self.state = 0
        self.charge_time = 3

    def update_charge_time(self):
        if self.state == 0:
            self.charge_time = 3

        elif self.state == 2:
            self.charge_time -= 1



def get_charger_direction(charger_loc, target_loc):
    charger_dir = target_loc - charger_loc

    if np.array_equal(target_loc, charger_loc):
        move = [0, 0]
    elif np.abs(charger_dir)[0] >= np.abs(charger_dir)[1]:
        if charger_dir[0] >= 0:
            move = [1, 0]
        else:
            move = [-1, 0]
    else:
        if charger_dir[1] >= 0:
            move = [0, 1]
        else:
            move = [0, -1]

    return np.array(move)


class Charging:
    def __init__(self, l_uav_locs):
        self.uav = UAV()
        self.charger = Charger()

        self.home_loc = np.array([0, 0])

        self.dist = 1
        self.bat_dep_rate = 10
        self.max_dist = 5

        self.i = 0
        self.l_uav_locs = l_uav_locs

        self.dep_time = 0

    def get_target_loc(self):
        if self.charger.state == 0:
            return self.home_loc

        elif self.charger.state == 1:
            return self.l_uav_locs[self.dep_time + self.max_dist]

        elif self.charger.state == 2:
            return self.uav.pos

        elif self.charger.state == 3:
            return self.home_loc

    def update_charger_state(self):
        if self.charger.state == 0 and self.uav.bat_lev <= self.max_dist * self.bat_dep_rate:
            self.charger.state = 1

        elif self.charger.state == 1 and np.array_equal(self.uav.pos, self.charger.pos):
            self.charger.state = 2

        elif self.charger.state == 2 and self.uav.bat_lev == 100:
            self.charger.state = 3

        elif self.charger.state == 3 and np.array_equal(self.home_loc, self.charger.pos):
            self.charger.state = 0

    def move_charger(self):
        goal_loc = self.get_target_loc()
        self.charger.pos += get_charger_direction(self.charger.pos, goal_loc)

    def update_uav_bat(self):
        self.uav.bat_lev -= 10

        if self.charger.state == 2:
            if self.charger.charge_time == 0:
                self.uav.bat_lev = 100

    def move_uav(self):
        self.uav.pos = self.l_uav_locs[i]

    def update_dep_time(self):
        if self.charger.state == 0 and self.uav.bat_lev <= self.max_dist * self.bat_dep_rate:
            self.dep_time = self.i



l_uav_locs = np.array([[1, 0], [2, 0], [2, 0], [2, 1], [2, 1], [2, 1], [2, 2], [2, 2], [2, 1]] + [[2, 2]] * 10)
charging = Charging(l_uav_locs)

for i in range(len(charging.l_uav_locs)):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlim(0, charging.max_dist)
    ax1.set_ylim(0, charging.max_dist)

    ax1.scatter([charging.uav.pos[0]], [charging.uav.pos[1]])
    ax1.scatter([charging.charger.pos[0]], [charging.charger.pos[1]])

    if charging.charger.state == 2:
        plt.annotate("charging", charging.uav.pos)
    else:
        plt.annotate(charging.uav.bat_lev, charging.uav.pos)

    plt.show()

    charging.move_uav()
    charging.update_uav_bat()
    charging.update_charger_state()
    charging.charger.update_charge_time()
    charging.update_dep_time()
    charging.move_charger()

    charging.i += 1
