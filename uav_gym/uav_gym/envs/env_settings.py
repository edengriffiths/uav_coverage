import json


class Settings(object):
    def __init__(self):
        # ----

        self.V = {
            # ----
            # SIMULATION SETTINGS
            'SIM_SIZE': 1000,  # keep sim_size divisible by scale
            'HOME_LOC': [0, 0],
            'TIME_PER_STEP': 3,

            # ----
            # USER SETTINGS
            'NUM_USERS': 50,
            'BOUNDARY_FACTOR': 2,
            'PREF_PROP': 15,
            'PREF_FACTOR': 4,

            # ----
            # UAV SETTINGS
            'NUM_UAV': 4,
            'COV_RANGE': 200,  # radius of the coverage range on the ground
            'COMM_RANGE': 500,  # distance UAVs can be and still communicate with each other or home base.
            'DIST': 50,
            'INIT_POSITION': [0, 0],

            # ----
            # REWARD
            'P_OUTSIDE_COV': 0.8,
            'P_DISCONNECT': 1,
            'P_OUT_BOUNDS': 1,
            'ALPHA': 1,
            'BETA': 9,
            'GAMMA': 9,
            'DELTA': 1
        }
