import json

class Settings(object):
    def __init__(self):
        # ----

        self.V = {
            # ----
            # SIMULATION SETTINGS
            'SIM_SIZE': 1000,  # keep sim_size divisible by scale
            'SCALE': 50,  # metres per unit
            'HOME_LOC': [0, 0],
            'MAX_TIMESTEPS': 1800,

            # ----
            # USER SETTINGS
            'NUM_USERS': 100,
            'BOUNDARY_FACTOR': 2,

            # ----
            # UAV SETTINGS
            'NUM_UAV': 4,
            'COV_RANGE': 200,  # radius of the coverage range on the ground
            'COMM_RANGE': 500,  # distance UAVs can be and still communicate with each other or home base.

            'INIT_POSITION': [0, 0],

            # ----
            # REWARD
            'P_OUTSIDE_COV': 0.8,
            'P_DISCONNECT': 100,
            'P_OUT_BOUNDS': 100
        }

        self.V['DIST'] = self.V['SCALE']
