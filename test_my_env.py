import json
import numpy as np
import pandas as pd
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

import sys
sys.path.append('CityLearn/')
from citylearn.gen_citylearn import CityLearnEnv

env = CityLearnEnv('/home/yunxiang.li/FRL/CityLearn/citylearn/data/gen_data/schema.json')
# buildings = env.buildings
# encoders = np.array([buildings[i].observation_encoders for i in range(5)])
# print(encoders[0])
# action = [[i*0.01] * 5 for i in range(24)]
action = [[-0.1] * 5 for i in range(24)]

state = env.reset()
print("state[2]", state[2])
# {'month': 1, 'hour': 24, 'outdoor_dry_bulb_temperature': -1.486232030935568, 'outdoor_relative_humidity': 91.86305328428016, 'non_shiftable_load': 1.459056575944642, 'solar_generation': 0.0, 'electrical_storage_soc': 0, 'net_electricity_consumption': 1.459056575944642}
# state = np.array([[j for j in np.hstack(encoders[i]*state[i][:-5]) if j != None] + state[i][-5:] for i in range(5)])
# print(state[0])


done = False
i = 0
while not done:
    print(env.time_step)
    print("action", action[i])
    state, reward, done, info = env.step(action[i])
    i += 1
    print("state[2]", state[2])
    print("reward", reward[2])
