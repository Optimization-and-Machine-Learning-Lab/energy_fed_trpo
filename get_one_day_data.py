import numpy as np

import sys

from Citylearn.citylearn.my_citylearn import CityLearnEnv

schema_filepath = './CityLearn/citylearn/data/my_data/schema_eval.json'
env = CityLearnEnv(schema_filepath)


state = env.reset()
done = False
states = []
reward = 0
i = 0
my_reward = 0
# -9 price
# -10 consumption
while not done:
    i += 1
    # print(i)
    states.append([s[-10] for s in state])
    # print(state[0][-9], end=", ")
    # print(state)
    state, r, done, _ = env.step([1.0] * 5)
    # reward += r[0]
    # print(info)
    # print(state)
# print(reward)
states = np.array(states)
for i in range(5):
    # print(states[:, i])
    print("net_consumption = ", end='')
    print(np.array2string(states[:, i], separator=","))