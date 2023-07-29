import json
import numpy as np
import pandas as pd
from TRPO.models import *
from torch.autograd import Variable
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

import sys
sys.path.append('CityLearn/')
from citylearn.gen_citylearn import CityLearnEnv

env = CityLearnEnv('/home/yunxiang.li/FRL/CityLearn/citylearn/data/gen_data/schema.json')
action = [[-0.1] * 5 for i in range(24)]

state = env.reset()
print("state[2]", state[2])

building_count = 5
num_inputs = env.observation_space[0].shape[0] + building_count
num_actions = env.action_space[0].shape[0]
# print("num_inputs", num_inputs)
# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)

policy_net = Policy(num_inputs, num_actions)
value_net = Value(num_inputs)

print(policy_net(torch.from_numpy(state)))
print(value_net(torch.from_numpy(state)))

done = False
i = 0
while not done:
    print(env.time_step)
    print("action", action[i])
    state, reward, done, info = env.step(action[i])
    i += 1
    print("state[2]", state[2])
    print("reward", reward[2])
