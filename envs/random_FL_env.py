import numpy as np
import pandas as pd
from gym import Env
from citylearn.preprocessing import PeriodicNormalization

# TODO: try different house configurations
class RandomFLEnv(Env):
    def __init__(self, building_id=0, action_diff_penalty=0.0):
        super(RandomFLEnv, self).__init__()
        self.time = 0
        self.end_time_step = 24
        self.price_coef = 1.0
        self.action_diff_penalty = action_diff_penalty
        self.time_encoder = PeriodicNormalization(24)
        # features: hour, solar generation, non_shiftable_load, electrical_storage_soc, price   
        # # TODO: one-hot for buildings, something private for the buildings, don't aggregate this part jiu zhe yang ba
        self.obs_dim = 9
        self.act_dim = 1
        self.building_id = building_id

        self.daily_solar, self.daily_load, self.price = self.__load()
        self.load_mean = self.daily_load.mean()
        self.solar_mean = self.daily_solar.mean()
        self.price_mean = self.price.mean()
        self.storage = 0.0
        self.capacity = 1.0

        self.id_encode = [0., 0., 0.]
        self.id_encode[building_id] = 1.

        self.last_state = None

    def __load(self):
        # price with sin
        price = np.array([0.2, 0.22610523844401031, 0.25176380902050416, 0.276536686473018, 0.3, 
                 0.32175228580174414, 0.3414213562373095, 0.35867066805824704, 0.37320508075688774, 
                 0.38477590650225735, 0.39318516525781366, 0.3982889722747621, 0.4, 0.3982889722747621, 
                 0.39318516525781366, 0.38477590650225735, 0.3732050807568878, 0.35867066805824704, 0.3414213562373095, 
                 0.3217522858017442, 0.3, 0.276536686473018, 0.2517638090205042, 0.22610523844401031])

        daily_load_0 = np.array([1.11978309, 0.76816295, 0.87315243, 0.83054701, 0.86020929,
                               0.95714514, 1.10605024, 0.81951836, 0.88964521, 1.09147795,
                               1.43532721, 1.38397538, 1.48023339, 1.02653598, 0.97702104,
                               0.97477498, 1.20762802, 1.56440962, 1.76958789, 2.06710011,
                               1.86987534, 1.5113112 , 1.28299162, 1.12902468])
        daily_solar_0 = np.array([0.        , 0.        , 0.        , 0.        , 0.        , 0., 
                                0.03038468, 0.42636759, 1.18112355, 1.8242703 , 2.34805091, 2.63622672,
                                2.74510293, 2.62572737, 2.32459339, 1.76772156, 1.12567972, 0.56241727,
                                0.15598965, 0.00660886, 0.        , 0.        , 0.        , 0.        ])
        
        daily_load_1 = np.array([1.23199787, 0.93239101, 0.7390323,  0.63671716, 0.62135378, 0.58901651,
                                0.72372195, 0.85391257, 0.7758803 , 0.75702422, 0.82366865, 0.86955419,
                                0.94555207, 1.01490129, 1.11632849, 1.2289218 , 1.38111585, 1.51055552,
                                1.41555887, 1.44632175, 1.48303525, 1.58620858, 1.50592838, 1.43760654])

        daily_solar_1 = np.array([0.        , 0.        , 0.        , 0.        , 0.        , 0.,
                                0.01407143, 0.10060265, 0.36446386, 0.85000747, 1.35910078, 1.73225129,
                                1.99024091, 2.07564793, 2.01187868, 1.752272  , 1.32439623, 0.86547022,
                                0.39690627, 0.02101913, 0.        , 0.        , 0.        , 0.        ])
        
        daily_load_2 = np.array([0.84718426, 0.76916794, 0.68303212, 0.57014196, 0.57307572, 0.57804768,
                                0.58258989, 0.61232901, 0.62427926, 0.63573304, 0.67968882, 0.75552722,
                                0.86882288, 0.88747182, 0.98349459, 1.03937964, 1.04109839, 1.08824749,
                                1.03174424, 1.03987603, 0.95918681, 0.94127279, 0.94252558, 0.91111042])

        daily_solar_2 = np.array([0.        , 0.        , 0.        , 0.        , 0.        , 0.,
                                0.01802811, 0.11788893, 0.39756944, 0.9145707 , 1.46692884, 1.869249,
                                2.14590546, 2.23700863, 2.16652147, 1.87964307, 1.40429001, 0.91607386,
                                0.39393407, 0.01230009, 0.        , 0.        , 0.        , 0.        ])

        return [daily_solar_0, daily_solar_1, daily_solar_2][self.building_id], [daily_load_0, daily_load_1, daily_load_2][self.building_id], price

    def generate_state(self, time, storage):
        solar = np.random.normal(self.daily_solar[time], 0.1 * self.solar_mean)
        load = np.random.normal(self.daily_load[time], 0.1 * self.load_mean)
        price = np.random.normal(self.price[time], 0.1 * self.price_mean)

        return list(time * self.time_encoder) + [solar, load, storage, price] + self.id_encode

    def reset(self):
        self.time = 0
        self.storage = 0.0
        # state = list(self.time*self.time_encoder) + [self.daily_solar[self.time], self.daily_load[self.time], 0.0, self.price[self.time]] #+ self.id_encode
        state = self.generate_state(self.time, self.storage)

        self.last_state = state

        return np.array(state)

    def step(self, action):
        self.time += 1
        # solar_gen = self.daily_solar[(self.time-1)%24]
        # load = self.daily_load[(self.time-1)%24]
        solar_gen = self.last_state[2]
        load = self.last_state[3]
        price = self.last_state[5]
        net_con = load - solar_gen      # negative: more solar, positive: more load

        action = action[0]
        action = max(min(action, 1.0), -1.0)        # clip to [-1, 1]
        if action > 0.0:                            # charge
            valid_action = min(action, self.capacity - self.storage)    
        elif action < 0.0 and net_con > 0.0:        # discharge
            valid_action = max(max(action, -net_con), -self.storage)
        else:
            valid_action = 0.0
        self.storage += valid_action
        net_con += valid_action
        net_con = max(net_con, 0)

        # price_reward = self.price[(self.time-1)%24] * net_con   #grid_con
        price_reward = price * net_con
        
        action_diff = (action - valid_action) ** 2

        reward = - price_reward - self.action_diff_penalty * action_diff

        done = False

        if self.time == self.end_time_step:
            state = self.reset()
            done = True
        else:
            # state = list(self.time*self.time_encoder) + [self.daily_solar[self.time%24], self.daily_load[self.time%24], self.storage, self.price[self.time%24]] #+ self.id_encode
            state = self.generate_state(self.time%24, self.storage)
            self.last_state = state

        return np.array(state), reward, done, {"hour": self.time, "valid_action": valid_action, "action_diff": action_diff, "price_reward": -price_reward}

if __name__ == "__main__":
    env = FLEnv(0)

    done = False
    action = [0.0]
    state = env.reset()
    actions = [ 0.88, -0.2,  -0.39, -0.57, -0.61, -0.61, -0.59, -0.65, -0.68, -0.68, -0.67, -0.68,
                -0.69, -0.7,  -0.72, -0.66,  0.98,  0.12,  1.,    1.,    1.,    1.,    0.99,  0.96]
    for i in range(24):
        print("action", actions[i], end=', ')
        action = [actions[i]]
        state, reward, done, info = env.step(action)

        print("reward", reward)
        print("price", info["price_ratio"])
        print("emission", info["emission_ratio"])
        print()
