import numpy as np
import pandas as pd
from gym import Env
from citylearn.preprocessing import PeriodicNormalization

# TODO: try different house configurations
class RandomBanmaEnv(Env):
    def __init__(self, n_actions=1, action_diff_penalty=0.0):
        super(RandomBanmaEnv, self).__init__()
        self.dim_action = n_actions

        self.time = 0
        self.end_time_step = 24
        self.action_diff_penalty = action_diff_penalty
        self.time_encoder = PeriodicNormalization(24)
        # features: hour, solar generation, non_shiftable_load, carbon_intens, electrical_storage_soc, price
        self.obs_dim = 6
        self.act_dim = 1

        self.daily_solar, self.daily_load, self.price = self.__load()
        self.load_mean = self.daily_load.mean()
        self.solar_mean = self.daily_solar.mean()
        self.price_mean = self.price.mean()

        self.storage = 0.0
        self.capacity = 1.0

        self.last_state = None

    def __load(self):
        # price from the env
        # price = np.array([0.21334247, 0.21334247, 0.21334247, 0.21334247, 0.21334247, 0.21334247,
        #                  0.21334247, 0.21334247, 0.21334247, 0.21334247, 0.21334247, 0.21334247,
        #                  0.21334247, 0.21334247, 0.21334247, 0.21334247, 0.50032877, 0.50032877,
        #                  0.50032877, 0.50032877, 0.50032877, 0.21334247, 0.21334247, 0.21334247])

        # price with sin
        price = np.array([0.2, 0.22610523844401031, 0.25176380902050416, 0.276536686473018, 0.3, 
                 0.32175228580174414, 0.3414213562373095, 0.35867066805824704, 0.37320508075688774, 
                 0.38477590650225735, 0.39318516525781366, 0.3982889722747621, 0.4, 0.3982889722747621, 
                 0.39318516525781366, 0.38477590650225735, 0.3732050807568878, 0.35867066805824704, 0.3414213562373095, 
                 0.3217522858017442, 0.3, 0.276536686473018, 0.2517638090205042, 0.22610523844401031])

        daily_load = np.array([1.11978309, 0.76816295, 0.87315243, 0.83054701, 0.86020929,
                               0.95714514, 1.10605024, 0.81951836, 0.88964521, 1.09147795,
                               1.43532721, 1.38397538, 1.48023339, 1.02653598, 0.97702104,
                               0.97477498, 1.20762802, 1.56440962, 1.76958789, 2.06710011,
                               1.86987534, 1.5113112 , 1.28299162, 1.12902468])

        daily_solar = np.array([0.        , 0.        , 0.        , 0.        , 0.        , 0., 
                                 0.03038468, 0.42636759, 1.18112355, 1.8242703 , 2.34805091, 2.63622672,
                                 2.74510293, 2.62572737, 2.32459339, 1.76772156, 1.12567972, 0.56241727,
                                 0.15598965, 0.00660886, 0.        , 0.        , 0.        , 0.        ])

        # net consumption without battery
        # [ 1.11978309  0.76816295  0.87315243  0.83054701  0.86020929  0.95714514
        # 1.07566556  0.39315077 -0.29147834 -0.73279235 -0.9127237  -1.25225134
        # -1.26486954 -1.59919139 -1.34757235 -0.79294658  0.0819483   1.00199235
        # 1.61359824  2.06049125  1.86987534  1.5113112   1.28299162  1.12902468]
        return daily_solar, daily_load, price

    def generate_state(self, time, storage):
        solar = np.random.normal(self.daily_solar[time], 0.1 * self.solar_mean)
        load = np.random.normal(self.daily_load[time], 0.1 * self.load_mean)
        price = np.random.normal(self.price[time], 0.1 * self.price_mean)

        return list(time * self.time_encoder) + [solar, load, storage, price]

    def reset(self):
        self.time = 0
        self.storage = 0.0
        # state = list(self.time*self.time_encoder) + [self.daily_solar[self.time], 
        #          self.daily_load[self.time], 0.0, self.price[self.time]]
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

        price_reward = price * net_con #grid_con
        
        action_diff = (action - valid_action) ** 2

        reward = - price_reward - self.action_diff_penalty * action_diff

        done = False

        if self.time == self.end_time_step:
            state = self.reset()
            done = True
        else:
            # state = list(self.time*self.time_encoder) + [self.daily_solar[self.time%24], 
            #          self.daily_load[self.time%24], self.storage, self.price[self.time%24]]
            state = self.generate_state(self.time%24, self.storage)
            self.last_state = state

        return np.array(state), reward, done, {"hour": self.time, "valid_action": valid_action, "action_diff": action_diff, "price": -price_reward}

if __name__ == "__main__":
    env = BanmaEnv()

    done = False
    action = [0.0]
    state = env.reset()
    actions = [ 0.88, -0.2,  -0.39, -0.57, -0.61, -0.61, -0.59, -0.65, -0.68, -0.68, -0.67, -0.68,
                -0.69, -0.7,  -0.72, -0.66,  0.98,  0.12,  1.,    1.,    1.,    1.,    0.99,  0.96]
    for i in range(24):
        print("action", actions[i], end=', ')
        action = [actions[i]]
        state, reward, done, info = env.step(action)

        # multiplier = 3
        # hour_day = info["hour"]
        # action = 0.0
        # if hour_day >= 7 and hour_day <= 11:
        #     action = 0.08 * multiplier
        # elif hour_day >= 12 and hour_day <= 15:
        #     action = 0.15 * multiplier
        # elif hour_day >= 16 and hour_day <= 18:
        #     action = -0.11 * multiplier
        # elif hour_day >= 19 and hour_day <= 22:
        #     action = -0.06 * multiplier
        # # Early nightime: store DHW and/or cooling energy
        # if hour_day >= 23 and hour_day <= 24:
        #     action = -0.085 * multiplier
        # elif hour_day >= 1 and hour_day <= 6:
        #     action = -0.2 * multiplier
        # action = np.array([action], dtype=np.float32)
        print("reward", reward)
        print("price", info["price_ratio"])
        print("emission", info["emission_ratio"])
        print()
        # print(state.shape)
        # break
