import math
from typing import List, Mapping, Union
from gym import spaces
import numpy as np
from citylearn.base import Environment
from citylearn.data import EnergySimulation, CarbonIntensity, Pricing, Weather
from citylearn.gen_energy_model import Battery, PV
from citylearn.preprocessing import Encoder, PeriodicNormalization, OnehotEncoding, RemoveFeature, Normalize, NoNormalization


# TODO: remove everything about price and carbon
class Building():
    def __init__(
        self, ac_efficiency, solar_efficiency, panel, solar_intercept, energy_simulation: EnergySimulation, weather: Weather, observation_metadata: Mapping[str, bool], action_metadata: Mapping[str, bool],
        pricing: Pricing = None, electrical_storage: Battery = None, pv: PV = None, name: str = None, **kwargs
    ):
        self.name = name
        self.ac_efficiency = ac_efficiency
        self.solar_efficiency = solar_efficiency
        self.panel = panel
        self.solar_intercept = solar_intercept

        self.energy_simulation = energy_simulation
        self.weather = weather
        self.electrical_storage = Battery(1.0, 1.0)
        self.pv = PV(1.0)
        self.pricing = pricing
        self.__solar_generation_base = self.energy_simulation.solar_generation
        self.__non_shiftable_load_base = self.energy_simulation.non_shiftable_load
        self.__temperature_base = weather.outdoor_dry_bulb_temperature
        self.__humidity_base = weather.outdoor_relative_humidity
        
        self.observation_metadata = observation_metadata
        self.action_metadata = action_metadata
        self.__observation_epsilon = 0.0 # to avoid out of bound observations
        self.active_observations = [k for k, v in self.observation_metadata.items() if v]
        self.observation_space = self.estimate_observation_space()
        self.action_space = self.estimate_action_space()
        
        self.time_step = 0

    def estimate_observation_space(self) -> spaces.Box:
        r"""Get estimate of observation spaces.

        Find minimum and maximum possible values of all the observations, which can then be used by the RL agent to scale the observations and train any function approximators more effectively.

        Returns
        -------
        observation_space : spaces.Box
            Observation low and high limits.

        Notes
        -----
        Lower and upper bounds of net electricity consumption are rough estimates and may not be completely accurate hence,
        scaling this observation-variable using these bounds may result in normalized values above 1 or below 0.
        """

        low_limit, high_limit = [], []
        data = {
            'solar_generation':np.array(self.pv.get_generation(self.energy_simulation.solar_generation)),
            **vars(self.energy_simulation),
            **vars(self.weather),
            **vars(self.pricing),
        }

        # for key in self.active_observations:
        for key in self.active_observations:
            if key == 'net_electricity_consumption':
                net_electric_consumption = self.energy_simulation.non_shiftable_load\
                                            + (self.electrical_storage.capacity/0.8)\
                                                - data['solar_generation']
    
                low_limit.append(-max(abs(net_electric_consumption)))
                high_limit.append(max(abs(net_electric_consumption)))

            elif key == 'solar_generation':
                low_limit.append(0.0)
                high_limit.append(max(data[key]) * 1.0 / 1000)      # TODO: nominal_power, I am lazy

            elif key in ['electrical_storage_soc']:
                low_limit.append(0.0)
                high_limit.append(1.0)

            else:
                low_limit.append(min(data[key]))
                high_limit.append(max(data[key]))

        low_limit = [v - self.__observation_epsilon for v in low_limit]
        high_limit = [v + self.__observation_epsilon for v in high_limit]
        return spaces.Box(low=np.array(low_limit, dtype='float32'), high=np.array(high_limit, dtype='float32'))

    def estimate_action_space(self) -> spaces.Box:
        return spaces.Box(low=np.array([-1.0], dtype='float32'), high=np.array([1.0], dtype='float32'))

    @property
    def observations(self) -> Mapping[str, float]:
        """Observations at current time step."""

        observations = {}
        data = {
            **{k: v[self.time_step] for k, v in vars(self.pricing).items()},
            # **{k: v[self.time_step] for k, v in vars(self.energy_simulation).items()},
            'hour':self.energy_simulation.hour[self.time_step],
            'outdoor_dry_bulb_temperature':self.__outdoor_dry_bulb_temperature[self.time_step],
            'outdoor_relative_humidity':self.__humidity[self.time_step],
            'non_shiftable_load':self.__non_shiftable_load[self.time_step],
            'solar_generation':self.__solar_generation[self.time_step],
            'electrical_storage_soc':self.electrical_storage.soc[self.time_step],
            'net_electricity_consumption': self.__net_electricity_consumption[self.time_step],
        }

        observations = {k: data[k] if k in data.keys() else 0. for k in self.active_observations}
        # print(observations)
        unknown_observations = list(set([k for k in self.active_observations]).difference(observations.keys()))
        assert len(unknown_observations) == 0, f'Unkown observations: {unknown_observations}'
        return observations

    def reset(self, temp_random, hum_random):
        self.time_step = 0
        self.electrical_storage.reset()
        self.pv.reset()

        self.__outdoor_dry_bulb_temperature = self.__temperature_base + temp_random
        self.__humidity = self.__humidity_base + hum_random
        self.__non_shiftable_load = (self.__humidity-60)/20/self.ac_efficiency + (30-self.__outdoor_dry_bulb_temperature)/25/self.ac_efficiency + self.__non_shiftable_load_base
        self.__solar_generation = self.pv.get_generation((self.__outdoor_dry_bulb_temperature*self.solar_efficiency*self.panel**2) * self.__solar_generation_base) * -1
        
        self.__net_electricity_consumption = []
        self.__net_electricity_consumption_price = []
        self.update_variables()

    def next_time_step(self):
        self.time_step += 1
        self.update_variables()

    def update_variables(self):
        # net electricity consumption
        net_electricity_consumption = self.electrical_storage.electricity_consumption[self.time_step] \
                        + self.__non_shiftable_load[self.time_step] \
                        + self.__solar_generation[self.time_step]
                        # + self.energy_simulation.non_shiftable_load[self.time_step] \
                        #     + self.__solar_generation[self.time_step]

        self.__net_electricity_consumption.append(net_electricity_consumption)

        self.__net_electricity_consumption_price.append(net_electricity_consumption*self.pricing.electricity_pricing[self.time_step])
    
    def apply_actions(self, electrical_storage_action: float = 0):
        energy = electrical_storage_action*self.electrical_storage.capacity
        self.electrical_storage.charge(energy)

    @property
    def net_electricity_consumption(self):
        return self.__net_electricity_consumption

    @property
    def net_electricity_consumption_price(self) -> List[float]:
        return self.__net_electricity_consumption_price

    # @property
    # def observation_encoders(self) -> List[Encoder]:
    #     r"""Get observation value transformers/encoders for use in agent algorithm.

    #     The encoder classes are defined in the `preprocessing.py` module and include `PeriodicNormalization` for cyclic observations,
    #     `OnehotEncoding` for categorical obeservations, `RemoveFeature` for non-applicable observations given available storage systems and devices
    #     and `Normalize` for observations with known mnimum and maximum boundaries.
        
    #     Returns
    #     -------
    #     encoders : List[Encoder]
    #         Encoder classes for observations ordered with respect to `active_observations`.
    #     """

    #     remove_features = ['net_electricity_consumption']
    #     demand_observations = {
    #         'electrical_storage_soc': np.nansum(np.nansum([
    #             self.energy_simulation.non_shiftable_load
    #         ], axis = 0)),
    #         'non_shiftable_load': np.nansum(self.energy_simulation.non_shiftable_load),
    #     }
    #     remove_features += [k for k, v in demand_observations.items() if v == 0]
    #     remove_features = [f for f in remove_features if f in self.active_observations]
    #     encoders = []

    #     for i, observation in enumerate(self.active_observations):
    #         if observation in ['month', 'hour']:
    #             encoders.append(PeriodicNormalization(self.observation_space.high[i]))
            
    #         elif observation == 'day_type':
    #             encoders.append(OnehotEncoding([1, 2, 3, 4, 5, 6, 7, 8]))
            
    #         elif observation == "daylight_savings_status":
    #             encoders.append(OnehotEncoding([0, 1]))
            
    #         elif observation in remove_features:
    #             # encoders.append(RemoveFeature())
    #             encoders.append(NoNormalization())
            
    #         else:
    #             # print("encoder", observation, self.observation_space.low[i], self.observation_space.high[i])
    #             encoders.append(Normalize(self.observation_space.low[i], self.observation_space.high[i]))

    #     return encoders
