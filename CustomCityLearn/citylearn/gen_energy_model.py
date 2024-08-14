from typing import Iterable, List, Union
import numpy as np
from CustomCityLearn.citylearn.base import Environment
np.seterr(divide = 'ignore', invalid = 'ignore')

class PV():
    def __init__(self, nominal_power: float, **kwargs):
        self.reset()
        self.nominal_power = nominal_power
    
    def reset(self):
        pass

    def get_generation(self, inverter_ac_power_per_kw: Union[float, Iterable[float]]) -> Union[float, Iterable[float]]:
        return self.nominal_power*np.array(inverter_ac_power_per_kw)/1000

class Battery():
    def __init__(self, capacity: float, efficiency=1, **kwargs):
        self.capacity = capacity
        self.efficiency = efficiency

        self.initial_soc = 0    # state of charge
        self.reset()
    
    def reset(self):
        self.soc = [self.initial_soc]
        self.electricity_consumption = [0.0]

    def charge(self, energy: float):
        soc = min(self.soc[-1] + energy*self.efficiency, self.capacity) if energy >= 0 else max(0, self.soc[-1] + energy/self.efficiency)
        self.soc.append(soc[0] if isinstance(soc, np.ndarray) else soc)
        self.electricity_consumption.append(self.soc[-1] - self.soc[-2])
