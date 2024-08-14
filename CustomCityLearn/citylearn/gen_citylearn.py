import importlib
import os
from pathlib import Path
from typing import Any, List, Mapping, Tuple, Union
from gym import Env, spaces
import numpy as np
import pandas as pd
from CustomCityLearn.citylearn.gen_building import Building
from CustomCityLearn.citylearn.cost_function import CostFunction
from CustomCityLearn.citylearn.data import DataSet, EnergySimulation, CarbonIntensity, Pricing, Weather
from CustomCityLearn.citylearn.gen_reward_function import RewardFunction
from CustomCityLearn.citylearn.utilities import read_json
from CustomCityLearn.citylearn.rendering import get_background, RenderBuilding, get_plots

class CityLearnEnv(Env):
    def __init__(self, schema: Union[str, Path, Mapping[str, Any]], **kwargs):
        self.schema = schema
        self.__rewards = None
        self.buildings, self.time_steps, self.seconds_per_time_step,\
            self.reward_function, self.one_hot = self.__load()
        super().__init__(**kwargs)
        self.time_step = 0

        self.observation_space = [b.observation_space for b in self.buildings]
        self.action_space = [b.action_space for b in self.buildings]

    @property
    def observations(self) -> List[List[float]]:
        """Observations at current time step.
        
        Notes
        -----
        If `central_agent` is True, a list of 1 sublist containing all building observation values is returned in the same order as `buildings`. 
        The `shared_observations` values are only included in the first building's observation values. If `central_agent` is False, a list of sublists 
        is returned where each sublist is a list of 1 building's observation values and the sublist in the same order as `buildings`.
        """

        if self.one_hot:
            tmp = [[1 if b==j else 0 for j in range(len(self.buildings))] + list(self.buildings[b].observations.values()) for b in range(len(self.buildings))]
            tmp = np.array([np.concatenate((t[:-1], t[-1]))for t in tmp]).astype(np.float32)
            return tmp
        else:
            tmp = [list(b.observations.values()) for b in self.buildings]
            tmp = np.array([np.concatenate((t[:-1], t[-1]))for t in tmp]).astype(np.float32)
            return tmp   # original one


    # test, temp_random \in [20, 21], hum_random \in [50, 51]
    # train, temp_random \in [0, 20], shum_randomr \in [0, 50]
    def reset(self):
        self.time_step = 0
        for building in self.buildings:
            building.reset()
        # self.__net_electricity_consumption = []
        # self.__net_electricity_consumption.append(sum([b.net_electricity_consumption[self.time_step] for b in self.buildings]))

        return self.observations

    def step(self, actions):
        for building, building_actions in zip(self.buildings, actions):
            building.apply_actions(building_actions)
            building.next_time_step()
        # self.__net_electricity_consumption.append(sum([b.net_electricity_consumption[self.time_step] for b in self.buildings]))
        self.time_step += 1
        self.reward_function.diff_square = [(actions[i] - self.buildings[i].electrical_storage.electricity_consumption[-1]) ** 2 
                                            for i in range(len(self.buildings))]
        reward = self.get_reward()
        done = (self.time_step == self.time_steps - 1)

        return self.observations, reward, done, {}

    def get_reward(self) -> List[float]:
        """Calculate agent(s) reward(s) using :attr:`reward_function`.
        
        Returns
        -------
        reward: List[float]
            Reward for current observations. If `central_agent` is True, `reward` is a list of length = 1 else, `reward` has same length as `buildings`.
        """

        self.reward_function.electricity_consumption = [b.net_electricity_consumption[self.time_step] for b in self.buildings]
        self.reward_function.electricity_price = [b.net_electricity_consumption_price[self.time_step] for b in self.buildings]
        reward = self.reward_function.calculate()
        return reward

    def __load(self) -> Tuple[List[Building], int, float, RewardFunction, bool, List[str]]:
        """Return `CityLearnEnv` and `Controller` objects as defined by the `schema`.
        
        Returns
        -------
        buildings : List[Building]
            Buildings in CityLearn environment.
        time_steps : int
            Number of simulation time steps.
        seconds_per_time_step: float
            Number of seconds in 1 `time_step` and must be set to >= 1.
        reward_function : RewardFunction
            Reward function class instance.
        """
        
        if isinstance(self.schema, (str, Path)) and os.path.isfile(self.schema):
            schema_filepath = Path(self.schema) if isinstance(self.schema, str) else self.schema
            self.schema = read_json(self.schema)
            self.schema['root_directory'] = os.path.split(schema_filepath.absolute())[0] if self.schema['root_directory'] is None\
                else self.schema['root_directory']
        elif isinstance(self.schema, str) and self.schema in DataSet.get_names():
            self.schema = DataSet.get_schema(self.schema)
            self.schema['root_directory'] = '' if self.schema['root_directory'] is None else self.schema['root_directory']
        elif isinstance(self.schema, dict):
            self.schema['root_directory'] = '' if self.schema['root_directory'] is None else self.schema['root_directory']
        else:
            raise UnknownSchemaError()

        root_directory = self.schema['root_directory']
        observations = {s: v for s, v in self.schema['observations'].items() if v['active']}
        actions = {a: v for a, v in self.schema['actions'].items() if v['active']}
        simulation_start_time_step = self.schema['simulation_start_time_step']
        simulation_end_time_step = self.schema['simulation_end_time_step']
        time_steps = (simulation_end_time_step - simulation_start_time_step) + 1
        seconds_per_time_step = self.schema['seconds_per_time_step']
        buildings = ()

        # ac_efficiency = [1, 1.2, 2.1, 0.9, 0.8]
        # solar_intercept = [3, 4, 5, 3.5, 2.5]
        # solar_efficiency = [1, 1.2, 0.5, 0.8, 1.2]#[3, 5, 2, 2.5, 5]
        # solar_panel = [0.6, 0.5, 0.5, 0.9, 0.6]
        
        for building_name, building_schema in self.schema['buildings'].items():
            if building_schema['include']:
                # data
                energy_simulation = pd.read_csv(os.path.join(self.schema['root_directory'],building_schema['energy_simulation'])).copy()
                energy_simulation = EnergySimulation(*energy_simulation.values.T)       # all building data, hour, month and so on
                weather = pd.read_csv(os.path.join(root_directory,building_schema['weather'])).iloc[simulation_start_time_step:simulation_end_time_step + 1].copy()
                weather = Weather(*weather.values.T)

                if building_schema.get('carbon_intensity', None) is not None:
                    carbon_intensity = pd.read_csv(os.path.join(self.schema['root_directory'],building_schema['carbon_intensity'])).iloc[simulation_start_time_step:simulation_end_time_step + 1].copy()
                    carbon_intensity = carbon_intensity['kg_CO2/kWh'].tolist()
                    carbon_intensity = CarbonIntensity(carbon_intensity)
                else:
                    carbon_intensity = None

                if building_schema.get('pricing', None) is not None:
                    pricing = pd.read_csv(os.path.join(self.schema['root_directory'],building_schema['pricing'])).copy()
                    pricing = Pricing(*pricing.values.T)
                else:
                    pricing = None
                    
                # observation and action metadata
                inactive_observations = [] if building_schema.get('inactive_observations', None) is None else building_schema['inactive_observations']
                inactive_actions = [] if building_schema.get('inactive_actions', None) is None else building_schema['inactive_actions']
                observation_metadata = {s: False if s in inactive_observations else True for s in observations}
                # {'month': True, 'hour': True, 'outdoor_dry_bulb_temperature': True, 'outdoor_relative_humidity': True, 'non_shiftable_load': True, 'solar_generation': True, 'electrical_storage_soc': True, 'net_electricity_consumption': True, 'electricity_pricing': True}
                action_metadata = {a: False if a in inactive_actions else True for a in actions}
                # {'electrical_storage': True}

                # construct building
                building = Building(#building_schema["ac_efficiency"], building_schema["solar_efficiency"], building_schema["solar_panel"], building_schema["solar_intercept"], 
                                    building_schema,
                                    energy_simulation, weather, observation_metadata, action_metadata, carbon_intensity=carbon_intensity, pricing=pricing, 
                                    name=building_name, seconds_per_time_step=seconds_per_time_step)

                # update devices
                # device_metadata = ['electrical_storage', 'pv']

                # for name in device_metadata:
                #     if building_schema.get(name, None) is None:
                #         device = None
                #     else:
                #         device_type = building_schema[name]['type']
                #         device_module = '.'.join(device_type.split('.')[0:-1])
                #         device_name = device_type.split('.')[-1]
                #         constructor = getattr(importlib.import_module(device_module),device_name)
                #         attributes = building_schema[name].get('attributes',{})
                #         attributes['seconds_per_time_step'] = seconds_per_time_step
                #         device = constructor(**attributes)
                #         building.__setattr__(name, device)
                
                building.observation_space = building.estimate_observation_space()
                building.action_space = building.estimate_action_space()
                buildings += (building,)
                
            else:
                continue

        reward_function_type = self.schema['reward_function']['type']
        reward_function_attributes = self.schema['reward_function'].get('attributes',None)
        reward_function_attributes = {} if reward_function_attributes is None else reward_function_attributes
        reward_function_module = '.'.join(reward_function_type.split('.')[0:-1])
        reward_function_name = reward_function_type.split('.')[-1]
        reward_function_constructor = getattr(importlib.import_module(f'CustomCityLearn.{reward_function_module}'), reward_function_name)
        agent_count = len(buildings)
        reward_function = reward_function_constructor(agent_count=agent_count,**reward_function_attributes)

        one_hot = self.schema["personalization"]

        return buildings, time_steps, seconds_per_time_step, reward_function, one_hot

class Error(Exception):
    """Base class for other exceptions."""

class UnknownSchemaError(Error):
    """Raised when a schema is not a data set name, dict nor filepath."""
    __MESSAGE = 'Unknown schema parsed into constructor. Schema must be name of CityLearn data set,'\
        ' a filepath to JSON representation or `dict` object of a CityLearn schema.'\
        ' Call citylearn.data.DataSet.get_names() for list of available CityLearn data sets.'
  
    def __init__(self,message=None):
        super().__init__(self.__MESSAGE if message is None else message)
