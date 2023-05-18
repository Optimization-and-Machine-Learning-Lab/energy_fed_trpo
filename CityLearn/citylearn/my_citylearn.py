import importlib
import os
from pathlib import Path
from typing import Any, List, Mapping, Tuple, Union
from gym import Env, spaces
import numpy as np
import pandas as pd
from citylearn.my_building import Building
from citylearn.cost_function import CostFunction
from citylearn.data import DataSet, EnergySimulation, CarbonIntensity, Pricing, Weather
from citylearn.reward_function import RewardFunction
from citylearn.utilities import read_json
from citylearn.rendering import get_background, RenderBuilding, get_plots

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
            return [list(self.buildings[b].observations.values()) + [1 if b==j else 0 for j in range(len(self.buildings))] for b in range(len(self.buildings))]
        else:
            return [list(b.observations.values()) for b in self.buildings]   # original one

    def reset(self):
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

        observations = {s: v for s, v in self.schema['observations'].items() if v['active']}
        actions = {a: v for a, v in self.schema['actions'].items() if v['active']}
        simulation_start_time_step = self.schema['simulation_start_time_step']
        simulation_end_time_step = self.schema['simulation_end_time_step']
        time_steps = (simulation_end_time_step - simulation_start_time_step) + 1
        seconds_per_time_step = self.schema['seconds_per_time_step']
        buildings = ()
        
        for building_name, building_schema in self.schema['buildings'].items():
            if building_schema['include']:
                # data
                energy_simulation = pd.read_csv(os.path.join(self.schema['root_directory'],building_schema['energy_simulation'])).iloc[simulation_start_time_step:simulation_end_time_step + 1].copy()
                energy_simulation = EnergySimulation(*energy_simulation.values.T)       # all building data, hour, month and so on
                weather = pd.read_csv(os.path.join(self.schema['root_directory'],building_schema['weather'])).iloc[simulation_start_time_step:simulation_end_time_step + 1].copy()
                weather = Weather(*weather.values.T)

                carbon_intensity = None
                pricing = None
                    
                # observation and action metadata
                inactive_observations = [] if building_schema.get('inactive_observations', None) is None else building_schema['inactive_observations']
                inactive_actions = [] if building_schema.get('inactive_actions', None) is None else building_schema['inactive_actions']
                observation_metadata = {s: False if s in inactive_observations else True for s in observations}
                # {'month': True, 'hour': True, 'outdoor_dry_bulb_temperature': True, 'outdoor_relative_humidity': True, 'non_shiftable_load': True, 'solar_generation': True, 'electrical_storage_soc': True, 'net_electricity_consumption': True, 'electricity_pricing': True}
                action_metadata = {a: False if a in inactive_actions else True for a in actions}
                # {'electrical_storage': True}

                # construct building
                building = Building(energy_simulation, weather, observation_metadata, action_metadata, carbon_intensity=carbon_intensity, pricing=pricing, name=building_name, seconds_per_time_step=seconds_per_time_step)

                # update devices
                device_metadata = ['electrical_storage', 'pv']

                for name in device_metadata:
                    if building_schema.get(name, None) is None:
                        device = None
                    else:
                        device_type = building_schema[name]['type']
                        device_module = '.'.join(device_type.split('.')[0:-1])
                        device_name = device_type.split('.')[-1]
                        constructor = getattr(importlib.import_module(device_module),device_name)
                        attributes = building_schema[name].get('attributes',{})
                        attributes['seconds_per_time_step'] = seconds_per_time_step
                        device = constructor(**attributes)
                        building.__setattr__(name, device)
                
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
        reward_function_constructor = getattr(importlib.import_module(reward_function_module), reward_function_name)
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
