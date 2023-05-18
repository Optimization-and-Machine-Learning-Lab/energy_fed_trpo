import math
from typing import List, Mapping, Union
from gym import spaces
import numpy as np
from citylearn.base import Environment
from citylearn.data import EnergySimulation, CarbonIntensity, Pricing, Weather
from citylearn.my_energy_model import Battery, PV
from citylearn.preprocessing import Encoder, PeriodicNormalization, OnehotEncoding, RemoveFeature, Normalize, NoNormalization


# TODO: remove everything about price and carbon
class Building():
    def __init__(
        self, energy_simulation: EnergySimulation, weather: Weather, observation_metadata: Mapping[str, bool], action_metadata: Mapping[str, bool],
        pricing: Pricing = None, electrical_storage: Battery = None, pv: PV = None, name: str = None, **kwargs
    ):
        r"""Initialize `Building`.

        Parameters
        ----------
        energy_simulation : EnergySimulation
            Temporal features, cooling, heating, dhw and plug loads, solar generation and indoor environment time series.
        weather : Weather
            Outdoor weather conditions and forecasts time sereis.
        observation_metadata : dict
            Mapping of active and inactive observations.
        action_metadata : dict
            Mapping od active and inactive actions.
        carbon_intensity : CarbonIntensity, optional
            Carbon dioxide emission rate time series.
        pricing : Pricing, optional
            Energy pricing and forecasts time series.
        dhw_storage : StorageTank, optional
            Hot water storage object for domestic hot water.
        cooling_storage : StorageTank, optional
            Cold water storage object for space cooling.
        heating_storage : StorageTank, optional
            Hot water storage object for space heating.
        electrical_storage : Battery, optional
            Electric storage object for meeting electric loads.
        dhw_device : Union[HeatPump, ElectricHeater], optional
            Electric device for meeting hot domestic hot water demand and charging `dhw_storage`.
        cooling_device : HeatPump, optional
            Electric device for meeting space cooling demand and charging `cooling_storage`.
        heating_device : Union[HeatPump, ElectricHeater], optional
            Electric device for meeting space heating demand and charging `heating_storage`.
        pv : PV, optional
            PV object for offsetting electricity demand from grid.
        name : str, optional
            Unique building name.

        Other Parameters
        ----------------
        **kwargs : dict
            Other keyword arguments used to initialize super class.
        """

        self.name = name
        self.energy_simulation = energy_simulation
        self.weather = weather
        self.electrical_storage = Battery(0.0, 0.0)
        self.pv = PV(0.0)
        
        self.observation_metadata = observation_metadata
        self.action_metadata = action_metadata
        self.__observation_epsilon = 0.0 # to avoid out of bound observations
        self.active_observations = [k for k, v in self.observation_metadata.items() if v]
        # TODO: remove useless ones??? use active_observations not observations_names
        self.observations_names = ["month", "day_type", "hour", "outdoor_dry_bulb_temperature", "outdoor_dry_bulb_temperature_predicted_6h", "outdoor_dry_bulb_temperature_predicted_12h", "outdoor_dry_bulb_temperature_predicted_24h", "outdoor_relative_humidity", "outdoor_relative_humidity_predicted_6h", "outdoor_relative_humidity_predicted_12h", "outdoor_relative_humidity_predicted_24h", "diffuse_solar_irradiance", "diffuse_solar_irradiance_predicted_6h", "diffuse_solar_irradiance_predicted_12h", "diffuse_solar_irradiance_predicted_24h", "direct_solar_irradiance", "direct_solar_irradiance_predicted_6h", "direct_solar_irradiance_predicted_12h", "direct_solar_irradiance_predicted_24h", "carbon_intensity", "non_shiftable_load", "solar_generation", "electrical_storage_soc", "net_electricity_consumption", "electricity_pricing", "electricity_pricing_predicted_6h", "electricity_pricing_predicted_12h", "electricity_pricing_predicted_24h"]
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
        }

        # for key in self.active_observations:
        for key in self.active_observations:
            if key == 'net_electricity_consumption':
                net_electric_consumption = self.energy_simulation.non_shiftable_load\
                                            + (self.electrical_storage.capacity/0.8)\
                                                - data['solar_generation']
    
                low_limit.append(-max(abs(net_electric_consumption)))
                high_limit.append(max(abs(net_electric_consumption)))

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
            **{k: v[self.time_step] for k, v in vars(self.energy_simulation).items()},
            **{k: v[self.time_step] for k, v in vars(self.weather).items()},
            'solar_generation':self.pv.get_generation(self.energy_simulation.solar_generation[self.time_step]),
            **{
                'electrical_storage_soc':self.electrical_storage.soc[self.time_step],
            },
            'net_electricity_consumption': self.__net_electricity_consumption[self.time_step],
        }

        observations = {k: data[k] if k in data.keys() else 0. for k in self.active_observations}
        # print(observations)
        unknown_observations = list(set([k for k in self.active_observations]).difference(observations.keys()))
        assert len(unknown_observations) == 0, f'Unkown observations: {unknown_observations}'
        return observations

    def reset(self):
        self.time_step = 0
        self.electrical_storage.reset()
        self.pv.reset()
        self.__solar_generation = self.pv.get_generation(self.energy_simulation.solar_generation)*-1    # solar generation value / 200
        self.__net_electricity_consumption = []
        self.update_consumption()

    def next_time_step(self):
        self.time_step += 1
        self.update_consumption()

    def update_consumption(self):
        # net electricity consumption
        net_electricity_consumption = self.electrical_storage.electricity_consumption[self.time_step] \
                        + self.energy_simulation.non_shiftable_load[self.time_step] \
                            + self.__solar_generation[self.time_step]
        self.__net_electricity_consumption.append(net_electricity_consumption)
    
    def apply_actions(self, electrical_storage_action: float = 0):
        energy = electrical_storage_action*self.electrical_storage.capacity
        self.electrical_storage.charge(energy)

    @property
    def net_electricity_consumption(self):
        return self.__net_electricity_consumption

    # TODO: no need for normalization
    @property
    def observation_encoders(self) -> List[Encoder]:
        r"""Get observation value transformers/encoders for use in agent algorithm.

        The encoder classes are defined in the `preprocessing.py` module and include `PeriodicNormalization` for cyclic observations,
        `OnehotEncoding` for categorical obeservations, `RemoveFeature` for non-applicable observations given available storage systems and devices
        and `Normalize` for observations with known mnimum and maximum boundaries.
        
        Returns
        -------
        encoders : List[Encoder]
            Encoder classes for observations ordered with respect to `active_observations`.
        """

        remove_features = ['net_electricity_consumption']
        demand_observations = {
            'electrical_storage_soc': np.nansum(np.nansum([
                self.energy_simulation.non_shiftable_load
            ], axis = 0)),
            'non_shiftable_load': np.nansum(self.energy_simulation.non_shiftable_load),
        }
        remove_features += [k for k, v in demand_observations.items() if v == 0]
        remove_features = [f for f in remove_features if f in self.active_observations]
        encoders = []

        for i, observation in enumerate(self.active_observations):
            if observation in ['month', 'hour']:
                encoders.append(PeriodicNormalization(self.observation_space.high[i]))
            
            elif observation == 'day_type':
                encoders.append(OnehotEncoding([1, 2, 3, 4, 5, 6, 7, 8]))
            
            elif observation == "daylight_savings_status":
                encoders.append(OnehotEncoding([0, 1]))
            
            elif observation in remove_features:
                # encoders.append(RemoveFeature())
                encoders.append(NoNormalization())
            
            else:
                encoders.append(Normalize(self.observation_space.low[i], self.observation_space.high[i]))

        return encoders
