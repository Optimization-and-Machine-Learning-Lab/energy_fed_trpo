import sys
import os
import json
import hashlib

import random
import pandas as pd
import numpy as np
import torch

from citylearn.data import DataSet
from citylearn.energy_model import PV
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def make_env(env):
    """
    Initialize the environment.

    Args:
        env_path (str): Path to the environment configuration file.
        dataset (str): Path to the dataset file.

    Returns:
        env (gym.Env): The environment.
    """

    # Wrap the environment with Monitor for logging and DummyVecEnv for normalization
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    return env

def create_building_csv(building_ix, base_path: str = 'data/schemas/warm_up/'):

    # Create the building csv file combining all the existing buildings

    building_csv = f'Building_{building_ix}.csv'

    # Read the 3 existing csv files for the buildings

    building_1 = np.genfromtxt(base_path + 'Building_1.csv', delimiter=',')
    building_2 = np.genfromtxt(base_path + 'Building_2.csv', delimiter=',')
    building_3 = np.genfromtxt(base_path + 'Building_3.csv', delimiter=',')

    # Get the headers for the csv

    headers = np.genfromtxt(base_path + 'Building_1.csv', delimiter=',', dtype=str)[0].tolist()
    headers = ",".join(headers)

    # Combine the 3 buildings with random coefficients but keep the first columns intact

    building = (building_1 * random.uniform(0.8, 0.9) + building_2 * random.uniform(0.8, 0.9) + building_3 * random.uniform(0.8, 0.9))[1:] / 4
    building[:,0:3] = building_1[1,0:3]
    building[:,2] = np.array([[random.randint(1,7)]*24 for _ in range(int(720/24))]).flatten()
    building[:,12] = building[:,12].astype(np.int32)
    building[:,14] = 1

    # Save the building csv

    np.savetxt(base_path + building_csv, building, delimiter=',', header=headers, fmt='%s')

def generate_simplified_data(
    base_dataset: str = 'citylearn_challenge_2022_phase_all',
    weather_randomness: dict = {'temperature': {'low': 15, 'high': 20}, 'humidity': {'low': 0, 'high': 50}},
):

    schema = DataSet.get_schema(base_dataset)

    # Read the base schema and process the json

    for building_name, info in schema['buildings'].items():
        
        # Create custom building CSVs

        base_csv = pd.read_csv(os.path.join(schema['root_directory'], info['energy_simulation']))
        weather = pd.read_csv(os.path.join(schema['root_directory'], info['weather']))

        # Define randomness for the weather data

        delta_temp = random.random() * (weather_randomness['temperature']['high'] - weather_randomness['temperature']['low']) + weather_randomness['temperature']['low']
        delta_hum = random.random() * (weather_randomness['humidity']['high'] - weather_randomness['humidity']['low']) + weather_randomness['humidity']['low']

        temp = weather['outdoor_dry_bulb_temperature'] + delta_temp
        hum = weather['outdoor_relative_humidity'] + delta_hum

        # Configure solar panel like it's processed in the library

        # in case device technical specifications are to be randomly sampled, make sure each device per building has a unique seed
        
        md5 = hashlib.md5()
        device_random_seed = 0

        for string in [building_name, 'citylearn.citylearn.Building', 'pv', 'citylearn.energy_model.PV']:
            md5.update(string.encode())
            hash_to_integer_base = 16
            device_random_seed += int(md5.hexdigest(), hash_to_integer_base)

        device_random_seed = int(str(device_random_seed*(schema['random_seed'] + 1))[:9])

        attributes = {
            **info['pv']['attributes'],
            'random_seed': device_random_seed
        }

        pv = PV(**attributes)

        # Generate simpler data for non_shiftable_load based on humidity and temperature 

        base_csv['non_shiftable_load'] = base_csv['non_shiftable_load'] + (hum - 60)/20 + (30 - temp)/25
        base_csv['solar_generation'] = pv.get_generation((temp * attributes['nominal_power'] ** 2) * base_csv['solar_generation']) * -1

        print('test')

    # with open(ref_schema + 'schema.json', 'r') as f:

    #     schema = f.read()
    #     schema = json.loads(schema)

    #     # Extract existing buildings information

    #     existing_buildings = schema['buildings']
    #     new_buildings = existing_buildings.copy()

    #     # Copy structure of existing buildings

    #     for i in range(len(existing_buildings), n_buildings):

    #         base_building_index = random.randint(1, len(new_buildings))
    #         generated_building = new_buildings[f'Building_{base_building_index}'].copy()

    #         # Replace building name

    #         name = f'Building_{1 + i}'
    #         generated_building['energy_simulation'] = f'Building_{1 + i}.csv'

    #         # Generate csv for the building

    #         create_building_csv(1 + i)

    #         # Randomly modify the cooling device by combinining the existing buildings linearly

    #         generated_building['cooling_device']['attributes']['nominal_power'] *= random.uniform(1.1, 1.3)
    #         generated_building['cooling_device']['attributes']['efficiency'] = random.uniform(0.25, 1)
    #         generated_building['cooling_device']['attributes']['target_cooling_temperature'] *= random.uniform(0.9, 1.1)
    #         generated_building['cooling_device']['attributes']['target_heating_temperature'] *= random.uniform(0.9, 1.1)

    #         # Randomly modify the dwh device

    #         generated_building['dhw_device']['attributes']['nominal_power'] *= random.uniform(0.9, 1.1)
    #         generated_building['dhw_device']['attributes']['efficiency'] = random.uniform(0.2, 5)

    #         # Randomly modify the dwh storage

    #         generated_building['dhw_storage']['attributes']['capacity'] *= random.uniform(0.9, 1.1)
    #         generated_building['dhw_storage']['attributes']['loss_coefficient'] *= random.uniform(0.9, 1.1)

    #         # Randomly modify electrical storage

    #         generated_building['electrical_storage']['attributes']['capacity'] *= random.uniform(0.9, 1.1)
    #         generated_building['electrical_storage']['attributes']['efficiency'] = random.uniform(0.8, 1)
    #         generated_building['electrical_storage']['attributes']['capacity_loss_coefficient'] *= random.uniform(0.9, 1.1)
    #         generated_building['electrical_storage']['attributes']['loss_coefficient'] *= random.uniform(0.9, 1.1)
    #         generated_building['electrical_storage']['attributes']['nominal_power'] *= random.uniform(0.9, 1.1)
    #         generated_building['electrical_storage']['attributes']['depth_of_discharge'] *= random.uniform(0.9, 1.1)

    #         # Randomly modify the pv generator

    #         generated_building['pv']['attributes']['nominal_power'] *= random.uniform(0.9, 1.1)

    #         # Randomly modify the power outage attributes

    #         generated_building['power_outage']["stochastic_power_outage_model"]['attributes']['random_seed'] = random.randint(0, 1000)
    #         generated_building['power_outage']["stochastic_power_outage_model"]['attributes']['saifi'] *= random.uniform(0.9, 1.1)
    #         generated_building['power_outage']["stochastic_power_outage_model"]['attributes']['caidi'] *= random.uniform(0.9, 1.1)

    #         # Replace building i

    #         new_buildings[name] = generated_building

    #     # Save the new schema

    #     schema['buildings'] = new_buildings

    #     with open(ref_schema + 'schema_generated.json', 'w') as f:
    #         json.dump(schema, f, indent=4)
