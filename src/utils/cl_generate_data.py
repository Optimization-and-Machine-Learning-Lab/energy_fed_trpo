from ast import arg
import os
import json
import random
import argparse
from weakref import ref

import numpy as np
import pandas as pd

from pathlib import Path
from citylearn.data import DataSet
from sklearn import base
from src.utils.utils import set_seed

# Bases used to generate simple data for the buildings

DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

MONTHLY_MEAN_TEMP = [
    [0, 1.9, 6.2, 11.5, 15.9, 19.8, 21.8, 21.5, 16.5, 10.9, 5.6, 1],
    [1, 4, 10, 16, 21, 25, 27, 25, 21, 15, 8, 2],
    [17, 17, 20, 23, 27, 29, 29, 29, 28, 26, 22, 18],
    [0, 0, 0, 3, 6, 9, 11, 7, 4, 3, 2, 1],
    [20, 20.5, 19, 16.4, 14.4, 12.1, 11.3, 11.9, 13.3, 14.6, 19.6, 15.65]
]

MONTHLY_MEAN_HUM = [
    [77, 70, 58, 49, 49, 50, 49, 51, 52, 62, 76, 80],
    [66, 62, 67, 68, 68, 62, 71, 75, 79, 77, 74, 69],
    [56.2, 69.9, 66.5, 68.2, 69.7, 70, 67.8, 67.5, 63.5, 57, 57.2, 52.8],
    [74.4, 72, 72.6, 69.1, 70, 72, 76.3, 76.8, 75, 75.8, 75.7, 74.6],
    [78, 80, 82, 83, 86, 86, 86, 83, 82, 77, 76, 77]
]

LOAD_DAY_CHANGE = [
    [0.5, -0.2, -0.2, -0.2, -0.2, -0.2, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1.5, 1.5, 1], # B1: Programmer
    [0.5, 1, 1, 2, 1, 0, -0.2, -0.3, -0.3, -0.3, -0.4, -0.3, -0.3, 3, 2, 2, 3, 2, 1, 2, 1, 1, 1, 1], # B2: Work from home at night
    [2, 2, 2, 2, 2, 2, 2, 2, 3, 3.5, 1.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 2, 3, 3, 2, 2, 2, 2], # B3: College student
    [-0.3, -0.3, -0.3, -0.3, -0.3, -0.3, 2, 0, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, -0.2, 2, 3, 3, 2, 1, -0.2], # B4: Civil servant
    [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 2, 3, 3, 3, 3, 2, 0, 0, 1, 1] # B5: Rich second generation
]

SOLAR_DAY_CHANGE = [
    [0, 0, 0, 0, 0, 0, -280, -200, -40, 150, 300, 325, 350, 350, 300, 250, 100, -150, -250, -280, 0, 0, 0, 0], #B1
    [0, 0, 0, 0, 0, 0, -140, -100, 0, 120, 210, 230, 249, 240, 103, 98, 12, -123, -140, -150, 0, 0, 0, 0], # B2
    [0, 0, 0, 0, 0, 0, -290, -210, -100, 50, 150, 210, 300, 320, 300, 310, 200, 50, -120, -250, 0, 0, 0, 0], # B3
    [0, 0, 0, 0, 0, 0, -300, -300, -300, -250, -150, 0, 129, 150, 92, 0, -170, -300, -300, -300, 0, 0, 0, 0], # B4
    [0, 0, 0, 0, 0, 0, -120, -50, 120, 210, 270, 320, 350, 380, 320, 280, 130, -50, -100, -130, 0, 0, 0, 0] # B5
]

WEATHER_RANDOMNESS = {
    "training": {
        "temperature": {"mu": 0, "sigma": 5},
        "humidity": {"mu": 0, "sigma": 5},
    },
    "eval": {
        "temperature": {"mu": 0, "sigma": 5},
        "humidity": {"mu": 0, "sigma": 5},
    }
}

OTHER_PARAMETERS = {
    "ac_eff": [1, 1.2, 2.1, 0.9, 1.2],
    "solar_eff": [1, 1.2, 0.5, 0.8, 1.2],
    "solar_escale": [0.6, 0.5, 0.5, 0.9, 0.5],
    "solar_incercept": [350, 368, 320, 290, 400],
}

# Useful function definitions

def shift_and_append(df, n):

    # Shift the DataFrame by n positions (downward)
    shifted_df = df.shift(n)
    
    # Collect the dropped values (the last n rows before the shift)
    dropped_values = df.iloc[-n:].reset_index(drop=True)
    
    # Replace the first n rows (now NaN) with the dropped values
    shifted_df.iloc[:n] = dropped_values.values
    
    return shifted_df

def generate_day_temperature(month, month_day, b):

    if month_day < 15:
        last_month = month - 1
        return MONTHLY_MEAN_TEMP[b][month] + (MONTHLY_MEAN_TEMP[b][last_month] - MONTHLY_MEAN_TEMP[b][month]) * (15 - month_day) / 30
    else:
        next_month = (month + 1) % 12
        return MONTHLY_MEAN_TEMP[b][month] + (MONTHLY_MEAN_TEMP[b][next_month] - MONTHLY_MEAN_TEMP[b][month]) * (month_day - 15) / 30
    
def generate_day_humidity(month, month_day, b):

    if month_day < 15:
        last_month = month - 1
        return MONTHLY_MEAN_HUM[b][month] + (MONTHLY_MEAN_HUM[b][month] - MONTHLY_MEAN_HUM[b][last_month]) * (15 - month_day) / 30
    else:
        next_month = (month + 1) % 12
        return MONTHLY_MEAN_HUM[b][month] + (MONTHLY_MEAN_HUM[b][next_month] - MONTHLY_MEAN_HUM[b][month]) * (month_day - 15) / 30

def generate_hour_load(efficiency, weather_data):      # extra power for ac and heating. efficiency \in [2, 5]

    temp = weather_data[0]
    hum = weather_data[1]
    
    # Rule base generation of load based on temperature and humidity

    if hum > 60:
        hum_load = (hum - 60) / 20 / efficiency
    else:
        hum_load = hum / 20 / efficiency

    if temp > 25:
        temp_load = temp / 25 / efficiency
    elif temp < 15:
        temp_load = (30 - temp) / 25 / efficiency
    else:
        temp_load = 0
    
    return hum_load + temp_load

def get_hour_solar(temp, intercept, efficiency, panel_scale):

    return (intercept + temp * efficiency) * panel_scale

def generate_simplified_data(base_dataset: str = 'citylearn_challenge_2022_phase_all', dest_folder: str = 'data/simple_data/'):

    # Make sure the destination folder exists including the subfolders

    Path(dest_folder).mkdir(parents=True, exist_ok=True)

    # Get reference schema

    schema = DataSet.get_schema(base_dataset)

    # Reduce the number of buildings to 5

    schema['buildings'] = {f'Building_{i}': schema['buildings'][f'Building_{i}'] for i in range(1, 6)}

    # Extract the base weather, emissions and pricing data

    base_weather = pd.read_csv(os.path.join(schema['root_directory'], schema['buildings']['Building_1']['weather']))
    base_pricing = pd.read_csv(os.path.join(schema['root_directory'], schema['buildings']['Building_1']['pricing']))
    base_emissions = pd.read_csv(os.path.join(schema['root_directory'], schema['buildings']['Building_1']['carbon_intensity']))

    # Read the base schema and process the json

    for building_no, (building_name, info) in enumerate(schema['buildings'].items()):

        # Create custom building CSVs

        base_csv = pd.read_csv(os.path.join(schema['root_directory'], info['energy_simulation']))

        sigma_load_hourly = 0.01
        sigma_solar_hourly = 5
        sigma_hourly_temp = 0.01
        sigma_hourly_hum = 1
        temp_day_change = 4
        hum_day_change = 12

        weather_data = [[0.] * 2 for _ in range(365 * 24)] # We need just to store the temperature and humidity
        building_data = [[0.] * 4 for _ in range(365 * 24)] # We need just to store the month, hour, load and generation

        for day in range(365):

            month = day // 31
            month_day = day % 31

            base_temp = generate_day_temperature(month=month, month_day=month_day, b=building_no)
            base_hum = generate_day_humidity(month=month, month_day=month_day, b=building_no)

            for hour in range(24):

                if hour < 12:
                    weather_data[day * 24 + hour][0] = random.gauss(base_temp + (temp_day_change / 12 * hour - temp_day_change / 2), sigma_hourly_temp)
                    weather_data[day * 24 + hour][1] = random.gauss(base_hum + (hum_day_change / 2 - hum_day_change / 12 * (hour - 12)), sigma_hourly_hum)
                else:
                    weather_data[day * 24 + hour][0] = random.gauss(base_temp + (temp_day_change / 2 - temp_day_change / 12 * (hour - 12)), sigma_hourly_temp)
                    weather_data[day * 24 + hour][1] = random.gauss(base_hum + (hum_day_change / 12 * hour - hum_day_change / 2), sigma_hourly_hum)

                # Generate simpler data for non_shiftable_load based on humidity and temperature 
        
                building_data[day * 24 + hour][0] = month + 1
                building_data[day * 24 + hour][1] = hour if hour != 0 else 24

                base_load = generate_hour_load(OTHER_PARAMETERS['ac_eff'][building_no], weather_data[day * 24 + hour])
                base_solar = get_hour_solar(
                    weather_data[day * 24 + hour][0], OTHER_PARAMETERS['solar_incercept'][building_no], OTHER_PARAMETERS['solar_eff'][building_no],
                    OTHER_PARAMETERS['solar_escale'][building_no]
                )

                building_data[day * 24 + hour][2] = random.gauss(base_load + LOAD_DAY_CHANGE[building_no][hour], sigma_load_hourly)

                if hour < 6 or hour > 19:
                    building_data[day * 24 + hour][3] = 0
                else:
                    building_data[day * 24 + hour][3] = random.gauss(base_solar + SOLAR_DAY_CHANGE[building_no][hour], sigma_solar_hourly)

        # Update the base csv with the new data

        building_data = np.array(building_data).clip(min=0)
        weather_data = np.array(weather_data)

        base_csv['month'] = building_data[:,0].astype(np.int32)
        base_csv['hour'] = building_data[:,1].astype(np.int32)
        base_csv['non_shiftable_load'] = building_data[:,2]
        base_csv['solar_generation'] = building_data[:,3]

        # Update the weather data for this building. 

        base_weather['outdoor_dry_bulb_temperature'] = weather_data[:,0]
        base_weather['outdoor_relative_humidity'] = weather_data[:,1]

        # Save the new CSVs to the destination folder

        base_csv.to_csv(os.path.join(dest_folder, f'{building_name}.csv'), index=False)
        base_weather.to_csv(os.path.join(dest_folder, f'weather_{building_name[-1]}.csv'), index=False)

        # Update the schema with the new paths

        schema['buildings'][building_name]['energy_simulation'] = f'{building_name}.csv'
        schema['buildings'][building_name]['weather'] = f'weather_{building_name[-1]}.csv'

    # Write pricing and emissions data to the destination folder

    base_pricing.to_csv(os.path.join(dest_folder, 'pricing.csv'), index=False)
    base_emissions.to_csv(os.path.join(dest_folder, 'carbon_intensity.csv'), index=False)

    # Save the new schema in the destination folder

    schema['root_directory'] = dest_folder

    with open(os.path.join(dest_folder, 'schema.json'), 'w') as f:
        json.dump(schema, f, indent=4)

def get_perturbed_data(source_folder: str = 'data/simple_data/', type: str = 'training'):

    # Get current schema

    schema = json.loads(open(os.path.join(source_folder, 'schema.json')).read())

    # Define object to return the updated data

    new_data = {
        'outdoor_dry_bulb_temperature': [],
        'outdoor_relative_humidity': [],
        'non_shiftable_load': [],
        'solar_generation': []
    }

    # Read the base schema and process the json

    for building_no, (building_name, info) in enumerate(schema['buildings'].items()):

        # Create custom building CSVs

        base_csv = pd.read_csv(os.path.join(schema['root_directory'], info['energy_simulation']))
        base_weather = pd.read_csv(os.path.join(schema['root_directory'], info['weather']))

        data_length = len(base_csv)
    
        # Compute randomness for the weather data (uniform sample between high and lower bounds)

        weather_randomness = WEATHER_RANDOMNESS[type]

        # delta_temp = np.random.uniform(weather_randomness['temperature']['low'], weather_randomness['temperature']['high'], data_length)
        # delta_hum = np.random.uniform(weather_randomness['humidity']['low'], weather_randomness['humidity']['high'], data_length)

        delta_temp = np.random.normal(weather_randomness['temperature']['mu'], weather_randomness['temperature']['sigma'], data_length)
        delta_hum = np.random.normal(weather_randomness['humidity']['mu'], weather_randomness['humidity']['sigma'], data_length)

        # Update temperature and humidity data

        base_temp = base_weather['outdoor_dry_bulb_temperature'].to_numpy()
        base_hum = base_weather['outdoor_relative_humidity'].to_numpy()

        temperature = base_temp + delta_temp
        humidity = base_hum + delta_hum

        # Compute the new solar generation and non-shiftable load

        base_load = base_csv['non_shiftable_load'].to_numpy()
        base_solar = base_csv['solar_generation'].to_numpy()

        non_shiftable_load = (humidity - 60) / 20 / OTHER_PARAMETERS['ac_eff'][building_no] + (temperature - 25) / 25 / OTHER_PARAMETERS['ac_eff'][building_no] + base_load
        solar_generation = (temperature * OTHER_PARAMETERS['solar_eff'][building_no] * OTHER_PARAMETERS['solar_escale'][building_no] ** 2) * base_solar
        
        # Add to dictionary

        new_data['outdoor_dry_bulb_temperature'].append(temperature)
        new_data['outdoor_relative_humidity'].append(humidity)
        new_data['non_shiftable_load'].append(non_shiftable_load)
        new_data['solar_generation'].append(solar_generation)

    # Return new data

    return new_data

def copy_data_from_ref(ref_dataset: str = 'citylearn_challenge_2022_phase_all', dest_folder: str = 'data/ref_data/'):

    # Make sure the destination folder exists including the subfolders

    Path(dest_folder).mkdir(parents=True, exist_ok=True)

    # Get reference schema

    schema = DataSet.get_schema(ref_dataset)

    # Copy the base weather, emissions and pricing data

    base_weather = pd.read_csv(os.path.join(schema['root_directory'], schema['buildings']['Building_1']['weather']))
    base_pricing = pd.read_csv(os.path.join(schema['root_directory'], schema['buildings']['Building_1']['pricing']))
    base_emissions = pd.read_csv(os.path.join(schema['root_directory'], schema['buildings']['Building_1']['carbon_intensity']))

    base_weather.to_csv(os.path.join(dest_folder, 'weather.csv'), index=False)
    base_pricing.to_csv(os.path.join(dest_folder, 'pricing.csv'), index=False)
    base_emissions.to_csv(os.path.join(dest_folder, 'carbon_intensity.csv'), index=False)

    # Copy the base building data

    for _, (building_name, info) in enumerate(schema['buildings'].items()):

        base_csv = pd.read_csv(os.path.join(schema['root_directory'], info['energy_simulation']))

        base_csv.to_csv(os.path.join(dest_folder, f'{building_name}.csv'), index=False)

    # Copy the base schema to the destination folder

    with open(os.path.join(schema['root_directory'], 'schema.json'), 'r') as f:

        schema = json.load(f)

        schema['root_directory'] = dest_folder

        with open(os.path.join(dest_folder, 'schema.json'), 'w') as f:

            json.dump(schema, f, indent=4)

def generate_simple_data_from_ref(base_dataset: str = 'citylearn_challenge_2022_phase_all', dest_folder: str = 'data/simple_data/', shift: bool = False):

    # Make sure the destination folder exists including the subfolders

    Path(dest_folder).mkdir(parents=True, exist_ok=True)

    # Get reference schema

    schema = DataSet.get_schema(base_dataset)

    # Reduce the number of buildings to 5

    schema['buildings'] = {f'Building_{i}': schema['buildings'][f'Building_{i}'] for i in range(1, 6)}

    # Extract the base weather, emissions and pricing data (doesn't change among buildings)

    base_weather = pd.read_csv(os.path.join(schema['root_directory'], schema['buildings']['Building_1']['weather']))
    base_pricing = pd.read_csv(os.path.join(schema['root_directory'], schema['buildings']['Building_1']['pricing']))
    base_emissions = pd.read_csv(os.path.join(schema['root_directory'], schema['buildings']['Building_1']['carbon_intensity']))

    # Read the base schema and process the json

    for building_no, (building_name, info) in enumerate(schema['buildings'].items()):

        # Create custom building CSVs

        base_csv = pd.read_csv(os.path.join(schema['root_directory'], info['energy_simulation']))

        sigma_load_hourly = 0.01
        sigma_solar_hourly = 50

        building_data = [[0.] * 2 for _ in range(365 * 24)] # We need just to store the load and generation

        for day in range(365):

            init_step = day * 24
            end_step = day * 24 + 24

            base_temp = base_weather['outdoor_dry_bulb_temperature'].iloc[init_step:end_step]
            base_hum = base_weather['outdoor_relative_humidity'][init_step:end_step]

            for hour in range(24):

                curr_hour = init_step + hour

                # Generate simpler data for non_shiftable_load based on humidity and temperature 

                base_load = generate_hour_load(OTHER_PARAMETERS['ac_eff'][building_no], (base_temp[curr_hour], base_hum[curr_hour]))
                base_solar = get_hour_solar(
                    base_temp[curr_hour], OTHER_PARAMETERS['solar_incercept'][building_no], OTHER_PARAMETERS['solar_eff'][building_no],
                    OTHER_PARAMETERS['solar_escale'][building_no]
                )

                building_data[curr_hour][0] = random.gauss(base_load + LOAD_DAY_CHANGE[building_no][hour], sigma_load_hourly)

                if hour < 6 or hour > 19:
                    building_data[curr_hour][1] = 0
                else:
                    building_data[curr_hour][1] = random.gauss(base_solar + SOLAR_DAY_CHANGE[building_no][hour], sigma_solar_hourly)

        # Update the base csv with the new data

        building_data = np.array(building_data).clip(min=0)

        base_csv['non_shiftable_load'] = building_data[:,0]
        base_csv['solar_generation'] = building_data[:,1]

        # Shift the data if needed, it's done randomly for each building. by months

        shift_hours = 0 if not shift else sum(DAYS_IN_MONTH[:random.randint(1, 12)] * 24)

        # Save the final CSVs (building and weather) to the destination folder

        shift_and_append(base_csv, shift_hours).to_csv(os.path.join(dest_folder, f'{building_name}.csv'), index=False)
        shift_and_append(base_weather, shift_hours).to_csv(os.path.join(dest_folder, f'weather_{building_name[-1]}.csv'), index=False)

        # Update the schema with the new paths

        schema['buildings'][building_name]['energy_simulation'] = f'{building_name}.csv'
        schema['buildings'][building_name]['weather'] = f'weather_{building_name[-1]}.csv'

    # Write pricing and emissions data to the destination folder

    base_pricing.to_csv(os.path.join(dest_folder, 'pricing.csv'), index=False)
    base_emissions.to_csv(os.path.join(dest_folder, 'carbon_intensity.csv'), index=False)

    # Save the new schema in the destination folder

    schema['root_directory'] = dest_folder

    with open(os.path.join(dest_folder, 'schema.json'), 'w') as f:
        json.dump(schema, f, indent=4)

def generate_simplest_data_from_ref(base_dataset: str = 'citylearn_challenge_2022_phase_all', dest_folder: str = 'data/simplest_data/'):

    # Make sure the destination folder exists including the subfolders

    Path(dest_folder).mkdir(parents=True, exist_ok=True)

    # Get reference schema

    schema = DataSet.get_schema(base_dataset)

    # Reduce the number of buildings to 1

    schema['buildings'] = {f'Building_{i}': schema['buildings'][f'Building_{i}'] for i in range(1, 2)}

    # Extract the base weather, emissions and pricing data (doesn't change among buildings)

    base_weather = pd.read_csv(os.path.join(schema['root_directory'], schema['buildings']['Building_1']['weather']))
    base_pricing = pd.read_csv(os.path.join(schema['root_directory'], schema['buildings']['Building_1']['pricing']))
    base_emissions = pd.read_csv(os.path.join(schema['root_directory'], schema['buildings']['Building_1']['carbon_intensity']))

    # Read the base schema and process the json

    # We need just to store the load and generation
    # 0 - Solar generation
    # 1 - Load

    building_data = np.array([[[0.] * 2 for _ in range(365 * 24)] for _ in range(1)])
    base_csvs = []

    for building_no, (building_name, info) in enumerate(schema['buildings'].items()):

        # Create custom building CSVs

        base_csv = pd.read_csv(os.path.join(schema['root_directory'], info['energy_simulation']))

        base_csvs.append(base_csv)

        constant_solar = 1000 # In kWh

        for day in range(365):

            init_step = day * 24
            end_step = day * 24 + 24

            for hour in range(24):

                curr_hour = init_step + hour

                # Generate simpler solar data

                if hour < 6 or hour > 18:
                    building_data[building_no,curr_hour,0] = 0
                else:
                    building_data[building_no,curr_hour,0] = constant_solar

                # Generate simple load following the rule

                building_data[building_no,curr_hour,1] += generate_hour_load(curr_hour, constant_solar)

    # Update the base csv with the new data

    building_data = np.array(building_data).clip(min=0)

    for building_no, (building_name, _) in enumerate(schema['buildings'].items()):

        base_csvs[building_no]['solar_generation'] = building_data[building_no,:,0]
        base_csvs[building_no]['non_shiftable_load'] = building_data[building_no,:,1] / 1000 # kWh

        base_csvs[building_no].to_csv(os.path.join(dest_folder, f'{building_name}.csv'), index=False)

        # Update the schema with the new paths

        schema['buildings'][building_name]['energy_simulation'] = f'{building_name}.csv'

        # Update the schema to guarantee no degradation in the battery efficiency

        schema['buildings'][building_name]['electrical_storage']['attributes'] = {
            "capacity": 6.4,
            "efficiency": 1,
            "capacity_loss_coefficient": 0.0,
            "loss_coefficient": 0.0,
            "nominal_power": 5.0,
            "power_efficiency_curve": [[1, 1],[0, 1],[1, 1],[0, 1],[1, 1]]
        }

        # Set nominal power to 1 for solar panel to simplify case

        schema['buildings'][building_name]['pv']['attributes']['nominal_power'] = 1.0

    # Write pricing and emissions data to the destination folder

    base_weather.to_csv(os.path.join(dest_folder, 'weather.csv'), index=False)
    base_pricing.to_csv(os.path.join(dest_folder, 'pricing.csv'), index=False)
    base_emissions.to_csv(os.path.join(dest_folder, 'carbon_intensity.csv'), index=False)

    # Save the new schema in the destination folder

    schema['root_directory'] = dest_folder

    with open(os.path.join(dest_folder, 'schema.json'), 'w') as f:
        json.dump(schema, f, indent=4)

def parse_args():

    parser = argparse.ArgumentParser(description='Generate simple data for the CityLearn Challenge')

    parser.add_argument('--base_dataset', type=str, default='citylearn_challenge_2022_phase_all', help='Base dataset to generate simple data from')
    parser.add_argument('--dest_folder', type=str, help='Destination folder to save the simple data')
    parser.add_argument('--shift', type=bool, default=False, help='Whether to shift the data or not')
    parser.add_argument('--function', type=str, default='generate_simple_data_from_ref', help='Function to call to generate the simple data')
    parser.add_argument('--seed', type=int, default=0, help='Seed to set for the random data generation')

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    set_seed(args.seed)

    if args.function == 'generate_simple_data_from_ref':

        generate_simple_data_from_ref(base_dataset=args.base_dataset, dest_folder=args.dest_folder, shift=args.shift)

    elif args.function == 'generate_simplest_data_from_ref':
            
        generate_simplest_data_from_ref(base_dataset=args.base_dataset, dest_folder=args.dest_folder)