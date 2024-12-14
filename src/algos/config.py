import json
import torch

from torch import multiprocessing
from src.utils import (
    set_seed,
    REWARDS,
)
from src.utils.cl_torchrl_helper import (
    create_env,
)

ACTIVE_OBSERVATIONS = [
    'hour',
    'day_type',
    'solar_generation',
    'net_electricity_consumption',
    'electrical_storage_soc',
    'non_shiftable_load',
    'non_shiftable_load_predicted_4h',
    'non_shiftable_load_predicted_6h',
    'non_shiftable_load_predicted_12h',
    'non_shiftable_load_predicted_24h',
    'direct_solar_irradiance',
    'direct_solar_irradiance_predicted_6h',
    'direct_solar_irradiance_predicted_12h',
    'direct_solar_irradiance_predicted_24h',
    'electricity_pricing',
    'selling_pricing',
    'carbon_intensity'
]

def get_exp_envs(data_path: str = "data/naive_data/", **kwargs):
    
    # Parse kwargs

    reward = kwargs.get("reward", "cost")
    seed = kwargs.get("seed", 0)
    day_count = kwargs.get("day_count", 1)
    device = kwargs.get("device", "cpu")
    gpu_device_ix = kwargs.get("gpu_device_ix", 0)
    personal_encoding = kwargs.get("personal_encoding", False)

    is_fork = multiprocessing.get_start_method() == "fork"

    device = (
        torch.device(f"cuda:{gpu_device_ix}")
        if torch.cuda.is_available() and not is_fork and device == "cuda"
        else torch.device("mps")
        if torch.backends.mps.is_available() and device == "mps"
        else torch.device("cpu")
    )

    # Prepare configs

    set_seed(seed)
    
    # Training configurations

    schema_filepath = data_path + 'schema.json'

    with open(schema_filepath) as json_file:
        schema_dict = json.load(json_file)

    train_env_config = {
        "schema": schema_dict,
        "central_agent": False,
        "active_observations": ACTIVE_OBSERVATIONS,
        "reward_function": REWARDS[reward],
        "random_seed": seed,
        "day_count": day_count,
        "personal_encoding": personal_encoding,
    }

    train_env = create_env(train_env_config, device, seed)

    # Validation configurations

    schema_filepath = data_path + 'eval/schema.json'

    with open(schema_filepath) as json_file:
        schema_dict = json.load(json_file)

    eval_env_config = {
        **train_env_config,
        "schema": schema_dict,
    }

    eval_env = create_env(eval_env_config, device, seed)
    
    return train_env, eval_env, device
    
