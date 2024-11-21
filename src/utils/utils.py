import math
import random
import torch
import numpy as np

from pathlib import Path
# from citylearn.citylearn import CityLearnEnv
from src.custom.citylearn import CityLearnEnv
from src.utils.cl_env_helper import select_simulation_period


def get_env_from_config(config: dict, seed : int = None):

    seed = seed if seed is not None else random.randint(2000, 2500)

    # Fix seed to maintain simulation period among different seeds for the models

    simulation_start_time_step, simulation_end_time_step = select_simulation_period(
        dataset_name='citylearn_challenge_2022_phase_all', count=config['day_count'], seed=seed
    ) 

    return CityLearnEnv(
        **config,
        simulation_start_time_step=simulation_start_time_step,
        simulation_end_time_step=simulation_end_time_step,
    )

def init_config():

    torch.set_num_threads(10)
    
    # Make sure logs folder exists

    Path("./logs").mkdir(exist_ok=True)

    # Get device

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Set numpy print options

    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

    # Set default tensor type to torch.DoubleTensor

    if device == "mps":
        torch.set_default_dtype(torch.float32)
    else:
        torch.set_default_dtype(torch.float32)
        # torch.set_default_dtype(torch.float64)

    # Enable backcompat warnings

    torch.utils.backcompat.broadcast_warning.enabled = True
    torch.utils.backcompat.keepdim_warning.enabled = True

    return device

def set_seed(seed):
    """
    Set the seed for all random number generators.

    Args:
        seed (int): The seed to be used.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    return None

def write_log(log_path, log):

    with open(log_path + '/logs.txt', 'a') as f:
        f.write(str(log) + '\n')

def normal_entropy(std):
    var = std.pow(2)
    entropy = 0.5 + 0.5 * torch.log(2 * var * math.pi)
    return entropy.sum(1, keepdim=True)

def normal_log_density(x, mean, log_std, std):
    # Precompute constants
    log2pi = math.log(2 * math.pi)
    
    # Square the std instead of calling pow(2)
    var = std * std
    
    # Vectorized and in-place operations
    log_density = -(x - mean).pow(2) / (2 * var) - 0.5 * log2pi - log_std
    
    # Return the sum along dimension 1, keeping dimension
    return log_density.sum(dim=1, keepdim=True)


def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params


def set_flat_params_to(model, flat_params):
    prev_ind = 0
    for param in model.parameters():
        flat_size = int(np.prod(list(param.size())))
        param.data.copy_(
            flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
        prev_ind += flat_size


def get_flat_grad_from(net, grad_grad=False):
    grads = []
    for param in net.parameters():
        if grad_grad:
            grads.append(param.grad.grad.view(-1))
        else:
            grads.append(param.grad.view(-1))

    flat_grad = torch.cat(grads)
    return flat_grad

