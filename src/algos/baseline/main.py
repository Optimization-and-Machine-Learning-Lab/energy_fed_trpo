import json
import wandb
import argparse

from time import time
from tqdm import tqdm
from pathlib import Path
from citylearn.citylearn import CityLearnEnv

from src.utils.utils import *
from stable_baselines3 import SAC
from src.utils.cl_rewards import *
from src.utils.cl_env_helper import *

from citylearn.agents.base import BaselineAgent

ACTIVE_OBSERVATIONS = [
    'hour', 'electrical_storage_soc', 'outdoor_dry_bulb_temperature', 'outdoor_relative_humidity',
    'non_shiftable_load', 'electricity_pricing', 'carbon_intensity'
]

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
        torch.set_default_dtype(torch.float64)

    # Enable backcompat warnings

    torch.utils.backcompat.broadcast_warning.enabled = True
    torch.utils.backcompat.keepdim_warning.enabled = True

    return device

def get_env_from_config(config: dict, seed : int = None):

    seed = seed if seed is not None else random.randint(2000, 2500)

    # Fix seed to maintain simulation period among different seeds for the models

    simulation_start_time_step, simulation_end_time_step = select_simulation_period(
        dataset_name='citylearn_challenge_2022_phase_all', count=config['day_count'], seed=seed
    ) 

    return CityLearnEnv(
        schema=config["schema"],
        active_observations=config["active_observations"],
        simulation_start_time_step=simulation_start_time_step,
        simulation_end_time_step=simulation_end_time_step,
        reward_function=config["reward_function"],
        random_seed=config["random_seed"]
    )

def parse_args():

    parser = argparse.ArgumentParser(description='CityLearn TRPO experiment')

    parser.add_argument('--seed', type=int, default=0, metavar='S', help='Random seed (default: 0)')
    parser.add_argument("--n_episodes", type=int, default=1500)
    parser.add_argument('--wandb-log', default=False, action='store_true', help='Log to wandb (default: True)')
    parser.add_argument('--device', type=str, default=None, help='device (default: None)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='LI', help='Interval between training status logs (default: 1)')
    parser.add_argument('--day-count', type=int, default=1, help='Number of days for training (default: 1)')
    parser.add_argument('--data-path', type=str, default='./data/simple_data/', help='Data path (default: ./data/simple_data)')

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    # Initialize configuration

    device = init_config()

    # Parse command line arguments

    args = parse_args()

    # Override device if specified

    if args.device is not None:

        device = args.device

    # Set random seed

    set_seed(seed=args.seed)

    # Setup WandB if enabled

    run = None

    if args.wandb_log:

        run =  wandb.init(
            name=f"baseline_seed_{args.seed}", 
            project="trpo_rl_energy",
            entity="optimllab",
            config={
                "algorithm": "baseline",
                "seed": args.seed,
            },
            sync_tensorboard=False,
        )

    # Configure Environment

    schema_filepath = args.data_path + 'schema.json'

    with open(schema_filepath) as json_file:
        schema_dict = json.load(json_file)

    # Environment initialization, we create two different to not affect the episode tracker
                
    env_config = {
        "schema": schema_dict,
        "active_observations": ACTIVE_OBSERVATIONS,
        "reward_function": ElectricityCostWithPenalization,
        # "reward_function": NetElectricity,
        "random_seed": args.seed,
        "day_count": args.day_count,
    }

    # initialize environment

    eval_env = get_env_from_config(env_config, seed=2500) # This is the evaluation environment seed, it's fixed among the experiments to evaluate in the same day

    # Define a path to create logs

    logs_path = wandb.run.dir if args.wandb_log else f"./logs/baseline_t_{str(int(time()))}"
    Path(logs_path).mkdir(exist_ok=True)

    # Initialize the model
    model = BaselineAgent(eval_env)

    # Execute model in train env

    with tqdm(total=args.n_episodes, nrows=10) as pbar:

        wandb_step = 0
        building_count = len(eval_env.unwrapped.buildings)

        for i in range(args.n_episodes):
            
            wandb_step += 1     # count training step

            reward_sum = np.array([0.] * building_count)
            emission_sum = np.array([0.] * building_count)

            # Check kpis for eval environment

            observations, _ = eval_env.reset()

            while not eval_env.unwrapped.terminated:

                actions = model.predict(observations, deterministic=True)
                observations, reward, _, _, _ = eval_env.step(actions)

                reward_sum += reward
                emission_sum += np.sum([max(0, eval_env.buildings[b].net_electricity_consumption[-1]) for b in range(building_count)]) * observations[0][3]
                
            # kpis = get_kpis(eval_env)

            eval_log = {}

            # rewards = np.array(pd.DataFrame(eval_env.unwrapped.episode_rewards)['sum'].tolist())

            #District level logging

            # eval_log["eval/d_emission_avg"] = kpis[kpis['kpi'] == 'Emissions']['value'].iloc[0]
            # eval_log["eval/d_cost_avg"] = kpis[kpis['kpi'] == 'Cost']['value'].iloc[0]
            # eval_log["eval/d_daily_peak_avg"] = kpis[kpis['kpi'] == 'Avg. daily peak']['value'].iloc[0]
            # eval_log["eval/d_load_factor_avg"] = kpis[kpis['kpi'] == '1 - load factor']['value'].iloc[0]
            # eval_log["eval/d_ramping_avg"] = kpis[kpis['kpi'] == 'Ramping']['value'].iloc[0]

            # Building level sampling and logging

            for b in range(len(eval_env.unwrapped.buildings)):

                # For logging

                # base_name = f"eval/b_{b+1}_"

                # Searching by index works as after grouping the District metric goes last

                # eval_log[f"{base_name}reward_avg"] = rewards[:,b].mean()
                # eval_log[f"{base_name}emission_avg"] = kpis[kpis['kpi'] == 'Emissions']['value'].iloc[b]
                # eval_log[f"{base_name}cost_avg"] = kpis[kpis['kpi'] == 'Cost']['value'].iloc[b]

                base_name = f"eval/b_{eval_env.buildings[b].name[-1]}_"

                eval_log[f"{base_name}mean_reward"] = reward_sum[b]/24
                eval_log[f"{base_name}mean_emission"] = emission_sum[b]/24

            if i % args.log_interval == 0:

                write_log(log_path=logs_path, log={'Episode': i, **eval_log})                    
                pbar.set_postfix({'Episode': i, **eval_log})

                if args.wandb_log:

                    wandb.log({**eval_log}, step = int(wandb_step))

            # Update pbar

            pbar.update(1)