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

from citylearn.agents.rbc import HourRBC

ACTION_MAP = {
    1: 1/12, # Rule for 1 AM
    2: 1/12, # Rule for 2 AM
    3: 1/12, # Rule for 3 AM
    4: 1/12, # Rule for 4 AM
    5: 1/12, # Rule for 5 AM
    6: 1/12, # Rule for 6 AM
    7: 1/12, # Rule for 7 AM
    8: 1/12, # Rule for 8 AM
    9: 1/12, # Rule for 9 AM
    10: 1/12, # Rule for 10 AM
    11: 1/12, # Rule for 11 AM
    12: 1/12, # Rule for 12 PM
    13: -1/12, # Rule for 1 PM
    14: -1/12, # Rule for 2 PM
    15: -1/12, # Rule for 3 PM
    16: -1/12, # Rule for 4 PM
    17: -1/12, # Rule for 5 PM
    18: -1/12, # Rule for 6 PM
    19: -1/12, # Rule for 7 PM
    20: -1/12, # Rule for 8 PM
    21: -1/12, # Rule for 9 PM
    22: -1/12, # Rule for 10 PM
    23: -1/12, # Rule for 11 PM
    24: -1/12, # Rule for 12 AM
}

def init_config():

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

def parse_args():

    parser = argparse.ArgumentParser(description='CityLearn TRPO experiment')

    parser.add_argument("--n_episodes", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--wandb-log', default=False, action='store_true', help='Log to wandb (default: True)')
    parser.add_argument('--device', type=str, default=None, help='device (default: None)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='LI', help='Interval between training status logs (default: 1)')

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

    algorithm_config = {
        "action_map": {
            **ACTION_MAP
        }
    }

    # Setup WandB if enabled

    run = None

    if args.wandb_log:

        run =  wandb.init(
            name=f"rbc_seed_{args.seed}_{str(int(time()))}", 
            project="trpo_rl_energy",
            entity="optimllab",
            config={
                "algorithm": "rbc",
                "seed": args.seed,
                **algorithm_config
            },
            sync_tensorboard=False,
        )

    # Initialize the environment
    # Configure Environment

    DATASET_NAME = 'citylearn_challenge_2022_phase_all'
    BUILDING_COUNT = 2
    DAY_COUNT = 1
    
    ACTIVE_OBSERVATIONS = ['hour']

    # select buildings
    buildings = select_buildings(DATASET_NAME, BUILDING_COUNT, 0) # Fix seed to maintain building selection among different seeds for the models

    # select days
    simulation_start_time_step, simulation_end_time_step = select_simulation_period(DATASET_NAME, DAY_COUNT, 0) # Fix seed to maintain simulation period among different seeds for the models

    # initialize environment

    train_env = CityLearnEnv(
        DATASET_NAME,
        central_agent=True,
        buildings=buildings,
        active_observations=ACTIVE_OBSERVATIONS,
        simulation_start_time_step=simulation_start_time_step,
        simulation_end_time_step=simulation_end_time_step,
        reward_function=NetElectricity
    )

    # select buildings
    buildings = select_buildings(DATASET_NAME, BUILDING_COUNT, 1)  # Fix seed to maintain building selection among different seeds for the models

    # select days
    simulation_start_time_step, simulation_end_time_step = select_simulation_period(DATASET_NAME, DAY_COUNT, 1) # Fix seed to maintain simulation period among different seeds for the models

    eval_env = CityLearnEnv(
        DATASET_NAME,
        central_agent=True,
        buildings=buildings,
        active_observations=ACTIVE_OBSERVATIONS,
        simulation_start_time_step=simulation_start_time_step,
        simulation_end_time_step=simulation_end_time_step,
        reward_function=NetElectricity
    )

    # Define a path to create logs

    logs_path = wandb.run.dir if args.wandb_log else f"./logs/sac_seed_{args.seed}_t_{str(int(time()))}"
    Path(logs_path).mkdir(exist_ok=True)

    # Initialize the model
    model = HourRBC(train_env, action_map=ACTION_MAP)

    # Execute model in train env

    with tqdm(total=args.n_episodes, nrows=10) as pbar:

        wandb_step = 0

        for i in tqdm(range(args.n_episodes)):

            observations, _ = train_env.reset()

            while not train_env.terminated:

                actions = model.predict(observations)
                observations, _, _, _, _ = train_env.step(actions)

            kpis = get_kpis(train_env)

            # Log metrics for train environment

            wandb_step += 1     # count training step

            train_log = {}

            rewards = np.array(pd.DataFrame(train_env.unwrapped.episode_rewards)['sum'].tolist())

            #District level logging

            train_log["train/d_emission_avg"] = kpis[kpis['kpi'] == 'Emissions']['value'].iloc[0]
            train_log["train/d_cost_avg"] = kpis[kpis['kpi'] == 'Cost']['value'].iloc[0]
            train_log["train/d_daily_peak_avg"] = kpis[kpis['kpi'] == 'Avg. daily peak']['value'].iloc[0]
            train_log["train/d_load_factor_avg"] = kpis[kpis['kpi'] == '1 - load factor']['value'].iloc[0]
            train_log["train/d_ramping_avg"] = kpis[kpis['kpi'] == 'Ramping']['value'].iloc[0]

            # Building level sampling and logging

            for b in range(len(train_env.unwrapped.buildings)):

                # For logging

                base_name = f"train/b_{b+1}_"

                # Searching by index works as after grouping the District metric goes last

                train_log[f"{base_name}reward_avg"] = rewards[:,b].mean()
                train_log[f"{base_name}emission_avg"] = kpis[kpis['kpi'] == 'Emissions']['value'].iloc[b]
                train_log[f"{base_name}cost_avg"] = kpis[kpis['kpi'] == 'Cost']['value'].iloc[b]

            # Check kpis for eval environment

            observations, _ = eval_env.reset()
            actions_list = []

            while not eval_env.unwrapped.terminated:

                actions = model.predict(observations, deterministic=True)
                observations, _, _, _, _ = eval_env.step(actions)

            kpis = get_kpis(eval_env)

            eval_log = {}

            rewards = np.array(pd.DataFrame(eval_env.unwrapped.episode_rewards)['sum'].tolist())

            #District level logging

            eval_log["eval/d_emission_avg"] = kpis[kpis['kpi'] == 'Emissions']['value'].iloc[0]
            eval_log["eval/d_cost_avg"] = kpis[kpis['kpi'] == 'Cost']['value'].iloc[0]
            eval_log["eval/d_daily_peak_avg"] = kpis[kpis['kpi'] == 'Avg. daily peak']['value'].iloc[0]
            eval_log["eval/d_load_factor_avg"] = kpis[kpis['kpi'] == '1 - load factor']['value'].iloc[0]
            eval_log["eval/d_ramping_avg"] = kpis[kpis['kpi'] == 'Ramping']['value'].iloc[0]

            # Building level sampling and logging

            for b in range(len(eval_env.unwrapped.buildings)):

                # For logging

                base_name = f"eval/b_{b+1}_"

                # Searching by index works as after grouping the District metric goes last

                eval_log[f"{base_name}reward_avg"] = rewards[:,b].mean()
                eval_log[f"{base_name}emission_avg"] = kpis[kpis['kpi'] == 'Emissions']['value'].iloc[b]
                eval_log[f"{base_name}cost_avg"] = kpis[kpis['kpi'] == 'Cost']['value'].iloc[b]

            if i % args.log_interval == 0:

                write_log(log_path=logs_path, log={'Episode': i, **train_log, **eval_log})                    
                pbar.set_postfix({'Episode': i, **train_log, **eval_log})

                if args.wandb_log:

                    wandb.log({**train_log, **eval_log}, step = int(wandb_step))

            # Update pbar

            pbar.update(1)