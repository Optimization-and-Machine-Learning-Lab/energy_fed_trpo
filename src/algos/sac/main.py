import json
from math import e
import wandb
import argparse

from time import time 
from tqdm import tqdm
from pathlib import Path
from citylearn.citylearn import CityLearnEnv

from stable_baselines3.common.noise import NormalActionNoise
from src.utils.utils import *
from stable_baselines3 import SAC
from src.utils.cl_rewards import *
from src.utils.cl_env_helper import *

from citylearn.wrappers import (
    NormalizedObservationWrapper,
    StableBaselines3Wrapper,
)

ACTIVE_OBSERVATIONS = [
    'hour', 'electrical_storage_soc', 'outdoor_dry_bulb_temperature', 'outdoor_relative_humidity',
    'non_shiftable_load', 'electricity_pricing', 'carbon_intensity'
]

REWARDS = {
    'cost': Cost,
    'weighted_cost_emissions': WeightedCostAndEmissions,
    'cost_pen_no_batt': CostIneffectiveActionPenalization,
    'cost_pen_bad_batt': CostBadBattUsePenalization,
    'cost_pen_bad_action': CostIneffectiveActionPenalization,
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
        random_seed=config["random_seed"],
        central_agent=config["central_agent"],
    )

def parse_args():

    parser = argparse.ArgumentParser(description='CityLearn TRPO experiment')

    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--qf_nns", type=int, default=256)
    parser.add_argument("--pi_nns", type=int, default=256)
    parser.add_argument("--buffer_size", type=int, default=int(2e6))
    parser.add_argument('--wandb-log', default=False, action='store_true', help='Log to wandb (default: True)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='LI', help='Interval between training status logs (default: 1)')
    parser.add_argument('--device', type=str, default=None, help='device (default: None)')
    parser.add_argument('--data-path', type=str, default='./data/simple_data/', help='Data path (default: ./data/simple_data)')
    parser.add_argument('--reward', type=str, help='Reward function (default: cost_pen_no_batt)')
    parser.add_argument('--day-count', type=int, default=1, help='Number of days for training (default: 1)')
    parser.add_argument('--noisy_training', default=False, action='store_true', help='Noisy training (default: False)')
    
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

    model_config = {
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        # "learning_starts": 1,
        "train_freq": (6, 'step'),
        "gamma": args.gamma,
        "tau": args.tau,
        "buffer_size": args.buffer_size,
        "device": device,
        "seed": args.seed,
        # "policy_kwargs": {
        #     "net_arch": {
        #         "pi": [args.pi_nns, args.pi_nns],
        #         "qf": [args.qf_nns, args.qf_nns],
        #     }
        # },
        "action_noise": NormalActionNoise(mean=np.zeros(5), sigma=0.1 * np.ones(5))
    }

    # Setup WandB if enabled

    run = None

    if args.wandb_log:

        run =  wandb.init(
            name=f"sac_seed_{args.seed}_{str(int(time()))}", 
            project="trpo_rl_energy",
            entity="optimllab",
            config={
                "algorithm": "sac",
                "seed": args.seed,
                **model_config
            },
            sync_tensorboard=False,
        )

    # Initialize the environment
    
    # Configure Environment

    schema_filepath = args.data_path + 'schema.json'

    with open(schema_filepath) as json_file:
        schema_dict = json.load(json_file)

    # Environment initialization, we create two different to not affect the episode tracker
                
    env_config = {
        "schema": schema_dict,
        "central_agent": True,
        "active_observations": ACTIVE_OBSERVATIONS,
        "reward_function": REWARDS[args.reward],
        "random_seed": args.seed,
        "day_count": args.day_count,
    }

    # initialize environment

    train_env = get_env_from_config(config=env_config, seed=0)
    train_env = NormalizedObservationWrapper(train_env)
    train_env = StableBaselines3Wrapper(train_env)

    eval_env = get_env_from_config(config=env_config, seed=2500)
    eval_env = NormalizedObservationWrapper(eval_env)
    eval_env = StableBaselines3Wrapper(eval_env)

    # Define a path to create logs

    logs_path = wandb.run.dir if args.wandb_log else f"./logs/sac_seed_{args.seed}_t_{str(int(time()))}"
    Path(logs_path).mkdir(exist_ok=True)

    # Initialize the model
    model = SAC("MlpPolicy", train_env, **model_config)

    with tqdm(total=args.n_episodes, nrows=10) as pbar:

        wandb_step = 0

        for i in range(args.n_episodes):

            if args.noisy_training and i > 0:
                
                train_env = get_env_from_config(config=env_config, seed=i)
                train_env = NormalizedObservationWrapper(train_env)
                train_env = StableBaselines3Wrapper(train_env)

                model = SAC("MlpPolicy", train_env, **model_config)
                model.load("./models/tmp/sac", env=train_env)

            # Train the model
            model.learn(
                total_timesteps=(train_env.unwrapped.time_steps - 1), # * 5,
            )

            if args.noisy_training:

                model.save(f"./models/tmp/sac")

            # kpis = get_kpis(train_env)

            # Check kpis for training environment

            observations, _ = train_env.reset()

            while not train_env.unwrapped.terminated:

                actions, _ = model.predict(observations, deterministic=True)
                observations, _, _, _, _ = train_env.step(actions)
                

            reward_sum = train_env.unwrapped.episode_rewards[-1]['sum']
            cost_sum = [sum(train_env.unwrapped.buildings[b].net_electricity_consumption_cost) for b in range(len(train_env.unwrapped.buildings))]
            emission_sum = [sum(train_env.unwrapped.buildings[b].net_electricity_consumption_emission) for b in range(len(train_env.unwrapped.buildings))]
           
            train_log = {}

            wandb_step += 1     # count training step

            #District level logging

            # train_log["train/d_emission_avg"] = kpis[kpis['kpi'] == 'Emissions']['value'].iloc[0]
            # train_log["train/d_cost_avg"] = kpis[kpis['kpi'] == 'Cost']['value'].iloc[0]
            # train_log["train/d_daily_peak_avg"] = kpis[kpis['kpi'] == 'Avg. daily peak']['value'].iloc[0]
            # train_log["train/d_load_factor_avg"] = kpis[kpis['kpi'] == '1 - load factor']['value'].iloc[0]
            # train_log["train/d_ramping_avg"] = kpis[kpis['kpi'] == 'Ramping']['value'].iloc[0]

            # Building level sampling and logging

            for b in range(len(train_env.unwrapped.buildings)):

                # For logging

                base_name = f"train/b_{eval_env.unwrapped.buildings[b].name[-1]}_"

                train_log[f"{base_name}mean_reward"] = reward_sum[b]/24
                train_log[f"{base_name}mean_cost"] = cost_sum[b]/24
                train_log[f"{base_name}mean_emission"] = emission_sum[b]/24

            # Check kpis for eval environment

            observations, _ = eval_env.reset()

            while not eval_env.unwrapped.terminated:

                actions, _ = model.predict(observations, deterministic=True)
                observations, _, _, _, _ = eval_env.step(actions)
                
            eval_log = {}

            reward_sum = eval_env.unwrapped.episode_rewards[-1]['sum']
            cost_sum = [sum(eval_env.unwrapped.buildings[b].net_electricity_consumption_cost) for b in range(len(eval_env.unwrapped.buildings))]
            emission_sum = [sum(eval_env.unwrapped.buildings[b].net_electricity_consumption_emission) for b in range(len(eval_env.unwrapped.buildings))]

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

                base_name = f"eval/b_{eval_env.unwrapped.buildings[b].name[-1]}_"

                eval_log[f"{base_name}mean_reward"] = reward_sum[b]/24
                eval_log[f"{base_name}mean_cost"] = cost_sum[b]/24
                eval_log[f"{base_name}mean_emission"] = emission_sum[b]/24

            if i % args.log_interval == 0:

                write_log(log_path=logs_path, log={'Episode': i, **train_log, **eval_log})                    
                pbar.set_postfix({'Episode': i, **train_log, **eval_log})

                if args.wandb_log:

                    wandb.log({**train_log, **eval_log}, step = int(wandb_step))

            # Update pbar

            pbar.update(1)

    # Close WandB run

    if args.noisy_training:

        # Delete the temp model

        os.remove(f"./models/tmp/sac.zip")

    if args.wandb_log:

        run.finish()

    print("Training finished")