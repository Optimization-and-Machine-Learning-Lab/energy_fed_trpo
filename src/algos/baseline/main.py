import json
import wandb
import argparse
import numpy as np

from time import time
from tqdm import tqdm
from pathlib import Path

from src.utils.utils import init_config, set_seed, write_log, get_env_from_config
from src.utils.cl_rewards import (
    Cost,
    WeightedCostAndEmissions,
    CostNoBattPenalization,
    CostBadBattUsePenalization,
    CostIneffectiveActionPenalization,
)

from citylearn.agents.base import BaselineAgent
from gymnasium.wrappers import NormalizeReward

ACTIVE_OBSERVATIONS = [
    'hour',
]

REWARDS = {
    'cost': Cost,
    'weighted_cost_emissions': WeightedCostAndEmissions,
    'cost_pen_no_batt': CostNoBattPenalization,
    'cost_pen_bad_batt': CostBadBattUsePenalization,
    'cost_pen_bad_action': CostIneffectiveActionPenalization,
}

def parse_args():

    parser = argparse.ArgumentParser(description='CityLearn TRPO experiment')

    parser.add_argument('--seed', type=int, default=0, metavar='S', help='Random seed (default: 0)')
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument('--wandb-log', default=False, action='store_true', help='Log to wandb (default: True)')
    parser.add_argument('--device', type=str, default=None, help='device (default: None)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='LI', help='Interval between training status logs (default: 1)')
    parser.add_argument('--day-count', type=int, default=1, help='Number of days for training (default: 1)')
    parser.add_argument('--data-path', type=str, default='./data/simple_data/', help='Data path (default: ./data/simple_data)')
    parser.add_argument('--reward', type=str, help='Reward function (default: cost_pen_no_batt)')
    parser.add_argument("--gamma", type=float, default=0.9)

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
                "reward": args.reward,
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
        "reward_function": REWARDS[args.reward],
        "random_seed": args.seed,
        "day_count": args.day_count,
        "central_agent": False,
    }

    # initialize environment

    eval_env = get_env_from_config(env_config, seed=args.seed)
    eval_env = NormalizeReward(eval_env, gamma=args.gamma, epsilon=1e-8)

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
            
            # Check kpis for eval environment

            observations, _ = eval_env.reset()

            while not eval_env.unwrapped.terminated:

                actions = model.predict(observations, deterministic=True)
                
                observations, reward, _, _, _ = eval_env.step(actions)
                
            b_reward_mean = eval_env.unwrapped.episode_rewards[-1]['mean']
            b_cost_sum = [np.mean(b.net_electricity_consumption_cost) for b in eval_env.buildings]
            b_emission_sum = [np.mean(b.net_electricity_consumption_emission) for b in eval_env.buildings]
            
            eval_log = {}

            eval_log["eval/mean_hour_reward"] = np.mean(b_reward_mean)

            for b in range(len(eval_env.unwrapped.buildings)):

                base_name = f"eval/b_{eval_env.buildings[b].name[-1]}_"

                eval_log[f"{base_name}mean_hour_reward"] = b_reward_mean[b]
                eval_log[f"{base_name}mean_hour_cost"] = b_cost_sum[b]
                eval_log[f"{base_name}mean_hour_emission"] = b_emission_sum[b]

            if i % args.log_interval == 0:

                write_log(log_path=logs_path, log={'Episode': i, **eval_log})                    
                pbar.set_postfix({'Episode': i, **eval_log})

                if args.wandb_log:

                    wandb.log({**eval_log}, step = int(wandb_step))

            wandb_step += 1     # count training step

            # Update pbar

            pbar.update(1)

    # Close WandB run

    if args.wandb_log:

        run.finish()

    print("Training finished")