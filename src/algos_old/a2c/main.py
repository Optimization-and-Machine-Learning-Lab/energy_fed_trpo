import json
from math import e
import wandb
import argparse

from time import time 
from tqdm import tqdm
from pathlib import Path

from stable_baselines3.common.noise import NormalActionNoise
from src.utils.utils import set_seed, write_log, get_env_from_config, init_config
from stable_baselines3 import A2C
from src.utils.cl_rewards import *
from src.utils.cl_env_helper import *
from stable_baselines3.common.monitor import Monitor
from citylearn.wrappers import (
    NormalizedObservationWrapper,
    StableBaselines3Wrapper,
)
from gymnasium.wrappers import NormalizeReward
from stable_baselines3.common.vec_env import SubprocVecEnv


ACTIVE_OBSERVATIONS = [
    'hour',
    'day_type',
    'solar_generation',
    'net_electricity_consumption',
    'electrical_storage_soc',
    'non_shiftable_load',
    'direct_solar_irradiance',
    'direct_solar_irradiance_predicted_6h',
    'direct_solar_irradiance_predicted_12h',
    'direct_solar_irradiance_predicted_24h',
]

REWARDS = {
    'cost': Cost,
    'weighted_cost_emissions': WeightedCostAndEmissions,
    'cost_pen_no_batt': CostNoBattPenalization,
    'cost_pen_bad_batt': CostBadBattUsePenalization,
    'cost_pen_bad_action': CostIneffectiveActionPenalization,
}

def parse_args():

    parser = argparse.ArgumentParser(description='CityLearn A2C experiment')

    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--gae_lambda", type=float, default=1.0)
    parser.add_argument("--qf_nns", type=int, default=16)
    parser.add_argument("--pi_nns", type=int, default=32)
    parser.add_argument("--buffer_size", type=int, default=int(2e6))
    parser.add_argument('--wandb-log', default=False, action='store_true', help='Log to wandb (default: True)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='LI', help='Interval between training status logs (default: 1)')
    parser.add_argument('--device', type=str, default=None, help='device (default: None)')
    parser.add_argument('--data-path', type=str, default='./data/simple_data/', help='Data path (default: ./data/simple_data)')
    parser.add_argument('--reward', type=str, help='Reward function (default: cost_pen_no_batt)')
    parser.add_argument('--day-count', type=int, default=1, help='Number of days for training (default: 1)')
    parser.add_argument('--noisy_training', default=False, action='store_true', help='Noisy training (default: False)')
    parser.add_argument('--price-margin', type=float, default=0.1, help='Price margin for noisy training (default: 0.1)')
    
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
    
    # Configure Environment

    schema_filepath = args.data_path + 'schema.json'
    eval_schema_filepath = args.data_path + 'eval/schema.json'

    with open(schema_filepath) as json_file:
        schema_dict = json.load(json_file)

    with open(eval_schema_filepath) as json_file:
        eval_schema_dict = json.load(json_file)

    # Environment initialization, we create two different to not affect the episode tracker
                
    env_config = {
        "schema": schema_dict,
        "central_agent": True,
        "active_observations": ACTIVE_OBSERVATIONS,
        "reward_function": REWARDS[args.reward],
        "random_seed": args.seed,
        "day_count": args.day_count,
        "extended_obs": True,
        "price_margin": args.price_margin,
    }

    # initialize environment

    train_env = SubprocVecEnv([get_env_from_config(config=env_config, seed=args.seed) for _ in range(args.batch_size)])
    train_env = NormalizeReward(train_env, gamma=args.gamma, epsilon=1e-8)
    train_env = NormalizedObservationWrapper(train_env)
    train_env = StableBaselines3Wrapper(train_env)
    train_env = Monitor(train_env)

    eval_env = SubprocVecEnv([get_env_from_config(config={
        **env_config,
        "schema": eval_schema_dict,
    }, seed=args.seed) for _ in range(args.batch_size)])
    eval_env = NormalizeReward(eval_env, gamma=args.gamma, epsilon=1e-8)
    eval_env = NormalizedObservationWrapper(eval_env)
    eval_env = StableBaselines3Wrapper(eval_env)
    eval_env = Monitor(eval_env)

    # Initialize the model

    n_buildings = len(train_env.unwrapped.buildings)

    model_config = {
        "learning_rate": args.learning_rate,
        # "n_steps": args.batch_size,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "device": device,
        "seed": args.seed,
        "policy_kwargs": {
            "net_arch": {
                "pi": [args.pi_nns, args.pi_nns],
                "vf": [args.qf_nns, args.qf_nns],
            }
        },
        "normalize_advantage": True,
        # "verbose": 1,
    }

    # Setup WandB if enabled

    run = None

    if args.wandb_log:

        run =  wandb.init(
            name=f"a2c_seed_{args.seed}_{str(int(time()))}", 
            project="trpo_rl_energy",
            entity="optimllab",
            config={
                "algorithm": "a2c",
                "seed": args.seed,
                **model_config
            },
            sync_tensorboard=False,
        )

    # Define a path to create logs

    logs_path = wandb.run.dir if args.wandb_log else f"./logs/a2c_seed_{args.seed}_t_{str(int(time()))}"
    Path(logs_path).mkdir(exist_ok=True)

    model = A2C("MlpPolicy", train_env, **model_config)#, tensorboard_log=logs_path)
    
    # Train the model
    
    # model.learn(
    #     total_timesteps=train_env.unwrapped.time_steps * args.n_episodes,
    #     progress_bar=True,
    #     callback=[
    #         CustomCallback(),
    #         # WandbCallback(verbose=1),
    #         CustomEvalCallback(
    #             eval_env=eval_env,
    #             n_eval_episodes=1,
    #             eval_freq=train_env.unwrapped.time_steps,
    #             deterministic=True,
    #         )
                
    #     ]
    # )

    with tqdm(total=args.n_episodes, nrows=10) as pbar:

        best_eval_reward = -np.inf
        wandb_step = 0

        for i_episode in range(args.n_episodes):

            if args.noisy_training and i_episode > 0:
                
                train_env = get_env_from_config(config=env_config, seed=i_episode)
                train_env = NormalizedObservationWrapper(train_env)
                train_env = StableBaselines3Wrapper(train_env)

                model = A2C("MlpPolicy", train_env, **model_config)
                model.load("./models/tmp/a2c", env=train_env)

            # Train the model
            model.learn(
                total_timesteps=(train_env.unwrapped.time_steps),
            )

            if args.noisy_training:

                model.save(f"./models/tmp/a2c")

            observations, _ = train_env.reset()

            while not train_env.unwrapped.terminated:

                actions, _ = model.predict(observations, deterministic=True)
                observations, _, _, _, _ = train_env.step(actions)
                
            b_reward_mean = train_env.unwrapped.episode_rewards[-1]['mean']
            b_cost_sum = [np.mean(b.net_electricity_consumption_cost) for b in train_env.unwrapped.buildings]
            b_emission_sum = [np.mean(b.net_electricity_consumption_emission) for b in train_env.unwrapped.buildings]
            
            train_log = {}

            train_log["train/mean_hour_reward"] = np.mean(b_reward_mean)

            for b in range(len(train_env.unwrapped.buildings)):

                # For logging

                base_name = f"train/b_{train_env.unwrapped.buildings[b].name[-1]}_"

                train_log[f"{base_name}mean_hour_cost"] = b_cost_sum[b]
                train_log[f"{base_name}mean_hour_emission"] = b_emission_sum[b]

            # Check kpis for eval environment

            observations, _ = eval_env.reset()

            eval_actions = []

            while not eval_env.unwrapped.terminated:

                action, _ = model.predict(observations, deterministic=True)
                eval_actions.append(action)
                observations, _, _, _, _ = eval_env.step(action)
                
            eval_log = {}

            b_reward_mean = eval_env.unwrapped.episode_rewards[-1]['mean']
            b_cost_sum = [np.mean(b.net_electricity_consumption_cost) for b in train_env.unwrapped.buildings]
            b_emission_sum = [np.mean(b.net_electricity_consumption_emission) for b in train_env.unwrapped.buildings]
            
            # Building level sampling and logging

            eval_log["eval/mean_hour_reward"] = np.mean(b_reward_mean)

            for b in range(len(eval_env.unwrapped.buildings)):

                base_name = f"eval/b_{eval_env.unwrapped.buildings[b].name[-1]}_"

                eval_log[f"{base_name}mean_hour_cost"] = b_cost_sum[b]
                eval_log[f"{base_name}mean_hour_emission"] = b_emission_sum[b]

            if i_episode % args.log_interval == 0:

                write_log(log_path=logs_path, log={'Episode': i_episode, **train_log, **eval_log})                    
                pbar.set_postfix({'Episode': i_episode, **train_log, **eval_log})

                if args.wandb_log:

                    wandb.log({**train_log, **eval_log}, step = int(wandb_step))

                    if eval_log["eval/mean_hour_reward"] > best_eval_reward:

                        best_eval_reward = eval_log["eval/mean_hour_reward"]

                        # Prepare actions for logging

                        eval_actions = np.array(eval_actions).transpose(1, 0)

                        for b in range(len(eval_env.unwrapped.buildings)):
                            
                            actions = []
                            socs = []

                            for h in range(len(eval_actions[b])):

                                actions.append([h, eval_actions[b][h], f"learned - e {i_episode}"])
                                actions.append([h, eval_env.unwrapped.optimal_actions[b][h], f"optimal"])

                                socs.append([h, eval_env.unwrapped.buildings[b].electrical_storage.soc[h + 1], f"learned - e {i_episode}"])
                                socs.append([h, eval_env.unwrapped.optimal_soc[b][h], "optimal"])

                            
                            table_actions = wandb.Table(data=actions, columns=["hour", "action", "policy"])
                            table_socs = wandb.Table(data=socs, columns=["hour", "soc", "policy"])

                            wandb.log({
                                f"policies/b_{b + 1}": wandb.plot.line(
                                    table_actions, x="hour", y="action", stroke="policy", title=f"Best Building {b + 1} Actions"
                                ),
                                f"socs/b_{b + 1}": wandb.plot.line(
                                    table_socs, x="hour", y="soc", stroke="policy", title=f"Best Building {b + 1} SOC"
                                )
                            }, step=int(wandb_step))

                        # model.save(f"./models/a2c_{args.seed}")

            wandb_step += 1     # count training step
            
            # Update pbar

            pbar.update(1)

    # Close WandB run

    if args.noisy_training:

        # Delete the temp model

        os.remove(f"./models/tmp/a2c.zip")

    if args.wandb_log:

        run.finish()

    print("Training finished")
