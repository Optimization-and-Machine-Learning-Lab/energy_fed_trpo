import json
import numpy as np

from src.utils import (
    set_seed, get_env_from_config,
    GeneralLogger, Cost, WeightedCostAndEmissions, CostNoBattPenalization,
    CostBadBattUsePenalization, CostIneffectiveActionPenalization,
    REWARDS,
)

# Torch
import torch

# Tensordict modules
from tensordict import TensorDict

# Data collection
from torchrl.data import Composite, Categorical, Bounded, Unbounded

# Env
from torchrl.envs import RewardSum, TransformedEnv, EnvBase
from torchrl.envs.utils import check_env_specs

# Utils

from matplotlib import pyplot as plt
from gymnasium.wrappers import NormalizeReward
from citylearn.wrappers import (
    NormalizedObservationWrapper,
)

def plot_rewards_and_actions(
    policy, train_env, eval_env, train_rewards, eval_rewards, save: bool = False,
    save_path: str = None, wandb_log: bool = False
):

    # Setup the logger

    logger = GeneralLogger()

    # Manage the case when we call the logger from a notebook (not Wandb logging)

    if not logger._initialized:
        logger.setup({}) # Init by default

    # Plot rewards and actions

    fig, axs = plt.subplots(2, 3, figsize=(18, 8))

    # Plot rewards
    axs[0, 0].plot(train_rewards, label="Training Reward Mean")
    axs[0, 0].plot(eval_rewards, label="Evaluation Reward Mean")
    axs[0, 0].set_xlabel("Training Iterations")
    axs[0, 0].set_ylabel("Reward")
    axs[0, 0].set_title("Training and Evaluation Rewards")
    axs[0, 0].grid()
    axs[0, 0].legend()

    logger.log_line_series(
        name='global/rewards',
        xs=list(range(len(train_rewards))),
        ys=[train_rewards, eval_rewards],
        keys=['Train', 'Eval'],
        title='Rewards',
        x_name='Epochs'
    ) 

    # Sample actions for train_env and eval_env
    with torch.no_grad():
        
        train_rollout = train_env.rollout(train_env.cl_env.unwrapped.time_steps - 1, policy=policy)
        train_actions = train_rollout.get(train_env.action_key).cpu().squeeze()
        train_soc = [b.electrical_storage.soc for b in train_env.cl_env.unwrapped.buildings]
        train_net_electricity_consumption = [b.net_electricity_consumption for b in train_env.cl_env.unwrapped.buildings]
        train_opt_actions = torch.tensor(np.array(train_env.cl_env.unwrapped.optimal_actions), requires_grad=False).swapaxes(0, 1)
        train_opt_soc = torch.tensor(np.array(train_env.cl_env.unwrapped.optimal_soc), requires_grad=False).swapaxes(0, 1)

        eval_rollout = eval_env.rollout(eval_env.cl_env.unwrapped.time_steps - 1, policy=policy)
        eval_actions = eval_rollout.get(eval_env.action_key).cpu().squeeze()
        eval_soc = [b.electrical_storage.soc for b in eval_env.cl_env.unwrapped.buildings]
        eval_net_electricity_consumption = [b.net_electricity_consumption for b in eval_env.cl_env.unwrapped.buildings]
        eval_opt_actions = torch.tensor(np.array(eval_env.cl_env.unwrapped.optimal_actions), requires_grad=False).swapaxes(0, 1)
        eval_opt_soc = torch.tensor(np.array(eval_env.cl_env.unwrapped.optimal_soc), requires_grad=False).swapaxes(0, 1)

    for i in range(train_env.n_agents):
        
        # Plot actions for each building
        
        axs[0, 1].plot(train_actions[:, i].numpy(), label=f"Train Agent {i} actions")
        axs[0, 1].plot(train_opt_actions[:, i], label=f"Train Agent {i} optimal actions", linestyle="--")

        logger.log_line_series(
            name=f'train/actions_building_{i}',
            xs=list(range(len(train_actions[:, i]))),
            ys=[train_actions[:, i], train_opt_actions[:, i]],
            keys=['Best Policy', 'Optimal'],
            title=f'Actions Building {i}',
            x_name='Hours'
        )
        
        axs[0, 2].plot(eval_actions[:, i].numpy(), label=f"Eval Agent {i} actions")
        axs[0, 2].plot(eval_opt_actions[:, i], label=f"Eval Agent {i} optimal actions", linestyle="--")

        logger.log_line_series(
            name=f'eval/actions_building_{i}',
            xs=list(range(len(eval_actions[:, i]))),
            ys=[eval_actions[:, i], eval_opt_actions[:, i]],
            keys=['Best Policy', 'Optimal'],
            title=f'Actions Building {i}',
            x_name='Hours'
        )

        # Plot SOC for each building
        
        axs[1, 1].plot(train_soc[i], label=f"Train Agent {i} SOC")
        axs[1, 1].plot(train_opt_soc[:, i], label=f"Train Agent {i} optimal SOC", linestyle="--")

        logger.log_line_series(
            name=f'train/soc_building_{i}',
            xs=list(range(len(train_soc[i]))),
            ys=np.stack([train_soc[i], train_opt_soc[:, i]], dtype=np.float32),
            keys=['Best Policy', 'Optimal'],
            title=f'SoC Building {i}',
            x_name='Hours'
        )

        axs[1, 2].plot(eval_soc[i], label=f"Eval Agent {i} SOC")
        axs[1, 2].plot(eval_opt_soc[:, i], label=f"Eval Agent {i} optimal SOC", linestyle="--")

        logger.log_line_series(
            name=f'eval/soc_building_{i}',
            xs=list(range(len(eval_soc[i]))),
            ys=np.stack([eval_soc[i], eval_opt_soc[:, i]], dtype=np.float32),
            keys=['Best Policy', 'Optimal'],
            title=f'SoC Building {i}',
            x_name='Hours'
        )
    
        # Plot net electricity consumption for each building
        
        axs[1, 0].plot(train_net_electricity_consumption[i], label=f"Train Agent {i} Net Electricity Consumption")
        axs[1, 0].plot(eval_net_electricity_consumption[i], label=f"Eval Agent {i} Net Electricity Consumption", linestyle="--")

        logger.log_line_series(
            name=f'eval/net_building_{i}',
            xs=list(range(len(train_net_electricity_consumption[i]))),
            ys=[train_net_electricity_consumption[i], eval_net_electricity_consumption[i]],
            keys=['Train', 'Eval'],
            title=f'Net Energy Building {i}',
            x_name='Hours'
        )

    # Configure plots, labels and titles

    axs[0, 1].set_xlabel("Hour of the day")
    axs[0, 1].set_ylabel("Action")
    axs[0, 1].set_title("Train Actions Comparison")
    axs[0, 1].grid()
    axs[0, 1].legend()

    axs[0, 2].set_xlabel("Hour of the day")
    axs[0, 2].set_ylabel("Action")
    axs[0, 2].set_title("Eval Actions Comparison")
    axs[0, 2].grid()
    axs[0, 2].legend()

    axs[1, 1].set_xlabel("Hour of the day")
    axs[1, 1].set_ylabel("SOC")
    axs[1, 1].set_title("Train SOC Comparison")
    axs[1, 1].grid()
    axs[1, 1].legend()

    axs[1, 2].set_xlabel("Hour of the day")
    axs[1, 2].set_ylabel("SOC")
    axs[1, 2].set_title("Eval SOC Comparison")
    axs[1, 2].grid()
    axs[1, 2].legend()

    axs[1, 0].set_xlabel("Hour of the day")
    axs[1, 0].set_ylabel("Net Electricity Consumption")
    axs[1, 0].set_title("Net Electricity Consumption Comparison")
    axs[1, 0].grid()
    axs[1, 0].legend()

    plt.suptitle(f'Results for day {train_env.cl_env.unwrapped.episode_tracker.episode_start_time_step // 24}')
    plt.tight_layout()

    # Plot only is there is a graphical interface

    plt.show()

    # Consider saving the figure

    if save:
        fig.savefig(save_path + 'results.png')

def create_env(env_config, device, seed):
    
    env = CityLearnMultiAgentEnv(
        env_config=env_config,
        device=device,
        seed=seed,
    )

    # Include reward sum in the environment
    env = TransformedEnv(
        env,
        RewardSum(in_keys=[env.reward_key], out_keys=[("agents", "episode_reward")]),
    )
    
    return env

# TODO: Handle the case when the environment is configured as a central agent

class CityLearnMultiAgentEnv(EnvBase):
    
    def __init__(self, env_config, device, seed, batch_size=1):

        super().__init__()

        self.cl_env = get_env_from_config(config=env_config, seed=seed)
        self.cl_env = NormalizeReward(self.cl_env, gamma=1, epsilon=1e-8)
        self.cl_env = NormalizedObservationWrapper(self.cl_env)
        self.n_agents = len(env_config["schema"]['buildings'])
        self.batch_size = torch.Size([batch_size,])
        self.device = device

        action_specs = []
        observation_specs = []
        reward_specs = []
        
        cl_env_as = self.cl_env.action_space
        cl_env_os = self.cl_env.observation_space
        self.n_actions = cl_env_as[0].shape[0]
        self.n_observations = cl_env_os[0].shape[0]

        for i in range(self.n_agents):
            
            action_specs.append(Bounded(
                low=cl_env_as[i].low,
                high=cl_env_as[i].high,
                shape=cl_env_as[i].shape,
                device=device
            ))
            reward_specs.append(Unbounded
                (shape=(1,), dtype=torch.float, device=device),
            )
            observation_specs.append(Bounded(
                low=cl_env_os[i].low,
                high=cl_env_os[i].high,
                shape=cl_env_os[i].shape,
                device=device
            ))

        # Define observation and action spaces

        self.action_spec = Composite({
            "agents": Composite(
                {"action": torch.stack(action_specs, dim=0).view(self.batch_size[0], self.n_agents, cl_env_as[0].shape[0])},
                shape=(self.batch_size[0], self.n_agents,)
            )
        }, batch_size=self.batch_size)

        self.unbatched_action_spec = Composite({
            "agents": Composite(
                {"action": torch.stack(action_specs, dim=0).view(self.n_agents, cl_env_as[0].shape[0])},
                shape=(self.n_agents,)
            )
        })

        self.reward_spec = Composite({
            "agents": Composite(
                {"reward": torch.stack(reward_specs, dim=0).view(self.batch_size[0], self.n_agents, 1)},
                shape=(self.batch_size[0], self.n_agents,)
            )
        }, batch_size=self.batch_size)

        self.unbatched_reward_spec = Composite({
            "agents": Composite(
                {"reward": torch.stack(reward_specs, dim=0).view(self.n_agents, 1)},
                shape=(self.n_agents,)
            )
        })

        self.observation_spec = Composite({
            "agents": Composite(
                {"observation": torch.stack(observation_specs, dim=0).expand(self.batch_size[0], self.n_agents, cl_env_os[0].shape[0])},
                shape=(self.batch_size[0], self.n_agents,)
            )
        }, batch_size=self.batch_size)

        self.unbatched_observation_spec = Composite({
            "agents": Composite(
                {"observation": torch.stack(observation_specs, dim=0).expand(self.n_agents, cl_env_os[0].shape[0])},
                shape=(self.n_agents,)
            )
        })

        self.done_spec = Categorical(n=2, shape=torch.Size((self.batch_size[0], )), dtype=torch.bool)

        # Initialize state variables
        self.current_step = 0
        self.done = False

    def _reset(self, tensordict=None):
        observations, _ = self.cl_env.reset()
        tensordict = TensorDict({
            "agents": TensorDict({
                # "info": torch.empty(self.batch_size),
                "observation": torch.tensor(
                    np.array(observations), dtype=torch.float
                ).reshape(self.batch_size[0], len(observations), len(observations[0])),
            }, torch.Size((self.batch_size[0], self.n_agents)), device=self.device),
            "done": torch.tensor(False, dtype=torch.bool).repeat(*self.batch_size)
        }, batch_size=self.batch_size, device=self.device)
        
        return tensordict

    def _step(self, tensordict):
       
        # Step through the environment
        next_obs, rewards, done, _, _ = self.cl_env.step(tensordict['agents','action'].cpu().squeeze(0).numpy())
        self.done = done

        # Prepare TensorDict for the step
        step_results = TensorDict({
            "agents": TensorDict({
                "reward": torch.tensor(
                    rewards, dtype=torch.float
                ).reshape(self.batch_size[0], self.n_agents, self.n_actions),
                "observation": torch.tensor(
                    np.array(next_obs), dtype=torch.float
                ).reshape(self.batch_size[0], self.n_agents, self.n_observations),
                # "info": torch.empty(self.batch_size),
            }, batch_size=torch.Size((self.batch_size[0], self.n_agents)), device=self.device),
            "done": torch.tensor(done, dtype=torch.bool).repeat(*self.batch_size)
        }, batch_size=self.batch_size, device=self.device)

        return step_results

    def _set_seed(self, seed):
        self.cl_env.seed(seed)
        torch.manual_seed(seed)

    def close(self):
        self.cl_env.close()

if __name__ == '__main__':
    
    REWARDS = {
        'cost': Cost,
        'weighted_cost_emissions': WeightedCostAndEmissions,
        'cost_pen_no_batt': CostNoBattPenalization,
        'cost_pen_bad_batt': CostBadBattUsePenalization,
        'cost_pen_bad_action': CostIneffectiveActionPenalization,
    }

    # Common configurations for environments

    active_observations = [
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
        'selling_price'
    ]

    data_path = 'data/naive_data/'
    reward = 'weighted_cost_emissions'
    seed = 0
    price_margin = 0.1
    day_count = 1

    device = 'cpu'

    set_seed(seed)

    # Training configurations

    schema_filepath = data_path + 'schema.json'

    with open(schema_filepath) as json_file:
        schema_dict = json.load(json_file)

    train_env_config = {
        "schema": schema_dict,
        "central_agent": False,
        "active_observations": active_observations,
        "reward_function": REWARDS[reward],
        "random_seed": seed,
        "day_count": day_count,
        "price_margin": price_margin,
        "personal_encoding": True,
    }

    env = create_env(train_env_config, device, seed)

    check_env_specs(env)
