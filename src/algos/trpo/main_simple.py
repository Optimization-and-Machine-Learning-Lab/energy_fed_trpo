from copy import deepcopy
import wandb
import torch
import json
import argparse
import random 
import numpy as np
import scipy.optimize

from tqdm import tqdm
from time import time
from pathlib import Path
from collections import namedtuple
from torch.autograd import Variable
from citylearn.preprocessing import OnehotEncoding, PeriodicNormalization

from replay_memory import Memory
from running_state import ZFilter
from src.algos.trpo.trpo import trpo_step
from src.algos.trpo.models import Policy, Value
from src.utils.cl_env_helper import *
from src.utils.utils import (
    get_env_from_config, set_seed, set_flat_params_to, get_flat_params_from, get_flat_grad_from, normal_log_density, write_log, init_config
)
from src.utils.cl_rewards import (
    Cost,
    WeightedCostAndEmissions,
    CostNoBattPenalization,
    CostBadBattUsePenalization,
    CostIneffectiveActionPenalization,
)

# ACTIVE_OBSERVATIONS = [
#     'hour', 'electrical_storage_soc', 'outdoor_dry_bulb_temperature', 'outdoor_relative_humidity',
#     'non_shiftable_load', 'electricity_pricing', 'carbon_intensity'
# ]

ACTIVE_OBSERVATIONS = [
    'hour',
    'day_type',
    'solar_generation',
    'net_electricity_consumption',
    'electrical_storage_soc',
    'non_shiftable_load',
    'outdoor_dry_bulb_temperature',
    'outdoor_relative_humidity',
    'direct_solar_irradiance',
    'direct_solar_irradiance_predicted_6h',
    'direct_solar_irradiance_predicted_12h',
    'direct_solar_irradiance_predicted_24h',
    'electricity_pricing',
    'carbon_intensity'
]

REWARDS = {
    'cost': Cost,
    'weighted_cost_emissions': WeightedCostAndEmissions,
    'cost_pen_no_batt': CostNoBattPenalization,
    'cost_pen_bad_batt': CostBadBattUsePenalization,
    'cost_pen_bad_action': CostIneffectiveActionPenalization,
}

class TRPO:

    def __init__(
        self, env_config : dict, device='cpu', gamma : float = 0.995, tau : float = 0.97, l2_reg : float = 1e-3, 
        max_kl : float = 1e-2, damping : float = 1e-1, seed : int = 1, batch_size : int = 15000, log_interval : int = 1, wandb_log : bool = False,
        training_type : str = 'fl', noisy : bool = False
    ):

        self.device = device
       
        self.env_config = env_config
        self.gamma = gamma
        self.tau = tau
        self.l2_reg = l2_reg
        self.max_kl = max_kl
        self.damping = damping
        self.seed = seed
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.wandb_log = wandb_log
        self.training_type = training_type
        self.personalization = training_type == 'fl-personalized'
        self.noisy = noisy
        self.best_eval_reward = -np.inf

        # Initialize evaluation environment (this will be fixed troughout the training)

        self.eval_env = get_env_from_config(config=env_config, seed=seed)

        # Extract environments characteristics

        self.num_inputs = self.eval_env.observation_space[0].shape[0] + 1 # We add one extra input from the time encoding (periodic)
        self.num_actions = self.eval_env.action_space[0].shape[0]
        self.building_count = len(self.eval_env.buildings)

        if self.personalization:

            self.num_inputs += self.building_count # We add extra length for the one-hot encoding of buildings

        self.encoder = OnehotEncoding(list(range(self.building_count)))
        self.periodic_encoder = PeriodicNormalization(x_max=24)

        # Initialize networks

        self.policy_net = Policy(
            num_inputs=self.num_inputs, num_outputs=self.num_actions, one_hot=self.personalization, one_hot_dim=self.building_count
        ).to(device=self.device)
        self.value_net = Value(
            num_inputs=self.num_inputs, one_hot=self.personalization, one_hot_dim=self.building_count
        ).to(device=self.device)

        if self.wandb_log:

            wandb.watch(self.policy_net, log='all')
            wandb.watch(self.value_net, log='all')

        # Define a path to create logs

        self.logs_path = wandb.run.dir if wandb.run is not None else f"./logs/trpo_seed_{self.seed}_n_inputs_{self.num_inputs}_t_{str(int(time()))}"
        Path(self.logs_path).mkdir(exist_ok=True)

        # Define running states

        # state: ['hour', 'outdoor_dry_bulb_temperature', 'outdoor_relative_humidity', 'carbon_intensity', 'non_shiftable_load', 'electrical_storage_soc', 'electricity_pricing']

        if self.personalization:
            self.train_running_state = [ZFilter((self.num_inputs - self.building_count - 2,), clip = 5) for _ in range(self.building_count)]
            self.eval_running_state = [ZFilter((self.num_inputs - self.building_count - 2,), clip = 5) for _ in range(self.building_count)]
        else:
            self.train_running_state = [ZFilter((self.num_inputs - 2,), clip = 5) for _ in range(self.building_count)]
            self.eval_running_state = [ZFilter((self.num_inputs - 2,), clip = 5) for _ in range(self.building_count)]

        # Properties
            
        self.transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state', 'reward'))

    def select_action(self, state, eval : bool = False):

        state = torch.tensor(state).to(torch.get_default_dtype()).to(device=self.device)
        action_mean, _, action_std = self.policy_net(Variable(state))
        action = torch.normal(action_mean, action_std)

        return action if not eval else action_mean

    def update_params(self, batch):

        rewards = torch.Tensor(np.stack(batch.reward)).to(device=self.device)
        masks = torch.Tensor(batch.mask).to(device=self.device)
        actions = torch.Tensor(np.concatenate(batch.action, 0)).unsqueeze(-1).to(device=self.device)
        states = torch.Tensor(np.array(batch.state)).to(device=self.device)
        values = self.value_net(Variable(states)).data

        returns = torch.Tensor(actions.size(0),1).to(device=self.device)
        deltas = torch.Tensor(actions.size(0),1).to(device=self.device)
        advantages = torch.Tensor(actions.size(0),1).to(device=self.device)

        prev_return = 0
        prev_value = 0
        prev_advantage = 0

        for i in reversed(range(rewards.size(0))):
        
            returns[i] = rewards[i] + self.gamma * prev_return * masks[i]       # rewards to go for each step
            deltas[i] = rewards[i] + self.gamma * prev_value * masks[i] - values.data[i]        # advantage calculated from the current network
            advantages[i] = deltas[i] + self.gamma * self.tau * prev_advantage * masks[i]       # GAE

            prev_return = returns[i, 0]
            prev_value = values[i, 0]
            prev_advantage = advantages[i, 0]

        targets = Variable(returns)

        # Original code uses the same LBFGS to optimize the value loss

        def get_value_loss(flat_params):

            set_flat_params_to(self.value_net, torch.Tensor(flat_params))

            for param in self.value_net.parameters():
                if param.grad is not None:
                    param.grad.data.fill_(0)

            values_ = self.value_net(Variable(states))

            value_loss = (values_ - targets).pow(2).mean()

            # weight decay
            for param in self.value_net.parameters():
                value_loss += param.pow(2).sum() * self.l2_reg

            value_loss.backward()
            
            return (value_loss.data.cpu().numpy(), get_flat_grad_from(self.value_net).data.cpu().numpy())

        # use L-BFGS-B to update value function

        flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(self.value_net).cpu().numpy(), maxiter=25)
        set_flat_params_to(self.value_net, torch.Tensor(flat_params).to(device=self.device))

        advantages = (advantages - advantages.mean()) / advantages.std()

        action_means, action_log_stds, action_stds = self.policy_net(Variable(states))

        fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

        def get_loss(volatile=False):

            if volatile:
                with torch.no_grad():
                    action_means, action_log_stds, action_stds = self.policy_net(Variable(states))
            else:
                action_means, action_log_stds, action_stds = self.policy_net(Variable(states))
                    
            log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
            action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
            return action_loss.mean()

        def get_kl():

            mean1, log_std1, std1 = self.policy_net(Variable(states))

            mean0 = Variable(mean1.data)
            log_std0 = Variable(log_std1.data)
            std0 = Variable(std1.data)
            kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5   # KL divergence of two normal distribution
            
            return kl.sum(1, keepdim=True)

        trpo_step(self.policy_net, get_loss, get_kl, self.max_kl, self.damping)

    def train(self, num_episodes : int = 1500, pre_training_episodes : int = 0):

        wandb_step = 0
        pre_training_idx = 0

        total_episodes = pre_training_episodes + num_episodes

        with tqdm(total = total_episodes, nrows=10) as pbar:

            for i_episode in range(total_episodes):

                memories = [Memory() for _ in range(self.building_count)]

                reward_batch = np.array([0.] * self.building_count)
                cost_batch = np.array([0.] * self.building_count)
                emission_batch = np.array([0.] * self.building_count)

                if self.noisy or i_episode == 0:
                    
                    self.train_env = get_env_from_config(config=self.env_config, seed=i_episode) if self.training_type != 'upperbound' else deepcopy(self.eval_env)

                # Increment the pre-training index when num_episodes // self.building_count timesteps passed
                
                if pre_training_episodes != 0:

                    pre_training_idx = i_episode // (pre_training_episodes // self.building_count)

                for _ in range(self.batch_size):

                    state, _ = self.train_env.reset()

                    # Add encoding if enabled

                    if self.personalization:
                        state = [np.concatenate((
                            self.encoder * i,
                            self.periodic_encoder*state[i][0],
                            self.train_running_state[i](state[i][1:])
                        )) for i in range(self.building_count)]
                    else:
                        state = [np.concatenate((
                            self.periodic_encoder*state[i][0],
                            self.train_running_state[i](state[i][1:])
                        )) for i in range(self.building_count)]

                    done = False
                    
                    while not done:

                        action = [self.select_action(state[b]).detach().cpu().numpy() for b in range(self.building_count)]

                        # Compute next step

                        next_state, reward, done, _, _ = self.train_env.step(action)

                        if self.personalization:
                            next_state = [np.concatenate((
                                self.encoder * i,
                                self.periodic_encoder * next_state[i][0],
                                self.train_running_state[i](next_state[i][1:])
                            )) for i in range(self.building_count)]
                        else:    
                            next_state = [np.concatenate((
                                self.periodic_encoder * next_state[i][0],
                                self.train_running_state[i](next_state[i][1:])
                            )) for i in range(self.building_count)]

                        ### END OF SECTION ###

                        mask = 0 if done else 1

                        for b in range(self.building_count):

                            memories[b].push(state[b], np.array([action[b]]), mask, next_state[b], reward[b])

                        state = next_state
                        
                    reward_batch += self.train_env.unwrapped.episode_rewards[-1]['mean']
                    cost_batch += [np.mean(b.net_electricity_consumption_cost) for b in self.train_env.buildings]
                    emission_batch += [np.mean(b.net_electricity_consumption_emission) for b in self.train_env.buildings]
                    
                # Perform TRPO update

                batch = []

                for b in range(self.building_count):
                    
                    if pre_training_episodes != 0 and pre_training_idx != b and i_episode < pre_training_episodes: # If in pre-training mode, only train one building at a time

                        continue

                    batch += memories[b].sample()

                batch = self.transition(*zip(*batch))

                # Logging

                if i_episode % self.log_interval == 0:

                    # Logging for current episode
                    
                    train_log = {}

                    train_log["train/mean_hour_reward"] = np.mean([r / self.batch_size for r in reward_batch])

                    # Building level sampling and logging

                    for b in range(self.building_count):

                        reward_batch[b] /= self.batch_size
                        cost_batch[b] /= self.batch_size
                        emission_batch[b] /= self.batch_size

                        base_name = f"train/b_{self.train_env.buildings[b].name[-1]}_"

                        train_log[f"{base_name}mean_hour_reward"] = reward_batch[b]
                        train_log[f"{base_name}mean_hour_cost"] = cost_batch[b]
                        train_log[f"{base_name}mean_hour_emission"] = emission_batch[b]
                    
                    eval_log, eval_actions, best_reward = self.evaluation()

                    write_log(log_path=self.logs_path, log={'Episode': i_episode, **train_log, **eval_log})                    
                    pbar.set_postfix({'Episode': i_episode, **train_log, **eval_log})

                    if self.wandb_log:

                        wandb.log({**train_log, **eval_log}, step = int(wandb_step))

                        if best_reward:

                            # Prepare actions for logging

                            eval_actions = np.array(eval_actions).transpose(1, 0, 2).squeeze()

                            for b in range(self.building_count):
                            
                                actions = []
                                socs = []

                                for h in range(len(eval_actions[b])):

                                    actions.append([h, eval_actions[b][h], f"learned - e {i_episode}"])
                                    actions.append([h, self.eval_env.unwrapped.optimal_actions[b][h], f"optimal"])

                                    socs.append([h, self.eval_env.unwrapped.buildings[b].electrical_storage.soc[h + 1], f"learned - e {i_episode}"])
                                    socs.append([h, self.eval_env.unwrapped.optimal_soc[b][h], "optimal"])

                                
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

                self.update_params(batch)                
                
                wandb_step += 1     # count training step

                # Update pbar

                pbar.update(1)

    def evaluation(self):

        done = False
        state, _ = self.eval_env.reset()
        best_reward = False

        # Add encoding if enabled

        if self.personalization:
            state = [np.concatenate((
                self.encoder * i,
                self.periodic_encoder * state[i][0],
                self.train_running_state[i](state[i][1:])
            )) for i in range(self.building_count)]
        else:
            state = [np.concatenate((
                self.periodic_encoder * state[i][0],
                self.train_running_state[i](state[i][1:])
            )) for i in range(self.building_count)]

        ### END OF SECTION ###

        actions = []

        while not done:

            action = [self.select_action(state[b], eval=True).detach().cpu().numpy() for b in range(self.building_count)]
            actions.append(action)

            next_state, _, done, _, _ = self.eval_env.step(action)

            if self.personalization:
                state = [np.concatenate((
                    self.encoder * i,
                    self.periodic_encoder*next_state[i][0],
                    self.train_running_state[i](next_state[i][1:])
                )) for i in range(self.building_count)]
            else:
                state = [np.concatenate((
                    self.periodic_encoder*next_state[i][0],
                    self.train_running_state[i](next_state[i][1:])
                )) for i in range(self.building_count)]
        
        b_reward_mean = self.eval_env.unwrapped.episode_rewards[-1]['mean']
        b_cost_sum = [np.mean(b.net_electricity_consumption_cost) for b in self.eval_env.buildings]
        b_emission_sum = [np.mean(b.net_electricity_consumption_emission) for b in self.eval_env.buildings]
        
        eval_log = {}

        eval_log["eval/mean_hour_reward"] = np.mean(b_reward_mean)

        if eval_log["eval/mean_hour_reward"] > self.best_eval_reward:

            self.best_eval_reward = eval_log["eval/mean_hour_reward"]
            best_reward = True

        for b in range(self.building_count):

            base_name = f"eval/b_{self.eval_env.buildings[b].name[-1]}_"

            eval_log[f"{base_name}mean_hour_reward"] = b_reward_mean[b]
            eval_log[f"{base_name}mean_hour_cost"] = b_cost_sum[b]
            eval_log[f"{base_name}mean_hour_emission"] = b_emission_sum[b]

        return eval_log, actions, best_reward

    def save_checkpoint(self, path : str):

        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
        }, path)

    def load_checkpoint(self, path : str):

        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])

def parse_args():

    parser = argparse.ArgumentParser(description='CityLearn TRPO experiment')

    parser.add_argument('--gamma', type=float, default=0.995, metavar='G', help='Discount factor (default: 0.995)')
    parser.add_argument('--tau', type=float, default=0.97, metavar='T', help='Tau (default: 0.97)')
    parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='L2', help='L2 regularization regression (default: 1e-3)')
    parser.add_argument('--max-kl', type=float, default=1e-2, metavar='Max_KL', help='Max kl value (default: 1e-2)')
    parser.add_argument('--damping', type=float, default=1e-1, metavar='D', help='Damping (default: 1e-1)')
    parser.add_argument('--seed', type=int, default=0, metavar='S', help='Random seed (default: 0)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS', help='Batch size (default: 15000)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='LI', help='Interval between training status logs (default: 1)')
    parser.add_argument('--training-type', type=str, default='individual', choices={'individual', 'upperbound', 'fl', 'fl-personalized'}, help='Training type (default: individual)')
    parser.add_argument('--pre_training_steps', type=int, default=0, help='Number of pre-training steps (default: 0)')
    parser.add_argument('--wandb-log', default=False, action='store_true', help='Log to wandb (default: True)')
    parser.add_argument('--device', type=str, default=None, help='device (default: None)')
    parser.add_argument('--n_episodes', type=int, default=100, help='Number of episodes (default: 1500)')
    parser.add_argument('--day-count', type=int, default=1, help='Number of days for training (default: 1)')
    parser.add_argument('--n_buildings', type=int, default=1, help='Number of buildings to train (default: 1)')
    parser.add_argument('--building_id', type=int, default=1, help='Trained building (if individual building training)')
    parser.add_argument('--data-path', type=str, default='./data/simple_data/', help='Data path (default: ./data/simple_data)')
    parser.add_argument('--noisy_training', default=False, action='store_true', help='Noisy training (default: False)')
    parser.add_argument('--reward', type=str, help='Reward function (default: cost_pen_no_batt)')

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
            name=f"{args.training_type}{'' if 'fl' in args.training_type else f'_b_{args.building_id}'}_seed_{args.seed}",  
            project="trpo_rl_energy",
            entity="optimllab",
            config={
                "algorithm": "trpo",
                "training_type": args.training_type,
                "seed": args.seed,
                "gamma": args.gamma,
                "tau": args.tau,
                "l2_reg": args.l2_reg,
                "max_kl": args.max_kl,
                "damping": args.damping,
                "batch_size": args.batch_size,
                "reward": args.reward,
                "noisy_training": args.noisy_training,
                "pre_training_steps": args.pre_training_steps,
            },
            sync_tensorboard=False,
        )

    # Configure Environment

    schema_filepath = args.data_path + 'schema.json'

    with open(schema_filepath) as json_file:
        schema_dict = json.load(json_file)

    if args.training_type != 'fl-personalized':

        schema_dict["personalization"] = False

    if args.training_type == 'upperbound' or args.training_type == 'individual':

        for b_i, b_name in enumerate(schema_dict["buildings"]):
            
            if b_i != args.building_id:

                schema_dict["buildings"][b_name]["include"] = False

    # Environment initialization, we create two different to not affect the episode tracker
                
    env_config = {
        "schema": schema_dict,
        "active_observations": ACTIVE_OBSERVATIONS,
        "reward_function": REWARDS[args.reward],
        "random_seed": args.seed,
        "day_count": args.day_count,
        "extended_obs": True,
    }

    # TRPO initialization

    trpo = TRPO(
        env_config, device, args.gamma, args.tau, args.l2_reg, args.max_kl, args.damping, args.seed, args.batch_size,
        args.log_interval, args.wandb_log, args.training_type, args.noisy_training
    )

    # Training

    trpo.train(num_episodes=args.n_episodes, pre_training_episodes=args.pre_training_steps)

    # Save checkpoint

    trpo.save_checkpoint(f"trpo_{args.training_type}_seed{args.seed}_{str(int(time()))}.pth")

    # Close WandB run

    if args.wandb_log:

        run.finish()

    print("Training finished")
