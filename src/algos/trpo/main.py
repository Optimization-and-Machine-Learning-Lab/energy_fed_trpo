import wandb
import torch
import argparse
import numpy as np
import scipy.optimize

from tqdm import tqdm
from time import time
from pathlib import Path
from collections import namedtuple
from torch.autograd import Variable
from citylearn.citylearn import CityLearnEnv

from trpo import *
from models import *
from rewards import *
from src.utils.utils import *
from replay_memory import Memory
from running_state import ZFilter
from src.utils.cl_env_helper import *

class TRPO:

    def __init__(
        self, train_env : CityLearnEnv, eval_env : CityLearnEnv, device='cpu', gamma : float = 0.995, tau : float = 0.97, l2_reg : float = 1e-3, 
        max_kl : float = 1e-2, damping : float = 1e-1, seed : int = 1, batch_size : int = 15000, log_interval : int = 1, wandb_log : bool = False,
        personalization : bool = False
    ):

        self.device = device
       
        self.train_env = train_env
        self.eval_env = eval_env
        self.gamma = gamma
        self.tau = tau
        self.l2_reg = l2_reg
        self.max_kl = max_kl
        self.damping = damping
        self.seed = seed
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.wandb_log = wandb_log
        self.personalization = personalization

        # Extract environments characteristics

        self.num_inputs = train_env.observation_space[0].shape[0]
        self.num_actions = train_env.action_space[0].shape[0]
        self.building_count = len(train_env.buildings)

        if personalization:

            self.num_inputs += self.building_count

        # Initialize networks

        self.policy_net = Policy(num_inputs=self.num_inputs, num_outputs=self.num_actions).to(device=self.device)
        self.value_net = Value(num_inputs=self.num_inputs).to(device=self.device)

        # Define a path to create logs

        self.logs_path = wandb.run.dir if self.wandb_log else f"./logs/trpo_seed_{self.seed}_n_inputs_{self.num_inputs}_t_{str(int(time()))}"
        Path(self.logs_path).mkdir(exist_ok=True)

        # Define running state

        # state: [onehot?, temperature, humidity, battery_storage, consumption, price, hour1, hour2]

        if personalization:
            self.train_running_state = [ZFilter((self.num_inputs-self.building_count,), clip=5) for _ in range(self.building_count)]
            self.eval_running_state = [ZFilter((self.num_inputs-self.building_count,), clip=5) for _ in range(self.building_count)]
        else:
            self.train_running_state = [ZFilter((self.num_inputs,), clip=5) for _ in range(self.building_count)]
            self.eval_running_state = [ZFilter((self.num_inputs,), clip=5) for _ in range(self.building_count)]

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

    def train(self, num_episodes : int = 1500):

        wandb_step = 0

        with tqdm(total=num_episodes, nrows=10) as pbar:

            for i_episode in range(num_episodes):

                num_steps = 0
                kpi_count = 0

                kpis = None

                memories = [Memory() for _ in range(self.building_count)]

                while num_steps < self.batch_size:

                    state, _ = self.train_env.reset()

                    # Add encoding if enabled

                    if self.personalization:
                        state = [np.concatenate((state[i][:self.building_count], self.train_running_state[i](state[i][self.building_count:]))) for i in range(self.building_count)]

                    done = False

                    while not done:

                        action = [self.select_action(state[b]).detach().cpu().numpy() for b in range(self.building_count)]
                        next_state, reward, done, _, _ = self.train_env.step(action)

                        if self.personalization:
                            next_state = [np.concatenate((next_state[i][:self.building_count], self.train_running_state[i](next_state[i][self.building_count:]))) for i in range(self.building_count)]
                        else:    
                            next_state = [self.train_running_state[i](next_state[i]) for i in range(self.building_count)]

                        mask = 0 if done else 1

                        for b in range(self.building_count):

                            # Add a penalization for trying to use more energy than what's available in the battery

                            storage_pen = (action[b] - self.train_env.buildings[b].electrical_storage.electricity_consumption[-1]) ** 2

                            memories[b].push(state[b], np.array([action[b]]), mask, next_state[b], reward[b] - storage_pen)

                        state = next_state
                        num_steps += 1

                    # Accumulate kpis

                    if kpis is None:

                        kpis = get_kpis(self.train_env).reset_index()
                        kpi_count += 1

                    else:

                        new_kpis = get_kpis(self.train_env)

                        # Update kpis

                        kpis = pd.concat((kpis, new_kpis)).groupby(['kpi', 'name', 'level'], as_index=False).sum()

                        kpi_count += 1

                    num_episodes += 1

                # Train 

                train_log = {}
                batch = []

                rewards = np.array(pd.DataFrame(self.train_env.unwrapped.episode_rewards)['sum'].tolist())

                wandb_step += 1     # count training step

                #District level logging

                train_log["train/d_emission_avg"] = kpis[kpis['kpi'] == 'Emissions']['value'].iloc[0] / kpi_count
                train_log["train/d_cost_avg"] = kpis[kpis['kpi'] == 'Cost']['value'].iloc[0] / kpi_count
                train_log["train/d_daily_peak_avg"] = kpis[kpis['kpi'] == 'Avg. daily peak']['value'].iloc[0] / kpi_count
                train_log["train/d_load_factor_avg"] = kpis[kpis['kpi'] == '1 - load factor']['value'].iloc[0] / kpi_count
                train_log["train/d_ramping_avg"] = kpis[kpis['kpi'] == 'Ramping']['value'].iloc[0] / kpi_count

                # Building level sampling and logging

                for b in range(self.building_count):

                    batch += memories[b].sample()

                    # For logging

                    base_name = f"train/b_{b+1}_"

                    # Searching by index works as after grouping the District metric goes last

                    train_log[f"{base_name}reward_avg"] = rewards[:,b].mean()
                    train_log[f"{base_name}emission_avg"] = kpis[kpis['kpi'] == 'Emissions']['value'].iloc[b] / kpi_count
                    train_log[f"{base_name}cost_avg"] = kpis[kpis['kpi'] == 'Cost']['value'].iloc[b] / kpi_count

                batch = self.transition(*zip(*batch))
                self.update_params(batch)

                eval_log = self.evaluation()

                if i_episode % self.log_interval == 0:

                    write_log(log_path=self.logs_path, log={'Episode': i_episode, **train_log, **eval_log})                    
                    pbar.set_postfix({'Episode': i_episode, **train_log, **eval_log})

                    if self.wandb_log:

                        wandb.log({**train_log, **eval_log}, step = int(wandb_step))

                # Update pbar

                pbar.update(1)

    def evaluation(self):

        done = False
        state, _ = self.eval_env.reset()

        # Add encoding if enabled

        if self.personalization:
            state = [np.concatenate((state[i][:self.building_count], self.eval_running_state[i](state[i][self.building_count:]))) for i in range(self.building_count)]

        while not done:

            action = [self.select_action(state[b], eval=True).detach().cpu().numpy() for b in range(self.building_count)]
            next_state, _, done, _, _ = self.eval_env.step(action)

            if self.personalization:
                state = [np.concatenate((next_state[i][:self.building_count], self.eval_running_state[i](next_state[i][self.building_count:]))) for i in range(self.building_count)]
            else:
                state = [self.eval_running_state[i](next_state[i]) for i in range(self.building_count)]
        
        # Logging

        eval_log = {}

        kpis = get_kpis(self.eval_env)
        rewards = np.array(pd.DataFrame(self.eval_env.unwrapped.episode_rewards)['sum'].tolist())

        #District level logging

        eval_log["eval/d_emission_avg"] = kpis[kpis['kpi'] == 'Emissions']['value'].iloc[0]
        eval_log["eval/d_cost_avg"] = kpis[kpis['kpi'] == 'Cost']['value'].iloc[0]
        eval_log["eval/d_daily_peak_avg"] = kpis[kpis['kpi'] == 'Avg. daily peak']['value'].iloc[0]
        eval_log["eval/d_load_factor_avg"] = kpis[kpis['kpi'] == '1 - load factor']['value'].iloc[0]
        eval_log["eval/d_ramping_avg"] = kpis[kpis['kpi'] == 'Ramping']['value'].iloc[0]

        # Building level sampling and logging

        for b in range(self.building_count):

            base_name = f"eval/b_{b+1}_"

            eval_log[f"{base_name}reward_avg"] = rewards[:,b].mean()
            eval_log[f"{base_name}emission_avg"] = kpis[kpis['kpi'] == 'Emissions']['value'].iloc[b + 1]
            eval_log[f"{base_name}cost_avg"] = kpis[kpis['kpi'] == 'Cost']['value'].iloc[b + 1]

        return eval_log

    def save_checkpoint(self, path : str):

        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'value_net': self.value_net.state_dict(),
        }, path)

    def load_checkpoint(self, path : str):

        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.value_net.load_state_dict(checkpoint['value_net'])


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

    parser.add_argument('--gamma', type=float, default=0.995, metavar='G', help='Discount factor (default: 0.995)')
    parser.add_argument('--tau', type=float, default=0.97, metavar='T', help='Tau (default: 0.97)')
    parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='L2', help='L2 regularization regression (default: 1e-3)')
    parser.add_argument('--max-kl', type=float, default=1e-2, metavar='Max_KL', help='Max kl value (default: 1e-2)')
    parser.add_argument('--damping', type=float, default=1e-1, metavar='D', help='Damping (default: 1e-1)')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed (default: 0)')
    parser.add_argument('--batch-size', type=int, default=15000, metavar='BS', help='Batch size (default: 15000)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='LI', help='Interval between training status logs (default: 1)')
    parser.add_argument('--training-type', type=str, default='individual', choices={'individual', 'upperbound', 'fl', 'fl-personalized'}, help='Training type (default: individual)')
    parser.add_argument('--building-no', type=int, default=0, help='Trained building (if individual building training)')
    parser.add_argument('--wandb-log', default=False, action='store_true', help='Log to wandb (default: True)')
    parser.add_argument('--data-path', type=str, default='./data/schemas/warmup/', help='data schema path')
    parser.add_argument('--device', type=str, default=None, help='device (default: None)')

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
            name=f"{args.training_type}_seed_{args.seed}_{str(int(time()))}", 
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
            },
            sync_tensorboard=False,
        )

    # Configure Environment

    DATASET_NAME = 'citylearn_challenge_2022_phase_all'
    BUILDING_COUNT = 2
    DAY_COUNT = 1
    
    ACTIVE_OBSERVATIONS = [
        'hour', 'electrical_storage_soc', 'outdoor_dry_bulb_temperature', 'outdoor_relative_humidity',
        'non_shiftable_load', 'electricity_pricing', 'carbon_intensity'
    ]

    # select buildings
    buildings = select_buildings(DATASET_NAME, BUILDING_COUNT, 0) # Fix seed to maintain building selection among different seeds for the models

    # select days
    simulation_start_time_step, simulation_end_time_step = select_simulation_period(DATASET_NAME, DAY_COUNT, 0) # Fix seed to maintain simulation period among different seeds for the models

    # initialize environment

    train_env = CityLearnEnv(
        DATASET_NAME,
        central_agent=False,
        buildings=buildings,
        active_observations=ACTIVE_OBSERVATIONS,
        simulation_start_time_step=simulation_start_time_step,
        simulation_end_time_step=simulation_end_time_step,
        # reward_function=ElectricityCostWithPenalization
        reward_function=NetElectricity
    )

    # train_env = NormalizedObservationWrapper(train_env)

    # select buildings
    buildings = select_buildings(DATASET_NAME, BUILDING_COUNT, 1)  # Fix seed to maintain building selection among different seeds for the models

    # select days
    simulation_start_time_step, simulation_end_time_step = select_simulation_period(DATASET_NAME, DAY_COUNT, 1) # Fix seed to maintain simulation period among different seeds for the models

    eval_env = CityLearnEnv(
        DATASET_NAME,
        central_agent=False,
        buildings=buildings,
        active_observations=ACTIVE_OBSERVATIONS,
        simulation_start_time_step=simulation_start_time_step,
        simulation_end_time_step=simulation_end_time_step,
        # reward_function=ElectricityCostWithPenalization
        reward_function=NetElectricity
    )

    # eval_env = NormalizedObservationWrapper(eval_env)

    # TRPO initialization

    trpo = TRPO(train_env, eval_env, device, args.gamma, args.tau, args.l2_reg, args.max_kl, args.damping, args.seed, args.batch_size, args.log_interval, args.wandb_log)

    # Training

    trpo.train()

    # Save checkpoint

    trpo.save_checkpoint(f"trpo_{args.training_type}_seed{args.seed}_{str(int(time()))}.pth")

    # Close WandB run

    if args.wandb_log:

        run.finish()

    print("Training finished")
