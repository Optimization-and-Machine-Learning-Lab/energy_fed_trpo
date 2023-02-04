import argparse
from itertools import count

import gym
import scipy.optimize

import torch
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *
import sys
sys.path.append('../envs')
from env import BanmaEnv
from simple_FL_env import FLEnv
from random_env import RandomBanmaEnv
from random_FL_env import RandomFLEnv

penalty = 0.5
building_count = 3

# env_name = "Humanoid-v4"
wandb_record = True
if wandb_record:
    import wandb
    wandb.init(project="TRPO_env_test")
    wandb.run.name = "TRPO_RandomFLEnv_train_step"
wandb_step = 0

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--env-name', default="Reacher-v4", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
                    help='damping (default: 1e-1)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=1500, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()
# args.env_name = env_name

# env = gym.make(args.env_name)
local_envs = [RandomFLEnv(building_id=i, action_diff_penalty=penalty) for i in [0, 1, 2]]#range(building_count)]

num_inputs = local_envs[0].obs_dim
num_actions = local_envs[0].act_dim

# env.seed(args.seed)
# torch.manual_seed(args.seed)

policy_nets = [Policy(num_inputs, num_actions) for _ in range(building_count)]
value_nets = [Value(num_inputs) for _ in range(building_count)]

def syn_model(models):
    state_dict = models[0].state_dict()
    mean_model = {}

    for layer in state_dict.keys():
        par_mean = torch.stack([models[i].state_dict()[layer] for i in range(building_count)], axis=0).mean(axis=0)
        mean_model[layer] = par_mean.clone()
    for m in models:
        for name, params in m.named_parameters():
            params.data.copy_(mean_model[name])

def select_action(state, model):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = model(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

def update_params(batch, policy_net, value_net):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(batch.state)
    values = value_net(Variable(states))

    returns = torch.Tensor(actions.size(0),1)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]       # rewards to go for each step
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]        # advantage calculated from the current network
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]       # GAE

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)

    # Original code uses the same LBFGS to optimize the value loss
    def get_value_loss(flat_params):
        set_flat_params_to(value_net, torch.Tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        values_ = value_net(Variable(states))

        value_loss = (values_ - targets).pow(2).mean()

        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * args.l2_reg
        value_loss.backward()
        return (value_loss.data.double().numpy(), get_flat_grad_from(value_net).data.double().numpy())

    # use L-BFGS-B to update value function
    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(value_net).double().numpy(), maxiter=25)
    set_flat_params_to(value_net, torch.Tensor(flat_params))

    advantages = (advantages - advantages.mean()) / advantages.std()

    action_means, action_log_stds, action_stds = policy_net(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

    def get_loss(volatile=False):
        if volatile:
            with torch.no_grad():
                action_means, action_log_stds, action_stds = policy_net(Variable(states))
        else:
            action_means, action_log_stds, action_stds = policy_net(Variable(states))
                
        log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
        action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
        return action_loss.mean()


    def get_kl():
        mean1, log_std1, std1 = policy_net(Variable(states))

        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5   # KL divergence of two normal distribution
        return kl.sum(1, keepdim=True)

    trpo_step(policy_net, get_loss, get_kl, args.max_kl, args.damping)

running_state = ZFilter((num_inputs,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)

for i_episode in count(1):
    memories = [Memory() for _ in range(building_count)]

    reward_batch = [0, 0, 0]
    num_episodes = [0, 0, 0]
    reward_sum = [0, 0, 0]

    # get data
    for b in range(building_count):
        num_steps = 0

        while num_steps < args.batch_size:  # 15000
            state = local_envs[b].reset()
            state = running_state(state)

            reward_sum[b] = 0
            for t in range(10000): # Don't infinite loop while learning
                action = select_action(state, policy_nets[b])
                action = action.data[0].numpy()
                next_state, reward, done, _ = local_envs[b].step(action)
                # wandb_step += 1
                reward_sum[b] += reward

                next_state = running_state(next_state)

                mask = 1
                if done:
                    mask = 0

                memories[b].push(state, np.array([action]), mask, next_state, reward)

                if done:
                    break

                state = next_state
            num_steps += (t-1)
            num_episodes[b] += 1
            reward_batch[b] += reward_sum[b]

    # train
    for b in range(building_count):
        reward_batch[b] /= num_episodes[b]
        batch = memories[b].sample()
        update_params(batch, policy_nets[b], value_nets[b])
        if b == 0:
            wandb_step += 1

        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {}\tAverage reward {:.2f}'.format(
                i_episode, reward_sum[b], reward_batch[b]))
            if wandb_record:
                wandb.log({"eval_"+str(b): reward_sum[b]}, step = int(wandb_step/building_count))
    
    if i_episode & 10 == 0:
        syn_model(value_nets)
        syn_model(policy_nets)
        # syn_count += 1
    if i_episode > 1000:
            break