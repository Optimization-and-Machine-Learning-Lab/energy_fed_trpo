# run simple TRPO on all five buildings, plot a baseline
import json
import random
import argparse
from itertools import count

import gym
import scipy.optimize

import torch
from models import *
from replay_memory import Memory
from torch.autograd import Variable
from trpo import trpo_step
from utils import *

import sys
sys.path.append('../CityLearn/')
from citylearn.my_citylearn import CityLearnEnv

np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})

wandb_record = True
if wandb_record:
    import wandb
    wandb.init(project="TRPO_rl")
    wandb.run.name = "baseline_train_0_test_0_func"
wandb_step = 0
selection_action_step = 0

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
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
parser.add_argument('--batch-size', type=int, default=15000, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--building-no', type=int, default=4,
                    help='trained building')
args = parser.parse_args()
schema_filepath = '/home/yunxiang.li/FRL/CityLearn/citylearn/data/my_data/schema.json'
eval_schema_filepath = '/home/yunxiang.li/FRL/CityLearn/citylearn/data/my_data/schema_eval.json'


def select_action(state, policy_net):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    # global selection_action_step
    # if wandb_record:
    #     wandb.log({"std": action_std}, step = int(selection_action_step))
    # selection_action_step += 1
    action = torch.normal(action_mean, action_std)
    return action

def select_action_eval(state, policy_net):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, _ = policy_net(Variable(state))
    return action_mean

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

def evaluation(schema_dict_eval, encoder, b):
    eval_env = CityLearnEnv(schema_dict_eval)
    eval_reward = 0.

    done = False
    state = eval_env.reset()
    # state = [running_state[i](state[i][:-building_count]) for i in range(building_count)]
    state = np.hstack(encoder*state[0])

    while not done:
        action = select_action_eval(state, policy_net).data[0].numpy()
        print("{:.2f}".format(action.item()), end=", ")
        next_state, reward, done, _ = eval_env.step(action)
        eval_reward += reward[0]

        # state = [running_state[i](next_state[i][:-building_count]) for i in range(building_count)]
        state = np.hstack(encoder*next_state[0])
    print()

    print('evaluate reward {:.2f}'.format(eval_reward/24))

    if wandb_record:
        wandb.log({"eval_"+str(b+1): eval_reward/24}, step = int(wandb_step))


with open(schema_filepath) as json_file:
    schema_dict = json.load(json_file)
with open(eval_schema_filepath) as json_eval_file:
    schema_dict_eval = json.load(json_eval_file)

schema_dict["personalization"] = False
schema_dict_eval["personalization"] = False

for b_i in range(5):
    if b_i == args.building_no:
        continue
    schema_dict["buildings"]["Building_"+str(b_i+1)]["include"] = False
    schema_dict_eval["buildings"]["Building_"+str(b_i+1)]["include"] = False

env = CityLearnEnv(schema_filepath)
building = env.buildings[0]
encoder = np.array(building.observation_encoders)

num_inputs = env.observation_space[0].shape[0] + 2
num_actions = env.action_space[0].shape[0]

# random.seed(args.seed)
# env.seed(args.seed)
# torch.manual_seed(args.seed)

policy_net = Policy(num_inputs, num_actions)
# print(num_inputs)
value_net = Value(num_inputs)

for i_episode in count(1):
    memory = Memory()

    num_steps = 0
    reward_batch = 0
    num_episodes = 0
    while num_steps < args.batch_size:
        env = CityLearnEnv(schema_dict)

        state = env.reset()
        # print("initial state")
        # print(state)
        # state = [running_state[i](state[i][:-building_count]) for i in range(building_count)]
        state = np.hstack(encoder*state[0])

        reward_sum = 0
        for t in range(10000): # Don't infinite loop while learning
            # print(state)
            # print()
            action = select_action(state, policy_net).data[0].numpy()
            next_state, reward, done, _ = env.step(action)
            # print("initial state")
            # print(next_state)
            reward_sum += reward[0]

            next_state = np.hstack(encoder*next_state[0])
            # next_state = [running_state[i](next_state[i]) for i in range(building_count)]

            mask = 1
            if done:
                mask = 0

            memory.push(state, np.array(action), mask, next_state, reward[0])

            if done:
                # exit()
                break

            state = next_state
        num_steps += (t-1)
        num_episodes += 1
        reward_batch += reward_sum

    wandb_step += 1     # count training step
    batch = []
    reward_batch /= num_episodes
    batch = memory.sample_batch()
    update_params(batch, policy_net, value_net)

    if i_episode % args.log_interval == 0:
        print('Episode {}\tLast reward: {}\tAverage reward {:.2f}'.format(
            i_episode, reward_sum/24, reward_batch/24))
        if wandb_record:
            wandb.log({"train_"+str(args.building_no+1): reward_sum/24}, step = int(wandb_step))
        
    evaluation(schema_dict_eval, encoder, args.building_no)

    if i_episode > 1500:
        break
