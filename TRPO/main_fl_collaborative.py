import json
import random
import argparse
from itertools import count
from collections import namedtuple

import scipy.optimize

import torch
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *

import sys
sys.path.append('./CityLearn/')
from citylearn.gen_citylearn import CityLearnEnv

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

building_count = 5
Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state', 'reward'))

wandb_record = True
if wandb_record:
    import wandb
    wandb.init(project="TRPO_rl_gen")
    wandb.run.name = "FL_diff"
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
parser.add_argument('--batch-size', type=int, default=15000, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--data-path', default='/home/yunxiang.li/FRL/CityLearn/citylearn/data/gen_data/', help='data schema path')
args = parser.parse_args()
schema_filepath = args.data_path+'schema.json'
eval_schema_filepath = args.data_path+'schema_eval.json'


env = CityLearnEnv(schema_filepath)

num_inputs = env.observation_space[0].shape[0] + building_count
num_actions = env.action_space[0].shape[0]
# print("num_inputs", num_inputs)
# random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)

policy_net = Policy(num_inputs, num_actions)
value_net = Value(num_inputs)

def syn_model(models):
    state_dict = models[0].state_dict()
    mean_model = {}

    for layer in state_dict.keys():
        par_mean = torch.stack([models[i].state_dict()[layer] for i in range(building_count)], axis=0).mean(axis=0)
        mean_model[layer] = par_mean.clone()
    for m in models:
        for name, params in m.named_parameters():
            params.data.copy_(mean_model[name])

def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

def select_action_eval(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, _ = policy_net(Variable(state))
    return action_mean

def update_params(batch, policy_net, value_net):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0)).unsqueeze(-1)
    states = torch.Tensor(np.array(batch.state))
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

# running_state = [ZFilter((num_inputs,), clip=5) for _ in range(building_count)]
running_state = [ZFilter((num_inputs-building_count-1,), clip=5) for _ in range(building_count)]        # no mean for price and one-hot
running_reward = [ZFilter((1,), demean=False, clip=10) for _ in range(building_count)]

def evaluation(schema_dict_eval):
    eval_env = CityLearnEnv(schema_dict_eval)
    eval_reward = np.array([0.] * building_count)

    done = False
    # load_random = [random.random()*0.2 for i in range(building_count)]
    # solar_random = [random.random()*0.1+1 for i in range(building_count)]
    temp_random = [21] * building_count
    hum_random = [51] * building_count
    state = eval_env.reset(temp_random, hum_random)
    # state = [running_state[i](state[i]) for i in range(building_count)]
    state = [np.concatenate((running_state[i](state[i][:-(building_count+1)]), state[i][-(building_count+1):])) for i in range(building_count)]
    # state = np.array([[j for j in np.hstack(encoders[i]*state[i][:-5]) if j != None] + state[i][-5:] for i in range(5)])

    while not done:
        action = [select_action_eval(state[b]).item() for b in range(building_count)]
        print("{:.2f}".format(action[0]), end=", ")
        next_state, reward, done, _ = eval_env.step(action)
        eval_reward += reward

        # state = [running_state[i](next_state[i]) for i in range(building_count)]
        state = [np.concatenate((running_state[i](next_state[i][:-(building_count+1)]), next_state[i][-(building_count+1):])) for i in range(building_count)]
        # state = np.array([[j for j in np.hstack(encoders[i]*next_state[i][:-5]) if j != None] + next_state[i][-5:] for i in range(5)])

    for b in range(building_count):
        print('evaluate reward {:.2f}'.format(eval_reward[b]/24))

    if wandb_record:
        for b in range(building_count):
            wandb.log({"eval_"+str(b+1): eval_reward[b]/24}, step = int(wandb_step))

with open(schema_filepath) as json_file:
    schema_dict = json.load(json_file)
with open(eval_schema_filepath) as json_eval_file:
    schema_dict_eval = json.load(json_eval_file)

# schema_dict["personalization"] = False
# schema_dict_eval["personalization"] = False

# for b_i in range(5):
#     if b_i == 0:
#         continue
#     schema_dict["buildings"]["Building_"+str(b_i+1)]["include"] = False
#     schema_dict_eval["buildings"]["Building_"+str(b_i+1)]["include"] = False


for i_episode in count(1):
    memories = [Memory() for _ in range(building_count)]

    reward_batch = np.array([0.] * building_count)
    num_episodes = 0

    # get data
    num_steps = 0

    while num_steps < args.batch_size:  # 15000

        temp_random = [random.random()*5+15 for i in range(building_count)]
        hum_random = [random.random()*50 for i in range(building_count)]
        state = env.reset(temp_random, hum_random)     # list of lists
        # state = [running_state[i](state[i]) for i in range(building_count)]
        state = [np.concatenate((running_state[i](state[i][:-(building_count+1)]), state[i][-(building_count+1):])) for i in range(building_count)]
        # state = np.array([[j for j in np.hstack(encoders[i]*state[i][:-5]) if j != None] + state[i][-5:] for i in range(5)])
        reward_sum = np.array([0.] * building_count)
        for t in range(10000): # Don't infinite loop while learning
            action = [select_action(state[b]).item() for b in range(building_count)]
            next_state, reward, done, _ = env.step(action)
            reward_sum += reward

            # next_state = [running_state[i](next_state[i]) for i in range(building_count)]
            next_state = [np.concatenate((running_state[i](next_state[i][:-(building_count+1)]), next_state[i][-(building_count+1):])) for i in range(building_count)]
            # next_state = np.array([[j for j in np.hstack(encoders[i]*next_state[i][:-5]) if j != None] + next_state[i][-5:] for i in range(5)])

            mask = 1
            if done:
                mask = 0

            for b in range(building_count):
                memories[b].push(state[b], np.array([action[b]]), mask, next_state[b], reward[b])

            if done:
                break

            state = next_state
        num_steps += (t-1)
        num_episodes += 1
        reward_batch += reward_sum

    # train
    wandb_step += 1     # count training step
    batch = []
    for b in range(building_count):
        reward_batch[b] /= num_episodes
        batch += memories[b].sample()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {}\tAverage reward {:.2f}'.format(
                i_episode, reward_sum[b]/24, reward_batch[b]/24))
            if wandb_record:
                wandb.log({"train_"+str(b+1): reward_sum[b]/24}, step = int(wandb_step))

    batch = Transition(*zip(*batch))
    update_params(batch, policy_net, value_net)

    evaluation(schema_dict_eval)

    if i_episode > 3000:
        break
