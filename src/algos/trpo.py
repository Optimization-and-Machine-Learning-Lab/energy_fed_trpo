
# Torch
import torch

# Tensordict modules
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor

# Data collection
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage

# Multi-agent network
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from src.custom.models import MultiAgentMLP as CustomMultiAgentMLP

# Loss
from torchrl.objectives import PPOLoss, ValueEstimators

# Utils

import copy
import argparse

from pathlib import Path
from tqdm import tqdm
from src.utils import GeneralLogger
from src.utils.torchtrustncg import TrustRegion
from src.utils.cl_torchrl_helper import plot_rewards_and_actions

# Configs

from src.algos.config import get_exp_envs

def create_probabilistic_policy(env, share_parameters_policy=True, device="cpu", group_features=False):

    # First: define a neural network n_obs_per_agent -> 2 * n_actions_per_agents (mean and std)

    policy_net = torch.nn.Sequential(
        CustomMultiAgentMLP(
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
            n_agent_inputs_names=list(env.cl_env.unwrapped.buildings[0].observations(normalize=True, periodic_normalization=True).keys()),
            n_agent_outputs=2 * env.action_spec.shape[-1],
            n_agents=env.n_agents,
            centralised=False,
            share_params=share_parameters_policy,
            device=device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        ) 
        if group_features else 
        MultiAgentMLP(
            n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
            n_agent_outputs=2 * env.action_spec.shape[-1],
            n_agents=env.n_agents,
            centralised=False,
            share_params=share_parameters_policy,
            device=device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        ),
        NormalParamExtractor(),
    )

    # Second: wrap the neural network in a TensordictModule

    policy_module = TensorDictModule(
        policy_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )

    # Third: define the probabilistic policy

    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.unbatched_action_spec,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.unbatched_action_spec[env.action_key].space.low,
            "high": env.unbatched_action_spec[env.action_key].space.high,
        },
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"),
    )  # we'll need the log-prob for the PPO loss

    return policy

def create_critic(env, share_parameters_critic=True, mappo=True, device="cpu", group_features=False):

    if mappo: # In case of MAPPO (multi-agent PPO), we concatenate all names as they are centralized
        n_agent_inputs_names = [
            name 
            for names in [list(b.observations(normalize=True, periodic_normalization=True).keys())
            for b in env.cl_env.unwrapped.buildings]
            for name in names
        ]
    else:
        n_agent_inputs_names = list(env.cl_env.unwrapped.buildings[0].observations(normalize=True, periodic_normalization=True).keys())


    critic_net = CustomMultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_inputs_names=n_agent_inputs_names,
        n_agent_outputs=1,  # 1 value per agent
        n_agents=env.n_agents,
        centralised=mappo,
        share_params=share_parameters_critic,
        device=device,
        depth=2,
        num_cells=256,
        activation_class=torch.nn.Tanh,
    ) if group_features else MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=1,  # 1 value per agent
        n_agents=env.n_agents,
        centralised=mappo,
        share_params=share_parameters_critic,
        device=device,
        depth=2,
        num_cells=256,
        activation_class=torch.nn.Tanh,
    )

    critic = TensorDictModule(
        module=critic_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "state_value")],
    )
    
    return critic

def sample_data(env, policy, device, frames_per_batch, n_iters):
    total_frames = frames_per_batch * n_iters
    collector = SyncDataCollector(
        env,
        policy,
        device=device,
        storing_device=device,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
    )
    return collector

def create_replay_buffer(frames_per_batch, device, minibatch_size):
    return ReplayBuffer(
        storage=LazyTensorStorage(
            frames_per_batch, device=device
        ),  # We store the frames_per_batch collected at each iteration
        sampler=SamplerWithoutReplacement(),
        batch_size=minibatch_size,  # We will sample minibatches of this size
    )

def create_loss_module(policy, critic, env, gamma, lmbda):
    loss_module = PPOLoss( # This loss function computes the same surrogate objective we need for TRPO
        actor_network=policy,
        critic_network=critic,
        normalize_advantage=False,  # Important to avoid normalizing across the agent dimension
    )
    loss_module.set_keys(  # We have to tell the loss where to find the keys
        reward=env.reward_key,
        action=env.action_key,
        sample_log_prob=("agents", "sample_log_prob"),
        value=("agents", "state_value"),
        # These last 2 keys will be expanded to match the reward shape
        done=("agents", "done"),
        terminated=("agents", "terminated"),
    )

    loss_module.make_value_estimator(
        ValueEstimators.GAE, gamma=gamma, lmbda=lmbda
    )  # We build GAE
    
    return loss_module

def train_policy(
    policy, env, eval_env, n_iters, local_epochs_policy, local_epochs_critic, loss_module, replay_buffer,
    frames_per_batch, minibatch_size, max_grad_norm, optim_policy, optim_critic, logger
):

    summary = {
        "train": {
            "reward": [],
            "loss_objective": [],
            "loss_critic": [],
            "loss_entropy": [],
            "cost": [],
            "cost_without_storage": [],
            "emissions": [],
            "emissions_without_storage": [],
        },
        "eval": {
            "reward": [],
            "cost": [],
            "cost_without_storage": [],
            "emissions": [],
            "emissions_without_storage": [],
        }
    }

    best_policy = None
    best_eval_reward = -float("inf")

    GAE = loss_module.value_estimator

    try:
        with tqdm(total=n_iters, nrows=10, desc="episode: 0, reward_mean: 0, eval_reward_mean: 0") as pbar:

            episode = 0

            for episode in range(n_iters):

                # Collect data

                for tensordict_data in sample_data(
                    env=env, policy=policy, device=policy.device, frames_per_batch=minibatch_size, n_iters=1
                ):

                    tensordict_data.set(
                        ("next", "agents", "done"),
                        tensordict_data.get(("next", "done"))
                        .unsqueeze(-1)
                        .repeat(1, 1, env.n_agents)
                        .unsqueeze(-1)
                        .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
                    )
                    tensordict_data.set(
                        ("next", "agents", "terminated"),
                        tensordict_data.get(("next", "terminated"))
                        .unsqueeze(-1)
                        .repeat(1, 1, env.n_agents)
                        .unsqueeze(-1)
                        .expand(tensordict_data.get_item_shape(("next", env.reward_key))),
                    )

                    # We need to expand the done and terminated to match the reward shape (this is expected by the value estimator)

                    with torch.no_grad():
                        GAE(
                            tensordict_data,
                            params=loss_module.critic_network_params,
                            target_params=loss_module.target_critic_network_params,
                        )  # Compute GAE and add it to the data

                        data_view = tensordict_data.reshape(-1)  # Flatten the batch size to shuffle data
                        replay_buffer.extend(data_view)

                        # Get train metrics

                        done = tensordict_data.get(("next", "agents", "done"))

                        # Compute mean reward

                        episode_reward_mean = (
                            tensordict_data.get(("next", "agents", "episode_reward"))[done].mean().item()
                        )

                        # Compute mean cost and emissions

                        cost = tensordict_data.get(
                            ("next", "agents", "info", "cost")
                        )[done].mean().item()
                        cost_without_storage = tensordict_data.get(
                            ("next", "agents", "info", "cost_without_storage")
                        )[done].mean().item()
                        emissions = tensordict_data.get(
                            ("next", "agents", "info", "emissions")
                        )[done].mean().item()
                        emissions_without_storage = tensordict_data.get(
                            ("next", "agents", "info", "emissions_without_storage")
                        )[done].mean().item()

                        # Evaluating

                        policy.eval()

                        eval_collector = sample_data(
                            env=eval_env, policy=policy, device=policy.device, frames_per_batch=minibatch_size, n_iters=1
                        )

                        for rollout in eval_collector:

                            done = rollout.get(
                                ("next", "done")
                            ).unsqueeze(-1).repeat(1, 1, env.n_agents).unsqueeze(-1).expand(rollout.get_item_shape(("next", env.reward_key)))

                            # Compute mean reward

                            episode_reward_mean_eval = rollout.get(("next", "agents", "episode_reward"))[done].mean().item()

                            # Compute mean cost and emissions

                            cost_eval = rollout.get(
                                ("next", "agents", "info", "cost")
                            )[done].mean().item()
                            cost_without_storage_eval = rollout.get(
                                ("next", "agents", "info", "cost_without_storage")
                            )[done].mean().item()
                            emissions_eval = rollout.get(
                                ("next", "agents", "info", "emissions")
                            )[done].mean().item()
                            emissions_without_storage_eval = rollout.get(
                                ("next", "agents", "info", "emissions_without_storage")
                            )[done].mean().item()

                            # Add to the output

                            summary["eval"]["reward"].append(episode_reward_mean_eval)
                            summary["eval"]["cost"].append(cost_eval)
                            summary["eval"]["cost_without_storage"].append(cost_without_storage_eval)
                            summary["eval"]["emissions"].append(emissions_eval)
                            summary["eval"]["emissions_without_storage"].append(emissions_without_storage_eval)

                        policy.train()

                    # Save best policy

                    if episode_reward_mean_eval > best_eval_reward:

                        best_eval_reward = episode_reward_mean_eval
                        best_policy = copy.deepcopy(policy)

                    # Run local steps

                    loss_metrics = {
                        "train/loss_objective": 0,
                        "train/loss_critic": 0,
                        "train/loss_entropy": 0,
                    }

                    for _ in range(frames_per_batch // minibatch_size):
                        
                        subdata = replay_buffer.sample()

                        for _ in range(local_epochs_policy):

                            # Optimize policy
                            
                            def closure(backward=True):

                                if torch.is_grad_enabled() and backward:
                                    optim_policy.zero_grad()

                                # Return elements needed to compute the gradients

                                if backward:
                                    return loss_module, subdata
                                    # loss_vals["loss_objective"].backward(retain_graph=True, create_graph=True)
                                
                                # No gradients needed

                                loss_vals = loss_module(subdata)

                                return loss_vals["loss_objective"]
                                # return loss_vals["loss_objective"], loss_module._log_weight, subdata
                                # return loss_module, subdata
                            
                            optim_policy.step(closure=closure)

                            # Optimize critic

                            for _ in range(local_epochs_critic):
                        
                                loss_vals = loss_module(subdata)

                                loss_vals["loss_critic"].backward()
                                optim_critic.step()
                                optim_critic.zero_grad()

                            try:
                                torch.nn.utils.clip_grad_norm_(
                                    loss_module.parameters(), max_grad_norm, error_if_nonfinite=True
                                )  # Optional
                            except RuntimeError as e:
                                print(f"Gradient clipping error: {e}")
                                return best_policy, summary

                            # Accumulate loss values
                            loss_metrics["train/loss_objective"] += loss_vals["loss_objective"].item()
                            loss_metrics["train/loss_critic"] += loss_vals["loss_critic"].item()
                            loss_metrics["train/loss_entropy"] += loss_vals["loss_entropy"].item()

                    # Compute mean loss values

                    num_policy_updates = local_epochs_policy * (frames_per_batch // minibatch_size)
                    num_critic_updates = local_epochs_critic * (frames_per_batch // minibatch_size)
                    loss_metrics["train/loss_objective"] /= num_policy_updates
                    loss_metrics["train/loss_critic"] /= num_critic_updates
                    loss_metrics["train/loss_entropy"] /= num_policy_updates

                    # Add to the output

                    summary["train"]["reward"].append(episode_reward_mean)
                    summary["train"]["loss_objective"].append(loss_metrics["train/loss_objective"])
                    summary["train"]["loss_critic"].append(loss_metrics["train/loss_critic"])
                    summary["train"]["loss_entropy"].append(loss_metrics["train/loss_entropy"])
                    summary["train"]["cost"].append(cost)
                    summary["train"]["cost_without_storage"].append(cost_without_storage)
                    summary["train"]["emissions"].append(emissions)
                    summary["train"]["emissions_without_storage"].append(emissions_without_storage)

                    # Log with the experiment logger

                    logger.log(
                        {
                            **loss_metrics,
                            "train/reward_mean": episode_reward_mean,
                            "train/cost_mean": cost,
                            "train/cost_without_storage": cost_without_storage,
                            "train/emissions_mean": emissions,
                            "train/emissions_without_storage": emissions_without_storage,
                            "eval/reward_mean": episode_reward_mean_eval,
                            "eval/cost_mean": cost_eval,
                            "eval/cost_without_storage": cost_without_storage_eval,
                            "eval/emissions_mean": emissions_eval,
                            "eval/emissions_without_storage": emissions_without_storage_eval
                        },
                        step=episode,
                    )

                pbar.set_description(f"episode: {episode}, reward_mean: {episode_reward_mean}, eval_reward_mean: {episode_reward_mean_eval}")
                pbar.update()

    except KeyboardInterrupt:
        print("Training interrupted. Returning the best policy and summary so far.")

    return best_policy, summary

def parse_args():

    parser = argparse.ArgumentParser(description='CityLearn TRPO experiment')

    # Training parameters

    parser.add_argument("-wl", "--wandb_logging", action="store_true", help="Use wandb for logging")
    parser.add_argument("-s", "--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("-d", "--device", type=str, default="cpu", help="Device to use for training (cpu or cuda)")
    parser.add_argument("-dp", "--data_path", type=str, default="data/naive_data/", help="Path to the data directory")
    parser.add_argument("-r", "--reward", type=str, default="cost", help="Reward type to use in the environment")
    parser.add_argument("-dc", "--day_count", type=int, default=1, help="Number of days to simulate in the environment")
    parser.add_argument("-gdi", "--gpu_device_ix", type=int, default=0, help="GPU device index to use if device is cuda")
    parser.add_argument("-i", "--iterations", type=int, default=30, help="Number of training iterations")
    parser.add_argument("-dii", "--days_in_iter", type=int, default=10, help="Number of days per training iteration")
    parser.add_argument("-dib", "--days_in_batch", type=int, default=2, help="Number of days per batch")
    parser.add_argument("-pe", "--personal_encoding", action="store_true", help="Use personal encoding in the environment")
    parser.add_argument("-gf", "--group_features", action="store_true", help="Use grouped features in the environment")
    parser.add_argument("-e_p", "--policy_epochs", type=int, default=1, help="Number of epochs (training steps) per policy training iteration")
    parser.add_argument("-e_c", "--critic_epochs", type=int, default=1, help="Number of epochs (training steps) per critic training iteration")
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3, help="Learning rate for the critic optimizer")

    # TRPO parameters

    parser.add_argument("-g", "--gamma", type=float, default=1, help="Discount factor for future rewards")
    parser.add_argument("-l", "--lmbda", type=float, default=1, help="Lambda for generalized advantage estimation")
    parser.add_argument("-spp", "--share_parameters_policy", action="store_true", help="Share parameters across agents for the policy network")
    parser.add_argument("-spc", "--share_parameters_critic", action="store_true", help="Share parameters across agents for the critic network")
    parser.add_argument("-mappo", "--multiagent_ppo", action="store_true", help="Use multi-agent PPO")
    parser.add_argument("-mg", "--max_grad_norm", type=float, default=1.0, help="Maximum norm for the gradients")

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    # Parse arguments

    args = parse_args()

    frames_per_batch = args.days_in_iter * 24  # Number of team frames collected per training iteration
    minibatch_size = args.days_in_batch * 24  # Size of the mini-batches in each optimization step

    # Configure logger

    exp_config = {**vars(args)}
    exp_config.pop('wandb_logging', None)

    logging_path = (
        f"./logs/trpo_s_{args.seed}_r_{args.reward}_pe_{args.personal_encoding}_"
        f"spp_{args.share_parameters_policy}_spc_{args.share_parameters_critic}_mappo_{args.multiagent_ppo}/"
    )

    Path(logging_path).mkdir(exist_ok=True)

    logger = GeneralLogger()
    logger.setup(
        config={
            "wdb_log": args.wandb_logging,
            "csv_log": True,
            "console_log": False,
            "exp_config": exp_config,
            "logging_path": logging_path
        }
    )

    # Create environments

    train_env, eval_env, device = get_exp_envs(
        data_path=args.data_path,
        reward=args.reward,
        seed=args.seed,
        day_count=args.day_count,
        device=args.device,
        gpu_device_ix=args.gpu_device_ix,
        personal_encoding=args.personal_encoding
    )

    # Create networks
    policy = create_probabilistic_policy(
        env=train_env, share_parameters_policy=args.share_parameters_policy, device=device, group_features=args.group_features
    )
    critic = create_critic(
        env=train_env, share_parameters_critic=args.share_parameters_critic, mappo=args.multiagent_ppo, device=device,
        group_features=args.group_features
    )

    logger.watch_model([policy, critic])

    # Configure data collection
    collector = sample_data(env=train_env, policy=policy, device=device, frames_per_batch=frames_per_batch, n_iters=args.iterations)

    # Create replay buffer
    replay_buffer = create_replay_buffer(frames_per_batch=frames_per_batch, device=device, minibatch_size=minibatch_size)

    # Create loss module
    loss_module = create_loss_module(
        policy=policy, critic=critic, env=train_env, gamma=args.gamma, lmbda=args.lmbda
    )

    # Create optimizers
    optim_policy = TrustRegion(
        loss_module.actor_network.parameters(), max_trust_radius=2, initial_trust_radius=.005,
        eta=0.15, kappa_easy=0.01, max_newton_iter=150, max_krylov_dim=150,
        lanczos_tol=1e-5, gtol=1e-05, hutchinson_approx=True,
        opt_method='krylov'
    )
    optim_critic = torch.optim.Adam(params=loss_module.parameters(), lr=args.learning_rate)

    # Train policy
    policy, summary = train_policy(
        env=train_env,
        policy=policy,
        eval_env=eval_env,
        n_iters=args.iterations,
        collector=collector,
        loss_module=loss_module,
        replay_buffer=replay_buffer,
        local_epochs_policy=args.policy_epochs,
        local_epochs_critic=args.critic_epochs,
        frames_per_batch=frames_per_batch,
        minibatch_size=minibatch_size,
        max_grad_norm=args.max_grad_norm,
        optim_policy=optim_policy,
        optim_critic=optim_critic,
        logger=logger
    )

    # Plot rewards and actions with the correct methods

    plot_rewards_and_actions(
        policy=policy,
        train_env=train_env,
        eval_env=eval_env,
        summary=summary,
        save=True,
        save_path=logging_path,
    )

    # Close the logger

    logger.finish()