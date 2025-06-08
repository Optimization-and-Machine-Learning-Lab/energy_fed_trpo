# Generalizing in Net-Zero Microgrids: A Study with Federated PPO and TRPO

<p>
    <a href="https://apache.org/licenses/LICENSE-2.0.txt" target="_blank">
        <img alt="License: Apache License 2.0" src="https://img.shields.io/badge/License-Apache License 2.0-yellow.svg" />
    </a>
    <img alt="Python" src="https://img.shields.io/badge/python-v3.9-green" />
    <img alt="PyTorch" src="https://img.shields.io/badge/torch-v2.5.0-important" />
    <img alt="TorchRL" src="https://img.shields.io/badge/torchrl-v0.6.0-important" />
    <img alt="Gymnasium" src="https://img.shields.io/badge/gymnasium-v0.29.1-important" />
</p>

<div style="text-align: justify">

This work addresses the challenge of optimal energy management in microgrids through a collaborative and privacy-preserving framework. We propose the FedTRPO methodology, which integrates Federated Learning (FL) and Trust Region Policy Optimization (TRPO) to manage distributed energy resources (DERs) efficiently. Using a customized version of the CityLearn environment and synthetically generated data, we simulate designed net-zero energy scenarios for microgrids composed of multiple buildings. Our approach emphasizes reducing energy costs and carbon emissions while ensuring privacy. Experimental results demonstrate that FedTRPO is comparable with state-of-the-art federated RL methodologies without hyperparameter tunning. The proposed framework highlights the feasibility of collaborative learning for achieving optimal control policies in energy systems, advancing the goals of sustainable and efficient smart grids.

</div>

## Prepare environment

- Create an environment and install requirements running:

  ```bash
  conda create --name <env> --file requirements.txt
  ```

- Make sure to define the `PYTHONPATH` environment variable running from the root folder the following command:

  ```bash
  export PYTHONPATH=$(pwd)
  ```

## Dataset

The data we used was generated using the notebooks in the folder `src/notebooks/data_generation`, particularly the notebook `naive_data_generation`. Run all the cells to generate the data and follow the instructions within the notebook to make modifications.

## Launch Experiments

The algoritms we used for our experiments (PPO, TRPO) are in the folder `src/algos/`. Each script parses arguments to launch the experiments.

### PPO

A complete explanation of the arguments:

| Argument                  | Default            | Description                                                                  |
|:--------------------------|:-------------------|:-----------------------------------------------------------------------------|
| `-wl`, `--wandb_logging`  | `False` (flag)     | Use wandb for logging                                                       |
| `-s`, `--seed`            | `0`                | Random seed for reproducibility                                             |
| `-d`, `--device`          | `"cpu"`            | Device to use for training (cpu, mps or cuda)                                    |
| `-dp`, `--data_path`      | `"data/naive_data/"`| Path to the data directory                                                  |
| `-r`, `--reward`          | `"cost"`           | Reward type to use in the environment                                       |
| `-dc`, `--day_count`      | `1`                | Number of days to simulate in the environment                               |
| `-gdi`, `--gpu_device_ix` | `0`                | GPU device index to use if device is cuda                                   |
| `-lr`, `--learning_rate`  | `1e-3`             | Learning rate for the optimizer                                             |
| `-i`, `--iterations`      | `30`               | Number of training iterations                                               |
| `-e`, `--epochs`          | `1`                | Number of epochs (training steps) per training iteration                    |
| `-dii`, `--days_in_iter`  | `10`               | Number of days per training iteration                                       |
| `-dib`, `--days_in_batch` | `2`                | Number of days per batch                                                    |
| `-pe`, `--personal_encoding` | `False` (flag) | Use personal encoding in the environment                                    |
| `-gf`, `--group_features` | `False` (flag)     | Use grouped features in the environment                                     |
| `-c`, `--clip_epsilon`    | `0.2`              | Clip value for PPO loss                                                     |
| `-g`, `--gamma`           | `1`                | Discount factor for future rewards                                          |
| `-l`, `--lmbda`           | `1`                | Lambda for generalized advantage estimation                                 |
| `-ee`, `--entropy_eps`    | `1e-3`             | Coefficient of the entropy term in the PPO loss                             |
| `-spp`, `--share_parameters_policy` | `False` (flag) | Share parameters across agents for the policy network                    |
| `-spc`, `--share_parameters_critic` | `False` (flag) | Share parameters across agents for the critic network                   |
| `-mappo`, `--multiagent_ppo` | `False` (flag) | Use multi-agent PPO                                                         |
| `-mg`, `--max_grad_norm`  | `1.0`              | Maximum norm for the gradients                                              |

You can use any of the arguments to launch or replicate the experiments in the paper. To launch the default arguments just run:

  ```python 
  # Default arguments
  python src/algos/ppo.py
  ```

To launch the hyperparameters we found and reported in the paper:

  ```python 
  # Remember we ran the experiments for seeds 0,1,2,3,4.

  # Base experiment
  python src/algos/ppo.py --clip_epsilon=0.2 --data_path=./data/naive_2_buildings_simple/ --days_in_batch=25 --days_in_iter=500 --device=cuda --entropy_eps=0.0001 --epochs=50 --iterations=30 --learning_rate=0.0001 --reward=weighted_cost_emissions --share_parameters_policy --share_parameters_critic --multiagent_ppo --seed=<seed>
  # With personal encoding
  python src/algos/ppo.py --clip_epsilon=0.2 --data_path=./data/naive_2_buildings_simple/ --days_in_batch=25 --days_in_iter=500 --device=cuda --entropy_eps=0.0001 --epochs=50 --iterations=30 --learning_rate=0.0001 --reward=weighted_cost_emissions --share_parameters_policy --share_parameters_critic --multiagent_ppo --seed=<seed> --personal_encoding
  # With group features
  python src/algos/ppo.py --clip_epsilon=0.2 --data_path=./data/naive_2_buildings_simple/ --days_in_batch=25 --days_in_iter=500 --device=cuda --entropy_eps=0.0001 --epochs=50 --iterations=30 --learning_rate=0.0001 --reward=weighted_cost_emissions  --share_parameters_policy --share_parameters_critic --multiagent_ppo --seed=<seed> --group_features
  # With personal encoding and group features
  python src/algos/ppo.py --clip_epsilon=0.2 --data_path=./data/naive_2_buildings_simple/ --days_in_batch=25 --days_in_iter=500 --device=cuda --entropy_eps=0.0001 --epochs=50 --iterations=30 --learning_rate=0.0001 --reward=weighted_cost_emissions  --share_parameters_policy --share_parameters_critic --multiagent_ppo --seed=<seed> --personal_encoding --group_features
  ```

### TRPO

A complete explanation of the arguments:

| Argument                  | Default            | Description                                                                  |
|:--------------------------|:-------------------|:-----------------------------------------------------------------------------|
| `-wl`, `--wandb_logging`  | `False` (flag)     | Use wandb for logging                                                       |
| `-s`, `--seed`            | `0`                | Random seed for reproducibility                                             |
| `-d`, `--device`          | `"cpu"`            | Device to use for training (cpu, mps, or cuda)                              |
| `-dp`, `--data_path`      | `"data/naive_data/"`| Path to the data directory                                                  |
| `-r`, `--reward`          | `"cost"`           | Reward type to use in the environment                                       |
| `-dc`, `--day_count`      | `1`                | Number of days to simulate in the environment                               |
| `-gdi`, `--gpu_device_ix` | `0`                | GPU device index to use if device is cuda                                   |
| `-i`, `--iterations`      | `30`               | Number of training iterations                                               |
| `-dii`, `--days_in_iter`  | `10`               | Number of days per training iteration                                       |
| `-dib`, `--days_in_batch` | `2`                | Number of days per batch                                                    |
| `-pe`, `--personal_encoding` | `False` (flag) | Use personal encoding in the environment                                    |
| `-gf`, `--group_features` | `False` (flag)     | Use grouped features in the environment                                     |
| `-e_p`, `--policy_epochs` | `1`                | Number of epochs (training steps) per policy training iteration             |
| `-e_c`, `--critic_epochs` | `1`                | Number of epochs (training steps) per critic training iteration             |
| `-lr`, `--learning_rate`  | `1e-3`             | Learning rate for the critic optimizer                                      |
| `-g`, `--gamma`           | `1`                | Discount factor for future rewards                                          |
| `-l`, `--lmbda`           | `1`                | Lambda for generalized advantage estimation                                 |
| `-spp`, `--share_parameters_policy` | `False` (flag) | Share parameters across agents for the policy network                    |
| `-spc`, `--share_parameters_critic` | `False` (flag) | Share parameters across agents for the critic network                   |
| `-mappo`, `--multiagent_ppo` | `False` (flag) | Use multi-agent PPO                                                         |
| `-mg`, `--max_grad_norm`  | `1.0`              | Maximum norm for the gradients                                              |
| `-itr`, `--initial_trust_radius` | `0.005`      | Initial trust region radius                                                 |
| `-mtr`, `--max_trust_radius` | `2`             | Maximum trust region radius                                                 |
| `-eta`, `--eta`           | `0.15`             | Trust region update factor                                                  |
| `-ke`, `--kappa_easy`     | `0.01`             | Trust region update factor for easy cases                                   |
| `-mni`, `--max_newton_iter` | `150`            | Maximum number of Newton iterations                                         |
| `-mkd`, `--max_krylov_dim` | `150`            | Maximum Krylov dimension                                                    |
| `-lt`, `--lanczos_tol`    | `1e-5`             | Lanczos tolerance                                                           |
| `-gt`, `--gtol`           | `1e-5`             | Gradient tolerance                                                          |
| `-ha`, `--hutchinson_approx` | `False` (flag) | Use Hutchinson approximation                                                |
| `-om`, `--opt_method`     | `"krylov"`         | Optimization method (krylov or cg)                                          |

You can use any of the arguments to launch or replicate the experiments in the paper. To launch the default arguments just run:

  ```python 
  # Default arguments
  python src/algos/trpo.py
  ```

To launch the hyperparameters we found and reported in the paper:

  ```python 
  # Remember we ran the experiments for seeds 0,1,2,3,4.

  # Base experiment
  python src/algos/trpo.py --critic_epochs=10 --data_path=./data/naive_2_buildings_simple_shifted/ --days_in_batch=500 --days_in_iter=500 --device=cuda --iterations=30 --learning_rate=0.0001 --opt_method=cg --policy_epochs=10 --reward=weighted_cost_emissions --wandb_logging --share_parameters_policy --share_parameters_critic --multiagent_ppo --seed=<seed>
  # With personal encoding
  python src/algos/trpo.py --critic_epochs=10 --data_path=./data/naive_2_buildings_simple_shifted/ --days_in_batch=500 --days_in_iter=500 --device=cuda --iterations=30 --learning_rate=0.0001 --opt_method=cg --policy_epochs=10 --reward=weighted_cost_emissions --wandb_logging --share_parameters_policy --share_parameters_critic --multiagent_ppo --seed=<seed> --personal_encoding
  # With group features
  python src/algos/trpo.py --critic_epochs=10 --data_path=./data/naive_2_buildings_simple_shifted/ --days_in_batch=500 --days_in_iter=500 --device=cuda --iterations=30 --learning_rate=0.0001 --opt_method=cg --policy_epochs=10 --reward=weighted_cost_emissions --wandb_logging --share_parameters_policy --share_parameters_critic --multiagent_ppo --seed=<seed> --group_features
  # With personal encoding and group features
  python src/algos/trpo.py --critic_epochs=10 --data_path=./data/naive_2_buildings_simple_shifted/ --days_in_batch=500 --days_in_iter=500 --device=cuda --iterations=30 --learning_rate=0.0001 --opt_method=cg --policy_epochs=10 --reward=weighted_cost_emissions --wandb_logging --share_parameters_policy --share_parameters_critic --multiagent_ppo --seed=<seed> --personal_encoding --group_features
  ```

## Common issues

### When running using the `--wandb-log` flag I've been asked for a Wandb (W&B) account

<div style="text-align: justify">
    We use <a target="_blank" href="https://wandb.ai/">wandb</a> to plot our results, you can create an account to
    visualize the progress of an execution, or just chose the option 3, it
    will ignore any logging.
</div>

  ```verbatim
    wandb: (1) Create a W&B account
    wandb: (2) Use an existing W&B account
    wandb: (3) Don\'t visualize my results
    wandb: Enter your choice: 
  ```

## Acknowledgements

Thanks to [@vchoutas](https://github.com/vchoutas) for the implementation of a PyTorch-compatible trust-region optimizer that we modified to implement a very complete version of our TRPO method.

## Show your support

Give a ⭐️ if this project helped you!

## 📝 License

This project is [Apache License 2.0](https://apache.org/licenses/LICENSE-2.0.txt) licensed.
