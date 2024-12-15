#!/bin/bash
#SBATCH --job-name=fed_rl_energy_ppo
#SBATCH --output=logs/R-%x.%j.out
#SBATCH --error=logs/R-%x.%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=160G
#SBATCH --time=12:00:00  # Maximum allowed time
#SBATCH -p cscc-cpu-p
#SBATCH -q cscc-cpu-qos

# !!!!!!!!!
# !!!   Remember to load the conda environment before running the script and setting the PYTHONPATH (pp)
# !!!!!!!!!

# W&B Parameters
export WANDB_PROJECT="fed_rl_energy"
export WANDB_ENTITY="optimllab"

# Define array with experiments to launch

BUILDINGS=("2b" "5b")
DATA_TYPES=("normal" "shifted")
EXPERIMENTS=("base" "gf" "pe" "pe_gf")
SESSIONS_PER_EXPERIMENT=2

SESSION_NAMES=()

for BUILDING in "${BUILDINGS[@]}"; do

    for DATA_TYPE in "${DATA_TYPES[@]}"; do

        for EXPERIMENT in "${EXPERIMENTS[@]}"; do

            export WANDB_GROUP="$ALGO:$EXPERIMENT:$DATA_TYPE" # ALGO env var is set before launch the script

            echo "Configuring sweep for experiment: $EXPERIMENT"

            # Path to the sweep configuration file

            SWEEP_CONFIG_PATH="./config/$ALGO/$BUILDING/$DATA_TYPE/$EXPERIMENT.yaml"

            echo $SWEEP_CONFIG_PATH

            # Configure the W&B sweep and capture the sweep ID
            SWEEP_ID=$(wandb sweep $SWEEP_CONFIG_PATH 2>&1 | grep "Run sweep agent with:" | awk '{print $NF}')
        
            for i in $(seq 1 $SESSIONS_PER_EXPERIMENT); do

                SESSION_NAME="$BUILDING-$ALGO-$EXPERIMENT-$DATA_TYPE-$i"
                SESSION_NAMES+=("$SESSION_NAME")

                # !!!!!!!!!
                # !!!   Use this echo to check the command before running the sweeps
                # !!!!!!!!!
                
                # echo "tmux new-session -d -s $SESSION_NAME \"conda activate rl_energy ; pp ; wandb agent $SWEEP_ID; tmux kill-session -t $SESSION_NAME\""
                
                # !!!!!!!!!
                # !!!   Check previous comment
                # !!!!!!!!!

                # Start the W&B sweep agent in a new tmux session
		tmux new-session -d -s $SESSION_NAME "export PYTHONPATH=$(pwd) ; wandb agent $SWEEP_ID ; tmux kill-session -t $SESSION_NAME"


            done

        done

    done

done

# Wait for all tmux sessions to finish
while true; do
    all_finished=true
    for SESSION_NAME in "${SESSION_NAMES[@]}"; do
        if tmux has-session -t $SESSION_NAME 2>/dev/null; then
            all_finished=false
            break
        fi
    done
    if $all_finished; then
        break
    fi
    sleep 10  # Check every 10 seconds
done

echo "All tmux sessions have finished."

