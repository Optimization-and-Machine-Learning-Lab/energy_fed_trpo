seed=0
training_type=upperbound

(trap 'kill 0' SIGINT; python TRPO/main.py --building-no 0 --seed $seed --wandb-log --training-type $training_type & python TRPO/main.py --building-no 1 --seed $seed --wandb-log --training-type $training_type & python TRPO/main.py --building-no 2 --seed $seed --wandb-log --training-type $training_type & python TRPO/main.py --building-no 3 --seed $seed --wandb-log --training-type $training_type & python TRPO/main.py --building-no 4 --seed $seed --wandb-log --training-type $training_type & wait)