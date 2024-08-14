import wandb
wandb.init(project="TRPO_rl_test")
wandb.run.name = "optimal_696-719_func"

# train_optimal = [-0.9406479791666659, -3.0243279408333303, -0.511524118333333, -4.5624995416666625, -0.12168762583333326]
# eval_optimal = [-0.5694304241666664, -2.906899351249997, -0.4709385033333331, -4.318719846666663, -0.09655958708333333]
# eval_optimal = [-0.92, -2.97, -0.55, -4.51, -0.21]
eval_optimal = [-0.50, -0.56, -0.88, -0.52, -1.37]

for i in range(1500):
    for b in range(5):
        wandb.log({"eval_"+str(b+1): eval_optimal[b]}, step = i)
        # wandb.log({"train_"+str(b+1): train_optimal[b]}, step = i)

