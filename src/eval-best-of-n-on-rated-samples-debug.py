# %%
import importlib

import utils

importlib.reload(utils)

# %%
num_samples_per_problem = 16

rated_problem_solution_hierarchical_samples_path_wo_basename = f"/data/users/zhangjunlei/tyx/reward-by-prm800k/eval/rated-samples/gpt-4-generatations/{num_samples_per_problem}-samples-per-problem"

# %%
problem_solution_hierarchical_samples = utils.load_pickle(
    rated_problem_solution_hierarchical_samples_path_wo_basename + ".pkl"
)

# %%
utils.eval_best_of_n_on_rated_problem_solution_samples(
    problem_solution_hierarchical_samples,
    num_trials=2,
    num_samples_per_problem=num_samples_per_problem,
    verbose=True,
    debug={
        "n": False,
        "solution_sample": True,
    },
)
