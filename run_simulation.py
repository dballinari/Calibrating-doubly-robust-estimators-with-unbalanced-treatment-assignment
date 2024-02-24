from sklearnex import patch_sklearn
patch_sklearn()

import argparse
import numpy as np
from tqdm import tqdm
import os
import git

from src import data
from src import estimators

# parse arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--num_simulations', type=int, default=100)
argparser.add_argument('--n', type=int, default=500)
argparser.add_argument('--p', type=int, default=20)
argparser.add_argument('--sigma', type=int, default=1)
argparser.add_argument('--share_treated', type=float, default=0.1)
argparser.add_argument('--mode', type=int, default=1)
argparser.add_argument('--n_folds', type=int, default=2)
argparser.add_argument('--n_estimators', type=int, default=100)
argparser.add_argument('--seed', type=int, default=256)
argparser.add_argument('--n_jobs', type=int, default=-1)
argparser.add_argument('--num_simulations_hyperparam', type=int, default=10)
args = argparser.parse_args()


# set seed
np.random.seed(args.seed)

# list to store optimal hyperparameters
hparams_reg_treated = []; hparams_reg_not_treated = []; hparams_propensity = []
hparams_reg_treated_under = []; hparams_reg_not_treated_under = []; hparams_propensity_under = []
# add progress bar
progress_bar = tqdm(total=args.num_simulations_hyperparam, desc='Hyperparameter tuning')
for i in range(args.num_simulations_hyperparam):
    x, w, y, ate = data.simulate_data(n=args.n, p=args.p, mode=args.mode, sigma=args.sigma, share_treated=args.share_treated)
    hparams_reg_treated_i, hparams_reg_not_treated_i, hparams_propensity_i = estimators.tune_nuisances(y,w,x,args.n_folds, under_sample=False, n_estimators=args.n_estimators, random_state=args.seed, n_jobs=args.n_jobs)
    hparams_reg_treated.append(hparams_reg_treated_i)
    hparams_reg_not_treated.append(hparams_reg_not_treated_i)
    hparams_propensity.append(hparams_propensity_i)

    hparams_reg_treated_under_i, hparams_reg_not_treated_under_i, hparams_propensity_under_i = estimators.tune_nuisances(y,w,x,args.n_folds, under_sample=True, n_estimators=args.n_estimators, random_state=args.seed, n_jobs=args.n_jobs)
    hparams_reg_treated_under.append(hparams_reg_treated_under_i)
    hparams_reg_not_treated_under.append(hparams_reg_not_treated_under_i)
    hparams_propensity_under.append(hparams_propensity_under_i)
    progress_bar.update(1)

progress_bar.close()

# define optimal hyperparameters as the most frequent ones
hyperparameters = {
    'regression_treated': max(set(hparams_reg_treated), key=hparams_reg_treated.count),
    'regression_not_treated': max(set(hparams_reg_not_treated), key=hparams_reg_not_treated.count),
    'propensity': max(set(hparams_propensity), key=hparams_propensity.count),
    'regression_treated_under': max(set(hparams_reg_treated_under), key=hparams_reg_treated_under.count),
    'regression_not_treated_under': max(set(hparams_reg_not_treated_under), key=hparams_reg_not_treated_under.count),
    'propensity_under': max(set(hparams_propensity_under), key=hparams_propensity_under.count),
}
print('\nHyperparameters:')
for key, value in hyperparameters.items():
    print(f'{key}: {value}')


res_dml = np.zeros((args.num_simulations, 2))
res_under_all = np.zeros((args.num_simulations, 2))
res_under = np.zeros((args.num_simulations, 2))
res_winsorized = np.zeros((args.num_simulations, 2))
res_reweighted = np.zeros((args.num_simulations, 2))
res_trimmed = np.zeros((args.num_simulations, 2))
# add progress bar
progress_bar = tqdm(total=args.num_simulations, desc='Simulations')
for i in range(args.num_simulations):
    while True:
        x, w, y, ate = data.simulate_data(n=args.n, p=args.p, mode=args.mode, sigma=args.sigma, share_treated=args.share_treated)
        if np.sum(w) > 0 and np.sum(1-w) > 0:
            break
    res_dml[i,:], res_under_all[i,:], res_under[i,:], res_winsorized[i,:], res_reweighted[i,:], res_trimmed[i,:] = estimators.estimate_ate(y,w,x,args.n_folds, 
                                                                                                                                           hyperparameters=hyperparameters,
                                                                                                                                           n_estimators=args.n_estimators, 
                                                                                                                                           random_state=args.seed+i, 
                                                                                                                                           n_jobs=args.n_jobs, 
                                                                                                                                           )

    progress_bar.update(1)

progress_bar.close()


# create results folder if it does not exist
if not os.path.exists('results'):
    os.makedirs('results')
# define file name from input arguments
args_list = list(vars(args).items())
file_name = '__'.join([f'{k}{v}' for k,v in args_list])

np.savez(f'results/{file_name}.npz', 
        dml=res_dml, 
        under_all=res_under_all,
        under=res_under,
        winsorized=res_winsorized,
        reweighted=res_reweighted,
        trimmed=res_trimmed,
        true_ate=ate,
        simulation_settings=vars(args),
        hyperparameters=hyperparameters,
        git_hash=git.Repo(search_parent_directories=True).head.object.hexsha,
        )