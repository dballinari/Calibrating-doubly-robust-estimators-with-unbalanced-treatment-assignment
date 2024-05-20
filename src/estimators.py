import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from typing import Tuple


MAX_DEPTH_RANGE = [10, 20, 50, 100]
MIN_SAMPLES_LEAF_RANGE = [1, 5, 10]

def estimate_ate(y: np.ndarray, w: np.ndarray, x: np.ndarray, nfolds: int=2, hyperparameters: dict={}, **kwargs) -> float:
    # compute pseudo-outcomes
    tau, tau_under_all, tau_under, tau_winsorized, tau_reweighted, tau_trimmed = _estimate_pseudo_outcomes(y, w, x, nfolds, hyperparameters, **kwargs)
    return _get_mean_and_variance(tau), _get_mean_and_variance(tau_under_all) \
        , _get_mean_and_variance(tau_under), _get_mean_and_variance(tau_winsorized) \
        , _get_mean_and_variance(tau_reweighted), _get_mean_and_variance(tau_trimmed)


def tune_nuisances(y: np.ndarray, w: np.ndarray, x: np.ndarray, nfolds: int=2, under_sample: bool=False, **kwargs) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    if under_sample:
        y, w, x =_under_sample_majority_treatment(y, w, x)
    # find optimal hyperparameters
    hyperparameters_reg_treated = _cv_regression(x[w==1], y[w==1], nfolds, **kwargs)
    hyperparameters_reg_not_treated = _cv_regression(x[w==0], y[w==0], nfolds, **kwargs)
    hyperparameters_propensity  = _cv_classification(x, w, nfolds, **kwargs)
    return hyperparameters_reg_treated, hyperparameters_reg_not_treated, hyperparameters_propensity

def _cv_regression(x: np.ndarray, y: np.ndarray, nfolds: int, **kwargs) -> Tuple[int, int]:
    # Parallelize only the grid search, not the random forest itself
    n_jobs = kwargs.pop('n_jobs', -1)
    estimator = RandomForestRegressor(**kwargs)
    param_grid = {
        'max_depth': MAX_DEPTH_RANGE,
        'min_samples_leaf': MIN_SAMPLES_LEAF_RANGE,
    }
    grid = GridSearchCV(estimator, param_grid, cv=nfolds, n_jobs=n_jobs)
    grid.fit(x, y)
    best_params = grid.best_params_
    return best_params['min_samples_leaf'], best_params['max_depth']

def _cv_classification(x: np.ndarray, w: np.ndarray, nfolds: int, **kwargs) -> Tuple[int, int]:
    # Parallelize only the grid search, not the random forest itself
    n_jobs = kwargs.pop('n_jobs', -1)
    estimator = RandomForestClassifier(**kwargs)
    param_grid = {
        'max_depth': MAX_DEPTH_RANGE,
        'min_samples_leaf': MIN_SAMPLES_LEAF_RANGE,
    }
    grid = GridSearchCV(estimator, param_grid, cv=nfolds, n_jobs=n_jobs)
    grid.fit(x, w)
    best_params = grid.best_params_
    return best_params['min_samples_leaf'], best_params['max_depth']

def _get_mean_and_variance(x: np.ndarray) -> Tuple[float, float]:
    return (np.nanmean(x), np.nanvar(x)/np.sum(~np.isnan(x)))

def _regression_prediction(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, **kwargs) -> np.ndarray:
    # fit random forest regression
    model = RandomForestRegressor(**kwargs)
    model.fit(x_train, y_train)
    # predict outcomes
    y_pred = model.predict(x_test)
    return y_pred

def _classification_prediction(x_train: np.ndarray, w_train: np.ndarray, x_test: np.ndarray, **kwargs) -> np.ndarray:
    # fit random forest classification
    model = RandomForestClassifier(**kwargs)
    model.fit(x_train, w_train)
    # predict treatment probabilities
    w_pred = model.predict_proba(x_test)[:,1]
    return w_pred


def _init_estimation(y: np.ndarray, w: np.ndarray, x: np.ndarray, nfolds: int, under_sample: bool=False) -> np.ndarray:
    if under_sample:
        y, w, x =_under_sample_majority_treatment(y, w, x)
    n = x.shape[0]
    idx = np.random.choice(np.arange(n), size=n, replace=False)
    idx = np.array_split(idx, nfolds)
    return y, w, x, idx

def _get_folds(y: np.ndarray, w: np.ndarray, x: np.ndarray, fold_indices: np.array, fold_idx: int):
    # split sample into train and test
    idx_test = fold_indices[fold_idx]
    idx_train = np.concatenate(fold_indices[:fold_idx] + fold_indices[(fold_idx+1):])
    x_train = x[idx_train,:]
    y_train = y[idx_train]
    w_train = w[idx_train]
    x_test = x[idx_test,:]
    y_test = y[idx_test]
    w_test = w[idx_test]
    return (y_train, w_train, x_train), (y_test, w_test, x_test), idx_test


def _nan_array(shape: Tuple[int, int]) -> np.ndarray:
    nan_array = np.full(shape, np.nan, dtype=float)
    return nan_array


def _update_kwargs(kwargs: dict, hyperparameters: dict, key: str) -> dict:
    return {'max_depth': hyperparameters.get(key)[1], 
            'min_samples_leaf': hyperparameters.get(key)[0],
            **kwargs,
            }


def _estimate_pseudo_outcomes(y: np.ndarray, w: np.ndarray, x: np.ndarray, nfolds: int=2, hyperparameters: dict={}, **kwargs) -> np.ndarray:
    # initialize data for cross-fitting
    y, w, x, idx = _init_estimation(y, w, x, nfolds)
    y_under_all, w_under_all, x_under_all, idx_under_all = _init_estimation(y, w, x, nfolds, True)

    # initialize arrays for predictions with nan
    y_pred_treated = _nan_array(y.shape)
    y_pred_not_treated = _nan_array(y.shape)
    w_pred = _nan_array(y.shape)
    w_pred_under = _nan_array(y.shape)

    y_pred_treated_under_all = _nan_array(y_under_all.shape)
    y_pred_not_treated_under_all = _nan_array(y_under_all.shape)
    w_pred_under_all = _nan_array(y_under_all.shape)

    # loop over folds
    for i in range(nfolds):
        (y_train, w_train, x_train), (_, w_test, x_test), idx_test = _get_folds(y, w, x, idx, i)
        (y_train_under_all, w_train_under_all, x_train_under_all), (_, _, x_test_under_all), idx_test_under_all =\
              _get_folds(y_under_all, w_under_all, x_under_all, idx_under_all, i)
        # if train and/or test sample have no treated or no non-treated, set tau to nan
        if (np.sum(w_train==1)==0) or (np.sum(w_train==0)==0) or (np.sum(w_test==1)==0) or (np.sum(w_test==0)==0):
            continue

        # Full sample:
        # predict outcomes using data on the treated
        y_pred_treated[idx_test] = _regression_prediction(x_train[w_train==1,:], y_train[w_train==1], x_test, **_update_kwargs(kwargs, hyperparameters, 'regression_treated'))
        # predict outcomes using data on the non-treated
        y_pred_not_treated[idx_test] = _regression_prediction(x_train[w_train==0,:], y_train[w_train==0], x_test, **_update_kwargs(kwargs, hyperparameters, 'regression_not_treated'))
        # predict treatment probabilities
        w_pred[idx_test] = _classification_prediction(x_train, w_train, x_test, **_update_kwargs(kwargs, hyperparameters, 'propensity'))

        # Under-sampled data:
        # predict outcomes using data on the treated
        y_pred_treated_under_all[idx_test_under_all] = _regression_prediction(x_train_under_all[w_train_under_all==1,:], y_train_under_all[w_train_under_all==1], x_test_under_all, **_update_kwargs(kwargs, hyperparameters, 'regression_treated_under'))
        # predict outcomes using data on the non-treated
        y_pred_not_treated_under_all[idx_test_under_all] = _regression_prediction(x_train_under_all[w_train_under_all==0,:], y_train_under_all[w_train_under_all==0], x_test_under_all, **_update_kwargs(kwargs, hyperparameters, 'regression_not_treated_under'))
        # predict treatment probabilities
        w_pred_under_all[idx_test_under_all] = _classification_prediction(x_train_under_all, w_train_under_all, x_test_under_all, **_update_kwargs(kwargs, hyperparameters, 'propensity_under'))
        
        # Under-sample fitting folds:
        _, w_train_under, x_train_under =_under_sample_majority_treatment(y_train, w_train, x_train)
        # Under-sampled train data: predict treatment probabilities
        w_pred_under[idx_test] = _classification_prediction(x_train_under, w_train_under, x_test, **_update_kwargs(kwargs, hyperparameters, 'propensity_under'))

    tau = _compute_tau(y, w, y_pred_treated, y_pred_not_treated, w_pred)
    tau_under_all = _compute_tau(y_under_all, w_under_all, y_pred_treated_under_all, y_pred_not_treated_under_all, w_pred_under_all)
    tau_under = _compute_tau(y, w, y_pred_treated, y_pred_not_treated, w_pred_under, calibrate=True)
    tau_winsorized = _compute_tau(y, w, y_pred_treated, y_pred_not_treated, w_pred, tol_propensity=0.01)
    tau_reweighted = _compute_tau(y, w, y_pred_treated, y_pred_not_treated, w_pred, reweight=True)
    tau_trimmed = _compute_tau(y, w, y_pred_treated, y_pred_not_treated, w_pred, trimming=0.04)
    
    return tau, tau_under_all, tau_under, tau_winsorized, tau_reweighted, tau_trimmed

def _compute_tau(y: np.array, w: np.array, y_pred_treated: np.array, y_pred_not_treated: np.array, w_pred: np.array, calibrate: bool=False, reweight: bool=False, tol_propensity: float=1e-10, trimming: float=0.0):
    # winsorize propensity scores
    w_pred = np.minimum(np.maximum(w_pred, tol_propensity), 1-tol_propensity)
    # calibrated propensity scores when under-sampling
    if calibrate:
        ratio_treated = np.sum(w)/np.sum(1-w)
        # estimate ration of treated to non-treated
        if ratio_treated < 1:
            # correct for under-sampling of the treated
            w_pred = ratio_treated*w_pred/(ratio_treated*w_pred - w_pred + 1)
        else:
            # correct for under-sampling of the non-treated
            w_pred = w_pred/((1-w_pred)/ratio_treated - w_pred)

    # define weights
    weight_treated = w/w_pred
    weight_not_treated = (1-w)/(1-w_pred)

    # normalize weights
    if reweight:
        weight_treated = weight_treated/np.nanmean(weight_treated) 
        weight_not_treated = weight_not_treated/np.nanmean(weight_not_treated)
    # trim weights
    if trimming > 0:
        # cap the maximal weight an observation recieves to trimming
        weight_treated[weight_treated/np.nansum(weight_treated) >trimming] = trimming
        weight_not_treated[weight_not_treated/np.nansum(weight_not_treated) >trimming] = trimming
        # the remaining weights are normalized again
        weight_treated = weight_treated/np.nanmean(weight_treated) 
        weight_not_treated = weight_not_treated/np.nanmean(weight_not_treated)
    
    tau = y_pred_treated - y_pred_not_treated + weight_treated*(y-y_pred_treated) - weight_not_treated*(y-y_pred_not_treated)
    return tau

def _under_sample_majority_treatment(y: np.ndarray, w: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = x.shape[0]
    # under-sample the majority class
    n_treated = np.sum(w)
    n_not_treated = n - n_treated
    if n_treated > n_not_treated:
        # under-sample treated
        idx = np.where(w == 1)[0]
        idx = np.random.choice(idx, size=n_not_treated, replace=False)
        idx = np.concatenate((idx, np.where(w == 0)[0]))
    else:
        # under-sample not treated
        idx = np.where(w == 0)[0]
        idx = np.random.choice(idx, size=n_treated, replace=False)
        idx = np.concatenate((idx, np.where(w == 1)[0]))
    x = x[idx,:]
    y = y[idx]
    w = w[idx]
    return y, w, x
