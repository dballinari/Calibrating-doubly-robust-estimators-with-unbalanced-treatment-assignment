import numpy as np
from scipy import stats
from scipy.special import expit, logit
from typing import Tuple

# Define constants
MIN_COVARIATES = 6

def simulate_data(n: int, p: int, mode: int=1, sigma: float=1, share_treated: float=0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if p < MIN_COVARIATES:
         raise ValueError(f"Number of covariates must be at least {MIN_COVARIATES}")

    dgps = {
         1: _dgp1,
         2: _dgp2,
    }
    if mode not in dgps.keys():
         raise ValueError(f"Mode {mode} not recognized")
    return dgps[mode](n, p, sigma, share_treated)


def _dgp1(n: int, p: int, sigma: float=1, share_treated: float=0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    # simulate nxp covariates from a uniform distribution
    x = np.random.uniform(size=(n, p))
    # define propensity score as in Kuenzel et. al (2019) [https://www.pnas.org/doi/epdf/10.1073/pnas.1804597116] and assign treatment
    alpha = share_treated*21/31
    e = alpha*(1+stats.beta.cdf(np.min(x[:,:2], axis=1), 2, 4))
    w = np.random.binomial(1, e)
    # baseline effect is the scaled Friedmann (2019) function  [https://projecteuclid.org/journals/annals-of-statistics/volume-19/issue-1/Multivariate-Adaptive-Regression-Splines/10.1214/aos/1176347963.full]
    b = np.sin(np.pi*x[:,0]*x[:,1]) + 2*(x[:,2]-0.5)**2 + x[:,3] + 0.5*x[:,4]
    # individual treatment effects as in set-up A in Nie and Wager (2017) [https://arxiv.org/abs/1712.04912]
    tau = x[:,0]+x[:,1]
    y = b + (w-0.5) * tau + sigma * np.random.normal(size=x.shape[0])
    ate = 1
    return x, w, y, ate

def _dgp2(n: int, p: int, sigma: float=1, share_treated: float=0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    # simulate nxp covariates from a uniform distribution
    x = np.random.uniform(size=(n, p))
    # define propensity score (experimental setting)
    w = np.random.binomial(1, share_treated)
    # baseline effect is the scaled Friedmann (2019) function  [https://projecteuclid.org/journals/annals-of-statistics/volume-19/issue-1/Multivariate-Adaptive-Regression-Splines/10.1214/aos/1176347963.full]
    b = np.sin(np.pi*x[:,0]*x[:,1]) + 2*(x[:,2]-0.5)**2 + x[:,3] + 0.5*x[:,4]
    # individual treatment effects as in set-up A in Nie and Wager (2017) [https://arxiv.org/abs/1712.04912]
    tau = x[:,0]+x[:,1]
    y = b + (w-0.5) * tau + sigma * np.random.normal(size=x.shape[0])
    ate = 1
    return x, w, y, ate
    