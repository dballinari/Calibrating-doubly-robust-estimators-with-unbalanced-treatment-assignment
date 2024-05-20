import numpy as np
from scipy import stats
from typing import Tuple
from sklearn.linear_model import LogisticRegression
import pandas as pd


class SyntheticData:
     """
     Class to generate synthetic data building on the simulation settings used in Kuenzel et. al (2019) [https://www.pnas.org/doi/epdf/10.1073/pnas.1804597116],
     Friedmann (2019) [https://projecteuclid.org/journals/annals-of-statistics/volume-19/issue-1/Multivariate-Adaptive-Regression-Splines/10.1214/aos/1176347963.full], and
     Nie and Wager (2017) [https://arxiv.org/abs/1712.04912]. The true ATE is 1.

     Attributes:
          p (int): The number of covariates
          sigma (float): The noise level

     """


     _MIN_COVARIATES = 6
     _ATE = 1

     def __init__(self, p: int=None, sigma: float=1):
          if p is None:
               p = self._MIN_COVARIATES
          
          if p < self._MIN_COVARIATES:
               raise ValueError(f"Number of covariates must be at least {self._MIN_COVARIATES}")

          self.p = p  # number of covariates
          self.sigma = sigma  # noise level

     def generate_sample(self, n: int, share_treated: float=0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
          """
          Generate a random sample of size n with a given share of treated observations.

          Args:
               n (int): The sample size
               share_treated (float): The share of treated observations in the sample

          Returns:
               np.ndarray: The covariates
               np.ndarray: The treatment variable
               np.ndarray: The outcome variable
          """
          # simulate nxp covariates from a uniform distribution
          x = np.random.uniform(size=(n, self.p))
          # define propensity score as in Kuenzel et. al (2019) [https://www.pnas.org/doi/epdf/10.1073/pnas.1804597116] and assign treatment
          alpha = share_treated*21/31
          e = alpha*(1+stats.beta.cdf(np.min(x[:,:2], axis=1), 2, 4))
          w = np.random.binomial(1, e)
          # baseline effect is the scaled Friedmann (2019) function  [https://projecteuclid.org/journals/annals-of-statistics/volume-19/issue-1/Multivariate-Adaptive-Regression-Splines/10.1214/aos/1176347963.full]
          b = np.sin(np.pi*x[:,0]*x[:,1]) + 2*(x[:,2]-0.5)**2 + x[:,3] + 0.5*x[:,4]
          # individual treatment effects as in set-up A in Nie and Wager (2017) [https://arxiv.org/abs/1712.04912]
          tau = x[:,0]+x[:,1]
          y = b + (w-0.5) * tau + self.sigma * np.random.normal(size=x.shape[0])
          return x, w, y
     
     def get_true_ate(self) -> float:
          return self._ATE
        




class EmpiricalMonteCarlo:
     """
     Class to perform an empirical Monte Carlo simulation as proposed by Huber et. al. (2013, https://www.sciencedirect.com/science/article/abs/pii/S0304407613000390)
     and Lechner and Wunsch (2013, https://www.sciencedirect.com/science/article/abs/pii/S0927537113000134). The true ATE and ITE are 0.

     Attributes:
          data (pd.DataFrame): The empirical data to be used for the simulation
          y_col (str): The column name of the outcome variable
          w_col (str): The column name of the treatment variable
          X_cols (list): The list of column names of the covariates
          propensity_model (sklearn.linear_model.LogisticRegression): The fitted propensity model
          data_non_treated (pd.DataFrame): The subset of the data where the treatment is 0 

     """
     
     _ATE = 0
     _MIN_PROPENSITY = 0.05

     def __init__(self, data: pd.DataFrame, y_col: str, w_col: str, X_cols: list):
          """
          Constructor of the EmpiricalMonteCarlo class

          Args:
               data (pd.DataFrame): The empirical data to be used for the simulation
               y_col (str): The column name of the outcome variable
               w_col (str): The column name of the treatment variable
               X_cols (list): The list of column names of the covariates
          """

          self.data = data
          self.y_col = y_col
          self.w_col = w_col
          self.X_cols = X_cols
          self._initialize_components()


     def _initialize_components(self):
          self.data_non_treated = self.data.loc[self.data[self.w_col]==0].copy()
          self.mean_x = self.data[self.X_cols].mean()
          self.std_x = self.data[self.X_cols].std()
          # fit a logistic regression model to predict the treatment assignment
          self.propensity_model = LogisticRegression(penalty=None, solver='lbfgs')
          self.propensity_model.fit((self.data[self.X_cols]-self.mean_x)/self.std_x, self.data[self.w_col])
          # remove observations with extreme propensity scores
          p = self.propensity_model.predict_proba((self.data_non_treated[self.X_cols]-self.mean_x)/self.std_x)[:,1]
          self.data_non_treated = self.data_non_treated.loc[(p >= self._MIN_PROPENSITY) & (p <= 1-self._MIN_PROPENSITY)]
          # compute the average propensity score of the non-treated observations
          self.non_treated_avg_propensity = self.propensity_model.predict_proba((self.data_non_treated[self.X_cols]-self.mean_x)/self.std_x)[:,1].mean()
        

     def generate_sample(self, n: int, share_treated: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
          """
          Generate a random sample of size n with a given share of treated observations. The treatment assignment is based on the propensity scores.

          Args:
               n (int): The sample size
               share_treated (float): The share of treated observations in the sample

          Returns:
               np.ndarray: The covariates
               np.ndarray: The treatment variable
               np.ndarray: The outcome variable
          """

          # randomly draw with replacement from the non-treated observations
          random_data = self.data_non_treated.sample(n=n, replace=True)
          # compute the propensity scores for the random sample
          propensity_scores = self.propensity_model.predict_proba((random_data[self.X_cols]-self.mean_x)/self.std_x)[:,1]
          # define a calibration factor 
          scaling_factor = self.non_treated_avg_propensity / share_treated
          # assign treatment based on the propensity scores
          random_data[self.w_col] = (propensity_scores/scaling_factor > np.random.uniform(size=n)).astype(int)
          return random_data[self.X_cols].values, random_data[self.w_col].values, random_data[self.y_col].values
     
     def get_true_ate(self) -> float:
          return self._ATE
    