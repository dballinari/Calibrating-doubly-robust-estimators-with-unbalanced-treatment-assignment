# Calibrating doubly-robust estimators with unbalanced treatment assignment - Simulation
Code to replicate the simulation study in the paper ["Calibrating doubly-robust estimators with unbalanced treatment assignment"](https://arxiv.org/abs/2403.01585).

The code implements two different simulations strategies:
- **Simulation 1**: Synthetic DGP
- **Simulation 2**: Empirical Monte Carlo Study

When the argument `--data_path` is `None`, the code will run the synthetic DGP simulation. Otherwise, it will run the empirical Monte Carlo study. The data used in the empirical Monte Carlo study is obtained from https://doi.org/10.23662/FORS-DS-1203-1.
