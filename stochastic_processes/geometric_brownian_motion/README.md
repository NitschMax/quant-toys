# Geometric Brownian Motion (GBM)

This project investigates the behavior of **geometric Brownian motion** and compares a simple numerical simulation with the analytical solution.
See Chapter III of Stochastic Calculus: An Introduction with Applications by G. F. Lawler for more details.

## Model

We consider the SDE

dS_t = μ S_t dt + σ S_t dB_t

with illustrative parameters:
- μ = 0.1
- σ = 0.2

## What the script does

- Simulates **1000 paths** up to **T = 40** using the **Euler–Maruyama** discretization.
- Computes, at each time point:
  - the **sample median**
  - an **80% empirical interval** (10th to 90th percentile)
- Compares these numerical summaries to the **analytical expressions** for GBM.

## Analytical reference

For GBM with initial value `S0`:
- Mean:  E[S_t] = S0 · exp(μ t)
- Median: med(S_t) = S0 · exp((μ - 0.5 σ²) t)

Also, log(S_t) is normally distributed:
log(S_t) ~ N(log(S0) + (μ - 0.5 σ²) t, σ² t)

This makes it straightforward to compute analytical quantiles and compare them to simulated percentile bands.

## Notes

- Euler–Maruyama is used here for simplicity. Since GBM has a closed form solution, one can also simulate it exactly via:
  S_t = S0 · exp((μ - 0.5 σ²) t + σ B_t)
- The goal of this toy project is to build intuition about path dispersion and the difference between typical (median) behavior and expectation (mean).

