import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append('../standard_brownian_motion')

import standard_brownian_motion as sbm

import geometric_brownian_motion as gbm


def main():
    dt = 1e-5
    T = 1
    N = int(T / dt)
    X0 = 1.0
    mu = 1.0
    sigma = 2.0

    number_runs = 20
    X_geometric = np.zeros((number_runs, N))
    X_alternative = np.zeros((number_runs, N))
    for i in range(number_runs):
        t, X_geometric[i, :] = gbm.single_simulation(X0,
                                                     T,
                                                     dt,
                                                     mu,
                                                     sigma,
                                                     random_seed=i)
        X_alternative[i, :] = alternative_geometric(X0,
                                                    T,
                                                    dt,
                                                    mu,
                                                    sigma,
                                                    random_seed=i)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    # plot the geometric Brownian motion in grey scale with alpha=0.5
    for i in range(number_runs):
        ax1.plot(t, X_geometric[i, :], color='grey', alpha=0.5)
        ax2.plot(t,
                 X_alternative[i, :] - X_geometric[i, :],
                 color='grey',
                 alpha=0.5)

    for ax in [ax1, ax2]:
        ax.set_xlabel('Time')
        ax.set_ylabel('X(t)')
        ax.grid()

    plt.show()


def alternative_geometric(X0, T, dt, mu, sigma, random_seed=None):
    N = int(T / dt)
    B_t = sbm.perform_single_run(dt, N, random_seed=random_seed)
    X = X0 * np.exp((mu - 0.5 * sigma**2) * np.linspace(0, T, N) + sigma * B_t)

    return X


if __name__ == '__main__':
    main()
