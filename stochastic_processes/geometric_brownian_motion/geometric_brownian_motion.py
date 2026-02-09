import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def main():
    # Parameters
    mu = +0.1  # Drift coefficient
    sigma = 0.2  # Diffusion coefficient

    X0 = 1.0  # Initial value
    T = 40.0  # Time horizon
    dt = 1e-2  # Time step

    # Perform a single simulation
    t, X = single_simulation(X0, T, dt, mu, sigma)
    # Do multiple simulations
    N_simulations = 1000
    all_X = np.zeros((N_simulations, len(t)))
    for i in range(N_simulations):
        _, X_sim = single_simulation(X0, T, dt, mu, sigma)
        all_X[i, :] = X_sim

    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    ax1.plot(t, X, color='blue')
    ax1.set_title("Geometric Brownian Motion - Single Simulation")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("X(t)")
    ax1.set_yscale('log')
    ax1.grid()

    #ax2.plot(t, all_X.T, color='lightgray', alpha=0.5)
    # plot numerical median
    numerical_median = np.median(all_X, axis=0)
    ax3.plot(t, numerical_median, color='red', label='Numerical Median')
    # Use data to plot 80% confidence interval
    confidence_level = 0.8
    lower_bound = np.percentile(all_X, (1 - confidence_level) / 2 * 100,
                                axis=0)
    upper_bound = np.percentile(all_X, (1 + confidence_level) / 2 * 100,
                                axis=0)
    ax3.fill_between(t,
                     lower_bound,
                     upper_bound,
                     color='orange',
                     alpha=0.5,
                     label='90% Confidence Interval')
    ax3.set_title(f"Geometric Brownian Motion - {N_simulations} Simulations")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("X(t)")
    ax3.grid()
    #ax2.set_ylim(0, 20)
    ax3.set_yscale('log')

    # Plot the analytical median and confidence interval
    median = median_analytical(X0, t, mu, sigma)
    lower_ci, upper_ci = confidence_interval_analytical(
        X0, t, mu, sigma, confidence_level)
    ax4.plot(t, median, color='green', label='Analytical Median')
    ax4.fill_between(t,
                     lower_ci,
                     upper_ci,
                     color='lightgreen',
                     alpha=0.5,
                     label='Analytical 90% CI')
    ax4.set_title("Analytical Median and Confidence Interval")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("X(t)")
    ax4.set_yscale('log')
    ax4.grid()

    plt.tight_layout()

    # Calculate the probability that X(T) < threshold for different time steps
    threshold = 1e0
    threshold = median_analytical(
        X0, T, mu, sigma)  # Use the analytical median as threshold
    analytical_prob = norm.cdf(
        (np.log(threshold / X0) -
         (mu - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T)))
    numerical_prob = np.mean(all_X[:, -1] < threshold)
    print(f"Numerical Probability X(T) < {threshold}: {numerical_prob:.4f}")
    print(f"Analytical Probability X(T) < {threshold}: {analytical_prob:.4f}")

    plt.show()

    return


def median_analytical(X0, t, mu, sigma):
    """Calculate the median of X(t) analytically."""
    return X0 * np.exp((mu - 0.5 * sigma**2) * t)


def confidence_interval_analytical(X0, t, mu, sigma, confidence_level=0.9):
    u = np.exp(sigma * np.sqrt(t) * norm.ppf(
        (1 + confidence_level) / 2)) * median_analytical(X0, t, mu, sigma)
    l = np.exp(sigma * np.sqrt(t) * norm.ppf(
        (1 - confidence_level) / 2)) * median_analytical(X0, t, mu, sigma)
    return l, u


def single_simulation(X0, T, dt, mu, sigma):
    """Perform a single simulation of Geometric Brownian Motion."""
    N = int(T / dt)  # Number of time steps
    t = np.linspace(0, T, N)  # Time vector
    X = np.zeros(N)  # Initialize array for X values
    X[0] = X0  # Set initial condition

    for i in range(1, N):
        X[i] = X[i - 1] + dX(X[i - 1], dt, mu, sigma)

    return t, X


def dX(X, dt, mu, sigma):
    """Compute the increment dX for Geometric Brownian Motion."""
    dW = np.sqrt(dt) * np.random.normal(0, 1)  # Brownian increment
    return mu * X * dt + sigma * X * dW


if __name__ == "__main__":
    main()
