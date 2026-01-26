import matplotlib.pyplot as plt
import numpy as np


def main():
    delta_t = 1e-2
    T = 1.0
    num_steps = int(T / delta_t) + 1
    drift = 0.0  # No drift for standard Brownian motion

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    # Generate standard Brownian motion
    time = np.linspace(0, delta_t * (num_steps - 1), num_steps)
    B_t = perform_single_run(delta_t, num_steps, drift)
    plot_single_run(fig, ax1, time, B_t)

    multiple_runs = 1000
    all_B_t = np.zeros((multiple_runs, num_steps))
    for run in range(multiple_runs):
        all_B_t[run] = perform_single_run(delta_t, num_steps, drift)

    # Plot mean and all single runs in ax2 in light gray
    for run in range(multiple_runs):
        ax2.plot(time, all_B_t[run], color='lightgray', alpha=0.1)
    mean_B_t = np.mean(all_B_t, axis=0)
    ax2.plot(time, mean_B_t, label=f"Mean of {multiple_runs} runs")
    ax2.set_title(f"{multiple_runs} Runs of Standard Brownian Motion")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("B(t)")
    ax2.grid()

    # Plot mean and analytical standard deviation in ax4
    ax4.plot(time, mean_B_t, label=f"Mean of {multiple_runs} runs")
    ax4.plot(time,
             drift * time + np.sqrt(time),
             'r--',
             label="Analytical Std Dev: √t")

    #Plot standard deviation as shaded area
    std_B_t = np.std(all_B_t, axis=0)
    ax4.fill_between(time,
                     mean_B_t - std_B_t,
                     mean_B_t + std_B_t,
                     color='gray',
                     alpha=0.5,
                     label="±1 Std Dev")

    ax4.set_title(
        f"Mean and Standard deviation of Brownian Motion over {multiple_runs} Runs"
    )
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Mean B(t)")
    ax4.grid()

    # Use the 1000 runs to estimate the probability of B(T) < threshold in dependence of the time
    threshold = 2.0
    # Create new array that holds B(t) < threshold for each run and time step
    count_below_threshold = np.sum(all_B_t[:, :] < threshold,
                                   axis=0) / multiple_runs
    ax3.plot(time, count_below_threshold, label="Estimated Probability")
    ax3.set_title(
        f"Estimated Probability of B(t) > {threshold} over {multiple_runs} Runs"
    )
    ax3.set_xlabel("Time")
    ax3.set_ylabel(f"P(B(t) < {threshold})")
    ax3.grid()

    # Analytical probability calculation
    from scipy.stats import norm
    analytical_prob = norm.cdf(2 / np.sqrt(time + 1e-10))
    ax3.plot(time, analytical_prob, 'r--', label="Analytical Probability")

    fig.tight_layout()
    plt.show()


def perform_single_run(delta_t, num_steps, drift=0.0):
    B_t = np.zeros(num_steps)
    for i in range(1, num_steps):
        B_t[i] = B_t[
            i - 1] + np.sqrt(delta_t) * np.random.normal() + drift * delta_t
    return B_t


def plot_single_run(fig, ax, time, B_t):
    ax.plot(time, B_t, label="Standard Brownian Motion")
    ax.set_title("Single Run of Standard Brownian Motion")
    ax.set_xlabel("Time")
    ax.set_ylabel("B(t)")
    ax.grid()
    return


if __name__ == "__main__":
    main()
