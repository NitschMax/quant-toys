import random

import matplotlib.pyplot as plt
import numpy as np


def main():
    trial_count = 4000
    martingale_arr = trial_run(trial_count)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(np.arange(len(martingale_arr)),
             martingale_arr,
             label="Martingale Value",
             color="blue")
    ax1.set_title("Martingale Process Over Time")
    ax1.set_xlabel("Trial Number")
    ax1.set_ylabel("Martingale Value")
    ax1.legend()
    ax1.grid()
    ax1.set_ylim(0, 1)

    # Create a histogram with many trial runs

    number_of_runs = 4000
    martingales = [trial_run(trial_count)[-1] for _ in range(number_of_runs)]
    # Create normalized histogram
    ax2.grid()
    ax2.hist(martingales, bins=30, density=True, alpha=0.7, color="green")
    ax2.set_title("Histogram of Final Martingale Values After Multiple Runs")
    ax2.set_xlabel("Final Martingale Value")
    ax2.set_ylabel("Density")

    plt.tight_layout()
    plt.show()
    return


def trial_run(trial_count):
    red_ball_count = 1
    total_balls = 2
    martingale = get_martingale(red_ball_count, total_balls)

    martingale_arr = [martingale]

    for _ in range(trial_count):
        # Draw ball from urn and put it back with an additional ball of the same color
        if random.random() < martingale:
            red_ball_count += 1
        total_balls += 1

        # Update martingale
        martingale = get_martingale(red_ball_count, total_balls)

        # Store values for plotting
        martingale_arr.append(martingale)

    return martingale_arr


def get_martingale(red_ball_count, total_balls):
    return red_ball_count / total_balls


if __name__ == "__main__":
    main()
