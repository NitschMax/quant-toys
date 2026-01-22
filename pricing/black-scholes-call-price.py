import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf


def main():
    St = 5
    sigma = 1.0
    r = 0.0

    mu = np.log(St) + r - 0.5 * sigma**2
    x = np.linspace(1e-6, 5 * St, 100)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    dist_plot(x, mu, sigma, ax1, ax2)
    call_price_plot(x, mu, sigma, St, r, ax3, ax4)

    plt.tight_layout()

    plt.show()
    plt.close()


def lognorm_dist(x, mu, sigma):
    return np.exp(-(np.log(x) - mu)**2 /
                  (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)


def option_price_at_strike(x, mu, sigma, St, r, option_type='call'):
    d_2 = (mu - np.log(x)) / sigma
    d_1 = d_2 + sigma
    if option_type == 'call':
        price = St * norm_cdf(d_1) - x * norm_cdf(d_2)
    else:
        if option_type == 'put':
            # Calculate put option price using put-call parity; put = call - stock + strike*exp(-rT)
            price = option_price_at_strike(
                x, mu, sigma, St, r, option_type='call') - St + x * np.exp(-r)
    return price


def dist_plot(x, mu, sigma, ax1, ax2):

    y = lognorm_dist(x, mu, sigma)

    ax1.plot(x, y)
    ax1.set_title('Linear Scale')
    ax1.set_xlabel('x')
    ax1.set_ylabel('Probability Density')
    ax1.grid()
    ax2.plot(x, y)
    ax2.set_yscale('log')
    ax2.set_ylim(1e-10, max(y) * 1.1)
    ax2.set_title('Logarithmic Scale')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Probability Density (log scale)')
    ax2.grid()

    # Create four plots side by side; one with linear scale and one with logarithmic scale
    # The second row of plots will later be used for option pricing
    # Mark expected value and one standard deviation points
    expected_value = np.exp(mu + 0.5 * sigma**2)
    for ax in (ax1, ax2):
        ax.axvline(expected_value,
                   color='red',
                   linestyle='--',
                   label='Expected Value')
        ax.axvline(np.exp(mu + sigma),
                   color='green',
                   linestyle='--',
                   label='+1 Std Dev')
        ax.axvline(np.exp(mu - sigma),
                   color='blue',
                   linestyle='--',
                   label='-1 Std Dev')
        ax.legend()

    return


def call_price_plot(x, mu, sigma, St, r, ax1, ax2):
    C_x = option_price_at_strike(x, mu, sigma, St, r, option_type='call')
    ax1.plot(x, C_x / St, color='orange')
    ax1.set_title('Call Option Price vs Strike Price')
    ax1.set_xlabel('Strike Price')
    ax1.set_ylabel('Call Option Price/Stock Price')
    ax1.grid()

    P_x = option_price_at_strike(x, mu, sigma, St, r, option_type='put')
    ax2.plot(x, P_x / St, color='orange')
    ax2.set_title('Put Option Price vs Strike Price')
    ax2.set_xlabel('Strike Price')
    ax2.set_ylabel('Put Option Price/Stock Price')
    ax2.grid()
    return


def norm_cdf(x):
    return (1.0 + erf(x / np.sqrt(2.0))) / 2.0


if __name__ == '__main__':
    main()
