import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import iisignature
except ImportError as e:
    raise ImportError(
        "Package iisignature is required. Install it with: pip install iisignature"
    ) from e


SEED = 123
N_PATHS = 2000

DT = 0.02
T_MAX = 5.0


X_GRID = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
X_PLOT = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])


BOND_MATURITY = 5.0


SIG_LEVEL = 2
WINDOW_LENGTH = 1.0


SIGMA_0 = 0.01
BETA = 0.30
WEIGHT_SCALE = 0.03

RESULTS_DIR = "/Users/francescoagro/Desktop/TESI/code/results/experiment7"
os.makedirs(RESULTS_DIR, exist_ok=True)


def initial_forward_curve(x):
    """
    Initial forward curve f_0(x).
    Chosen smooth and decreasing, roughly around 5%.
    """
    return 0.03 + 0.02 * np.exp(-0.25 * x)


def compute_signature_features(time_window, short_rate_window, level):
    """
    Compute signature features of Gamma_t = (t, R_t)
    over a rolling window.

    The path is piecewise-linear, so classical signatures are well-defined.
    """
    path = np.column_stack([time_window, short_rate_window])

    path[:, 0] = path[:, 0] - path[0, 0]

    sig = iisignature.sig(path, level)

    return np.concatenate([[1.0], sig])


def make_signature_weights(n_features):
    """
    Deterministic small weights for the signature-dependent volatility.
    Small weights keep the volatility stable.
    """
    rng = np.random.default_rng(SEED + 100)
    w = rng.normal(loc=0.0, scale=WEIGHT_SCALE, size=n_features)
    w[0] = 0.0
    return w


def sigma_hjm(x_grid, signature_state, weights):
    """
    Signature-dependent one-factor HJM volatility:

        sigma(x, X_t) = sigma_0 exp(-beta x) exp(w^T X_t)

    The exponential ensures positivity.
    A clipping is used only for numerical stability.
    """
    z = float(np.dot(weights, signature_state))
    z = np.clip(z, -0.5, 0.5)

    return SIGMA_0 * np.exp(-BETA * x_grid) * np.exp(z)


def hjm_drift(x_grid, sigma_values):
    """
    One-factor HJM drift in Musiela form:

        alpha(t,x) = sigma(t,x) * int_0^x sigma(t,u) du

    The integral is computed numerically on the maturity grid.
    """
    integral = np.zeros_like(x_grid)

    for j in range(1, len(x_grid)):
        integral[j] = np.trapezoid(
            sigma_values[: j + 1],
            x=x_grid[: j + 1],
        )

    return sigma_values * integral


def interpolate_curve(x_grid, curve, x_value):
    """
    Linear interpolation of f_t(x).
    """
    return np.interp(x_value, x_grid, curve)


def bond_price_from_forward_curve(x_grid, curve, tau):
    """
    Approximate

        P(t,T) = exp(- int_0^{T-t} f_t(x) dx)

    using the discretized Musiela curve.
    """
    if tau <= 0:
        return 1.0

    dense_grid = np.linspace(0.0, tau, 200)
    dense_curve = np.interp(dense_grid, x_grid, curve)
    integral = np.trapezoid(dense_curve, x=dense_grid)

    return np.exp(-integral)


def simulate_hjm():
    """
    Simulate the HJM-Musiela dynamics:

        df_t(x) = [partial_x f_t(x) + alpha(t,x)] dt
                  + sigma(t,x,X_t) dW_t

    The short rate is R_t = f_t(0).
    """
    rng = np.random.default_rng(SEED)

    times = np.arange(0.0, T_MAX + DT, DT)
    n_steps = len(times)

    curves = np.zeros((N_PATHS, n_steps, len(X_GRID)))
    short_rates = np.zeros((N_PATHS, n_steps))
    bank_accounts = np.ones((N_PATHS, n_steps))
    discounted_bonds = np.full((N_PATHS, n_steps), np.nan)

    f0 = initial_forward_curve(X_GRID)
    curves[:, 0, :] = f0
    short_rates[:, 0] = f0[0]

    dummy_time = np.linspace(0.0, WINDOW_LENGTH, 20)
    dummy_rate = np.full_like(dummy_time, f0[0])
    dummy_sig = compute_signature_features(dummy_time, dummy_rate, SIG_LEVEL)
    weights = make_signature_weights(len(dummy_sig))

    for n in range(n_steps - 1):
        t = times[n]

        dW = rng.normal(0.0, np.sqrt(DT), size=N_PATHS)

        for i in range(N_PATHS):
            current_curve = curves[i, n, :]

            start_time = max(0.0, t - WINDOW_LENGTH)
            start_idx = int(round(start_time / DT))

            time_window = times[start_idx : n + 1]
            rate_window = short_rates[i, start_idx : n + 1]

            if len(time_window) < 2:
                time_window = np.array([0.0, DT])
                rate_window = np.array([short_rates[i, n], short_rates[i, n]])

            X_sig = compute_signature_features(time_window, rate_window, SIG_LEVEL)

            sigma_values = sigma_hjm(X_GRID, X_sig, weights)
            alpha_values = hjm_drift(X_GRID, sigma_values)

            partial_x = np.zeros_like(X_GRID)
            partial_x[:-1] = np.diff(current_curve) / np.diff(X_GRID)
            partial_x[-1] = partial_x[-2]

            next_curve = current_curve + (partial_x + alpha_values) * DT + sigma_values * dW[i]

            next_curve = np.clip(next_curve, -0.02, 0.20)

            curves[i, n + 1, :] = next_curve
            short_rates[i, n + 1] = next_curve[0]

            bank_accounts[i, n + 1] = bank_accounts[i, n] * np.exp(short_rates[i, n] * DT)

        for i in range(N_PATHS):
            tau = BOND_MATURITY - times[n + 1]
            if tau >= 0:
                P_tT = bond_price_from_forward_curve(
                    X_GRID,
                    curves[i, n + 1, :],
                    tau,
                )
                discounted_bonds[i, n + 1] = P_tT / bank_accounts[i, n + 1]

    P_0T = bond_price_from_forward_curve(X_GRID, f0, BOND_MATURITY)
    discounted_bonds[:, 0] = P_0T

    return times, curves, short_rates, bank_accounts, discounted_bonds


def plot_forward_curves(times, curves):
    """
    Plot average simulated forward curves at selected times.
    """
    selected_times = [0.0, 1.0, 2.0, 3.0, 5.0]

    plt.figure(figsize=(9, 6))

    for t in selected_times:
        idx = int(round(t / DT))
        mean_curve = curves[:, idx, :].mean(axis=0)
        plt.plot(X_GRID, mean_curve, marker="o", label=f"t={t:.1f}")

    plt.xlabel("Time to maturity x")
    plt.ylabel("Forward rate f_t(x)")
    plt.title("Experiment 8: Mean simulated forward curves")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "experiment7_forward_curves.png")
    plt.savefig(path, dpi=300)
    plt.show()
    plt.close()


def plot_discounted_bond_martingale(times, discounted_bonds):
    """
    Plot the sample mean of discounted bond prices.
    """
    mean_discounted = np.nanmean(discounted_bonds, axis=0)
    std_discounted = np.nanstd(discounted_bonds, axis=0)

    valid = times <= BOND_MATURITY

    plt.figure(figsize=(9, 6))
    plt.plot(times[valid], mean_discounted[valid], label="Sample mean")
    plt.fill_between(
        times[valid],
        mean_discounted[valid] - 2.0 * std_discounted[valid] / np.sqrt(N_PATHS),
        mean_discounted[valid] + 2.0 * std_discounted[valid] / np.sqrt(N_PATHS),
        alpha=0.2,
        label="Approx. 95% CI",
    )

    plt.axhline(mean_discounted[0], linestyle="--", label="Initial value")

    plt.xlabel("Time t")
    plt.ylabel(r"Sample mean of $P(t,T)/B_t$")
    plt.title(f"Experiment 8: Discounted bond martingale diagnostic, T={BOND_MATURITY}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "experiment7_discounted_bond_martingale.png")
    plt.savefig(path, dpi=300)
    plt.show()
    plt.close()


def save_summary(times, discounted_bonds):
    """
    Save martingale diagnostic table.
    """
    valid = times <= BOND_MATURITY
    mean_discounted = np.nanmean(discounted_bonds[:, valid], axis=0)
    std_discounted = np.nanstd(discounted_bonds[:, valid], axis=0)

    df = pd.DataFrame(
        {
            "time": times[valid],
            "mean_discounted_bond": mean_discounted,
            "std_discounted_bond": std_discounted,
            "relative_deviation_from_initial": (
                mean_discounted - mean_discounted[0]
            )
            / mean_discounted[0],
        }
    )

    csv_path = os.path.join(RESULTS_DIR, "experiment7_martingale_diagnostic.csv")
    df.to_csv(csv_path, index=False)

    summary = pd.DataFrame(
        {
            "bond_maturity": [BOND_MATURITY],
            "initial_discounted_bond": [mean_discounted[0]],
            "final_mean_discounted_bond": [mean_discounted[-1]],
            "max_abs_relative_deviation": [
                np.max(np.abs(df["relative_deviation_from_initial"]))
            ],
            "mean_abs_relative_deviation": [
                np.mean(np.abs(df["relative_deviation_from_initial"]))
            ],
            "n_paths": [N_PATHS],
            "dt": [DT],
            "signature_level": [SIG_LEVEL],
            "window_length": [WINDOW_LENGTH],
        }
    )

    summary_path = os.path.join(RESULTS_DIR, "experiment7_summary.csv")
    summary.to_csv(summary_path, index=False)

    return summary


def main():
    start = time.time()

    print("\n================ Experiment 7: HJM Signature Volatility ================\n")
    print("Simulating HJM model with signature-dependent volatility...")

    times, curves, short_rates, bank_accounts, discounted_bonds = simulate_hjm()

    print(f"Curves shape: {curves.shape}")
    print(f"Short-rate paths shape: {short_rates.shape}")

    print("Creating plots...")
    plot_forward_curves(times, curves)
    plot_discounted_bond_martingale(times, discounted_bonds)

    print("Saving diagnostics...")
    summary = save_summary(times, discounted_bonds)

    print("\n================ Summary ================")
    print(summary)

    print("\nSaved results to:")
    print(RESULTS_DIR)

    print(f"\nTotal runtime: {time.time() - start:.2f} seconds")
    print("======================================================================\n")


if __name__ == "__main__":
    main()
