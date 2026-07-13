import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import iisignature

from scipy.integrate import solve_ivp
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


SEEDS = [11, 22, 33, 44, 55]
N_PATHS = 12000
N_TRAIN = 10000

DT = 0.01
T_MAX = 10.0
T0 = 1.0
MATURITIES = np.array([2.0, 3.0, 5.0, 7.0, 10.0])

SIGNATURE_LEVELS = [2, 3, 4]

KAPPA = 0.5
THETA = 0.05
ALPHA = 0.3
LAMBDA = 1.0
SIGMA = 0.01
R0 = 0.05
M0 = 0.0

RIDGE_ALPHAS = np.logspace(-8, 3, 25)

RESULTS_DIR = "/Users/francescoagro/Desktop/TESI/code/results/experiment6_multiseed"
os.makedirs(RESULTS_DIR, exist_ok=True)


def simulate_volterra_paths(seed):
    rng = np.random.default_rng(seed)

    times = np.arange(0.0, T_MAX + DT, DT)
    n_steps = len(times)

    R = np.zeros((N_PATHS, n_steps))
    M = np.zeros((N_PATHS, n_steps))

    R[:, 0] = R0
    M[:, 0] = M0

    sqrt_dt = np.sqrt(DT)

    for i in range(n_steps - 1):
        dW = rng.normal(0.0, sqrt_dt, size=N_PATHS)

        R[:, i + 1] = (
            R[:, i]
            + KAPPA * (THETA - R[:, i] - ALPHA * M[:, i]) * DT
            + SIGMA * dW
        )

        M[:, i + 1] = M[:, i] + (R[:, i] - LAMBDA * M[:, i]) * DT

    return times, R, M


def solve_riccati(maturities):
    max_tau = np.max(maturities - T0)

    a = np.array([KAPPA * THETA, 0.0])
    K = np.array([
        [-KAPPA, -KAPPA * ALPHA],
        [1.0, -LAMBDA]
    ])
    e_R = np.array([1.0, 0.0])

    def ode(tau, y):
        B = y[:2]
        A = y[2]
        dB = K.T @ B - e_R
        dA = a @ B + 0.5 * SIGMA**2 * B[0]**2
        return np.array([dB[0], dB[1], dA])

    y0 = np.array([0.0, 0.0, 0.0])

    sol = solve_ivp(
        ode, t_span=(0.0, max_tau), y0=y0,
        dense_output=True, rtol=1e-10, atol=1e-12
    )

    coeffs = {}
    for T in maturities:
        tau = T - T0
        values = sol.sol(tau)
        B_R, B_M, A = values[0], values[1], values[2]
        coeffs[T] = (A, B_R, B_M)

    return coeffs


def compute_log_price_targets(R_paths, M_paths, times, maturities):
    t0_idx = np.argmin(np.abs(times - T0))
    R_t0 = R_paths[:, t0_idx]
    M_t0 = M_paths[:, t0_idx]

    coeffs = solve_riccati(maturities)

    log_prices = np.zeros((R_paths.shape[0], len(maturities)))
    for j, T in enumerate(maturities):
        A, B_R, B_M = coeffs[T]
        log_prices[:, j] = A + B_R * R_t0 + B_M * M_t0

    return log_prices


def build_time_augmented_paths(times, R_paths):
    t0_idx = np.argmin(np.abs(times - T0))
    selected_times = times[: t0_idx + 1]
    selected_R = R_paths[:, : t0_idx + 1]

    paths = np.zeros((R_paths.shape[0], len(selected_times), 2))
    paths[:, :, 0] = selected_times.reshape(1, -1)
    paths[:, :, 1] = selected_R

    return paths


def compute_signature_features(paths, level):
    n_paths = paths.shape[0]
    dim = iisignature.siglength(2, level)
    X = np.zeros((n_paths, dim))

    start = time.time()
    for i in range(n_paths):
        X[i, :] = iisignature.sig(paths[i], level)
    runtime = time.time() - start

    return X, runtime


def compute_logsignature_features(paths, level):
    n_paths = paths.shape[0]
    prep = iisignature.prepare(2, level)
    dim = iisignature.logsiglength(2, level)
    X = np.zeros((n_paths, dim))

    start = time.time()
    for i in range(n_paths):
        X[i, :] = iisignature.logsig(paths[i], prep)
    runtime = time.time() - start

    return X, runtime


def fit_and_evaluate(seed, X, Y_logP, maturities, representation, level, runtime):
    X_train = X[:N_TRAIN]
    X_test = X[N_TRAIN:]
    Y_train = Y_logP[:N_TRAIN]
    Y_test = Y_logP[N_TRAIN:]

    rows = []

    for j, T in enumerate(maturities):
        tau = T - T0

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=RIDGE_ALPHAS, cv=5))
        ])
        model.fit(X_train, Y_train[:, j])

        pred_logP = model.predict(X_test)
        true_logP = Y_test[:, j]

        rmse_logP = np.sqrt(mean_squared_error(true_logP, pred_logP))
        r2_logP = r2_score(true_logP, pred_logP)

        true_price = np.exp(true_logP)
        pred_price = np.exp(pred_logP)
        rmse_price = np.sqrt(mean_squared_error(true_price, pred_price))

        true_yield = -true_logP / tau
        pred_yield = -pred_logP / tau
        rmse_yield_bp = np.sqrt(mean_squared_error(true_yield, pred_yield)) * 10000

        selected_alpha = model.named_steps["ridge"].alpha_

        rows.append({
            "seed": seed,
            "representation": representation,
            "level": level,
            "maturity": T,
            "tau": tau,
            "n_features": X.shape[1],
            "rmse_logP": rmse_logP,
            "rmse_price": rmse_price,
            "rmse_yield_bp": rmse_yield_bp,
            "r2_logP": r2_logP,
            "selected_alpha": selected_alpha,
            "runtime": runtime,
        })

    return rows


def run_single_seed(seed):
    print(f"\n================ Experiment 7 | Seed {seed} ================\n")

    times, R_paths, M_paths = simulate_volterra_paths(seed)
    Y_logP = compute_log_price_targets(R_paths, M_paths, times, MATURITIES)
    paths = build_time_augmented_paths(times, R_paths)

    all_rows = []

    for level in SIGNATURE_LEVELS:
        X_sig, runtime_sig = compute_signature_features(paths, level)
        rows = fit_and_evaluate(
            seed, X_sig, Y_logP, MATURITIES,
            representation="signature", level=level, runtime=runtime_sig,
        )
        all_rows.extend(rows)

        X_logsig, runtime_logsig = compute_logsignature_features(paths, level)
        rows = fit_and_evaluate(
            seed, X_logsig, Y_logP, MATURITIES,
            representation="logsignature", level=level, runtime=runtime_logsig,
        )
        all_rows.extend(rows)

    return pd.DataFrame(all_rows)


def build_summary(results_df):
    summary_df = (
        results_df
        .groupby(["representation", "level", "n_features"], as_index=False)
        .agg(
            avg_rmse_logP=("rmse_logP", "mean"),
            std_rmse_logP=("rmse_logP", "std"),
            avg_rmse_yield_bp=("rmse_yield_bp", "mean"),
            std_rmse_yield_bp=("rmse_yield_bp", "std"),
            avg_r2_logP=("r2_logP", "mean"),
            avg_runtime=("runtime", "mean"),
            std_runtime=("runtime", "std"),
        )
    )
    return summary_df



def make_plots(summary_df):
    plt.figure(figsize=(9, 6))

    for representation in ["signature", "logsignature"]:
        df = summary_df[summary_df["representation"] == representation]
        df = df.sort_values("level")

        plt.errorbar(
            df["level"],
            df["avg_rmse_yield_bp"],
            yerr=df["std_rmse_yield_bp"],
            marker="o",
            capsize=5,
            label=representation,
        )

    plt.xlabel("Truncation level N")
    plt.ylabel("Average yield RMSE (bp)")
    plt.title("Experiment 7: Signature versus log-signature")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "experiment7_multiseed_yield_error_vs_level.png"), dpi=300)
    plt.show()
    plt.close()


    plt.figure(figsize=(9, 6))

    for representation in ["signature", "logsignature"]:
        df = summary_df[summary_df["representation"] == representation]
        df = df.sort_values("n_features")

        plt.errorbar(
            df["n_features"],
            df["avg_rmse_yield_bp"],
            yerr=df["std_rmse_yield_bp"],
            marker="o",
            capsize=5,
            label=representation,
        )

    plt.xlabel("Number of features")
    plt.ylabel("Average yield RMSE (bp)")
    plt.title("Experiment 7: Error versus feature dimension")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "experiment7_multiseed_error_vs_dimension.png"), dpi=300)
    plt.show()
    plt.close()


    plt.figure(figsize=(9, 6))

    for representation in ["signature", "logsignature"]:
        df = summary_df[summary_df["representation"] == representation]
        df = df.sort_values("level")

        plt.errorbar(
            df["level"],
            df["avg_runtime"],
            yerr=df["std_runtime"],
            marker="o",
            capsize=5,
            label=representation,
        )

    plt.xlabel("Truncation level N")
    plt.ylabel("Feature computation runtime (seconds)")
    plt.title("Experiment 7: Runtime comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "experiment7_multiseed_runtime.png"), dpi=300)
    plt.show()
    plt.close()



def main():
    start_time = time.time()

    print("\n================ Experiment 7: Multi-seed Signature versus Log-Signature ================\n")

    all_results = []
    for seed in SEEDS:
        seed_results = run_single_seed(seed)
        all_results.append(seed_results)

    results_df = pd.concat(all_results, ignore_index=True)
    summary_df = build_summary(results_df)

    results_path = os.path.join(RESULTS_DIR, "experiment7_multiseed_results.csv")
    summary_path = os.path.join(RESULTS_DIR, "experiment7_multiseed_summary.csv")

    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\n================ Summary Results ================")
    print(summary_df)

    print("\nSaved results to:")
    print(results_path)
    print(summary_path)

    make_plots(summary_df)

    print("\nSaved plots to:")
    print(RESULTS_DIR)

    print("\nTotal runtime:", time.time() - start_time)
    print("==========================================================================\n")


if __name__ == "__main__":
    main()