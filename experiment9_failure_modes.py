import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import iisignature

from scipy.integrate import solve_ivp
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


SEEDS = [11, 22, 33, 44, 55]
N_PATHS = 12000
N_TRAIN = 10000
N_TEST = 2000

DT = 0.01
T_MAX = 10.0
T0 = 1.0
MATURITIES = np.array([2.0, 3.0, 5.0, 7.0, 10.0])


KAPPA = 0.5
THETA = 0.05
ALPHA = 0.3
LAMBDA = 0.1
SIGMA = 0.01
R0 = 0.05
M0 = 0.0

SIGNATURE_LEVELS = [1, 2, 3, 4, 5, 6]
RIDGE_ALPHAS = np.logspace(-8, 4, 25)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "results", "experiment9_multiseed"))
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

    for j in range(n_steps - 1):
        dW = rng.normal(0.0, sqrt_dt, size=N_PATHS)

        drift_R = KAPPA * (THETA - R[:, j] - ALPHA * M[:, j])
        drift_M = R[:, j] - LAMBDA * M[:, j]

        R[:, j + 1] = R[:, j] + drift_R * DT + SIGMA * dW
        M[:, j + 1] = M[:, j] + drift_M * DT

    return times, R, M


def solve_riccati():
    taus = MATURITIES - T0
    tau_max = float(np.max(taus))

    a = np.array([KAPPA * THETA, 0.0])
    K = np.array([
        [-KAPPA, -KAPPA * ALPHA],
        [1.0, -LAMBDA],
    ])
    e_R = np.array([1.0, 0.0])

    def ode(tau, y):
        B = y[:2]
        A = y[2]

        dB = K.T @ B - e_R
        dA = a @ B + 0.5 * SIGMA**2 * B[0] ** 2

        return np.array([dB[0], dB[1], dA])

    sol = solve_ivp(
        ode,
        t_span=(0.0, tau_max),
        y0=np.zeros(3),
        t_eval=taus,
        rtol=1e-10,
        atol=1e-12,
    )

    B_R = sol.y[0]
    B_M = sol.y[1]
    A = sol.y[2]

    return A, B_R, B_M


def compute_log_prices(R_t0, M_t0):
    A, B_R, B_M = solve_riccati()

    logP = np.zeros((len(R_t0), len(MATURITIES)))

    for j in range(len(MATURITIES)):
        logP[:, j] = A[j] + B_R[j] * R_t0 + B_M[j] * M_t0

    return logP


def log_prices_to_yields(logP, tau):
    return -logP / tau


def build_time_augmented_paths(times, R_paths):
    t0_idx = int(round(T0 / DT))
    selected_times = times[: t0_idx + 1]

    paths = np.zeros((R_paths.shape[0], len(selected_times), 2))
    paths[:, :, 0] = selected_times[None, :]
    paths[:, :, 1] = R_paths[:, : t0_idx + 1]

    return paths


def compute_signature_features(paths, level):
    features = []
    for i in range(paths.shape[0]):
        sig = iisignature.sig(paths[i], level)
        features.append(sig)

    return np.asarray(features)


def stable_condition_number(X):
    """
    Computes the condition number after standardization.
    Constant columns are removed to avoid artificial singularity.
    """
    std = X.std(axis=0)
    X_reduced = X[:, std > 1e-12]

    if X_reduced.shape[1] == 0:
        return np.nan

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reduced)

    s = np.linalg.svd(X_scaled, compute_uv=False)

    if s[-1] < 1e-14:
        return np.inf

    return s[0] / s[-1]


def fit_and_evaluate(seed, X, Y_logP):
    train_idx = np.arange(N_TRAIN)
    test_idx = np.arange(N_TRAIN, N_TRAIN + N_TEST)

    X_train = X[train_idx]
    X_test = X[test_idx]

    Y_train = Y_logP[train_idx]
    Y_test = Y_logP[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rows = []

    for j, T in enumerate(MATURITIES):
        model = RidgeCV(alphas=RIDGE_ALPHAS, cv=5)
        model.fit(X_train_scaled, Y_train[:, j])

        pred_train = model.predict(X_train_scaled)
        pred_test = model.predict(X_test_scaled)

        train_rmse_logP = np.sqrt(mean_squared_error(Y_train[:, j], pred_train))
        test_rmse_logP = np.sqrt(mean_squared_error(Y_test[:, j], pred_test))

        tau = T - T0
        true_yield = log_prices_to_yields(Y_test[:, [j]], tau)[:, 0]
        pred_yield = log_prices_to_yields(pred_test.reshape(-1, 1), tau)[:, 0]

        yield_rmse_bp = np.sqrt(mean_squared_error(true_yield, pred_yield)) * 10000

        rows.append({
            "seed": seed,
            "maturity": T,
            "tau": tau,
            "train_rmse_logP": train_rmse_logP,
            "test_rmse_logP": test_rmse_logP,
            "test_r2_logP": r2_score(Y_test[:, j], pred_test),
            "test_rmse_yield_bp": yield_rmse_bp,
            "selected_alpha": model.alpha_,
        })

    return rows


def run_single_seed(seed):
    print(f"\n================ Experiment 9 | Seed {seed} ================\n")

    times, R_paths, M_paths = simulate_volterra_paths(seed)

    t0_idx = int(round(T0 / DT))
    R_t0 = R_paths[:, t0_idx]
    M_t0 = M_paths[:, t0_idx]

    Y_logP = compute_log_prices(R_t0, M_t0)
    augmented_paths = build_time_augmented_paths(times, R_paths)

    all_rows = []

    for N in SIGNATURE_LEVELS:
        print(f"Computing signatures of order {N}...")

        start = time.time()
        X_sig = compute_signature_features(augmented_paths, N)
        runtime = time.time() - start

        cond_number = stable_condition_number(X_sig)

        rows = fit_and_evaluate(seed, X_sig, Y_logP)

        for row in rows:
            row["signature_order"] = N
            row["n_features"] = X_sig.shape[1]
            row["condition_number"] = cond_number
            row["signature_runtime"] = runtime
            row["lambda"] = LAMBDA
            all_rows.append(row)

    return pd.DataFrame(all_rows)


def build_summary(results_df):
    summary_df = (
        results_df
        .groupby(["signature_order", "n_features"], as_index=False)
        .agg(
            avg_train_rmse_logP=("train_rmse_logP", "mean"),
            std_train_rmse_logP=("train_rmse_logP", "std"),
            avg_test_rmse_logP=("test_rmse_logP", "mean"),
            std_test_rmse_logP=("test_rmse_logP", "std"),
            avg_test_rmse_yield_bp=("test_rmse_yield_bp", "mean"),
            std_test_rmse_yield_bp=("test_rmse_yield_bp", "std"),
            avg_condition_number=("condition_number", "mean"),
            std_condition_number=("condition_number", "std"),
            avg_signature_runtime=("signature_runtime", "mean"),
            std_signature_runtime=("signature_runtime", "std"),
            avg_selected_alpha=("selected_alpha", "mean"),
            std_selected_alpha=("selected_alpha", "std"),
        )
        .sort_values("signature_order")
        .reset_index(drop=True)
    )
    return summary_df



def plot_error_vs_level(summary):
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        summary["signature_order"],
        summary["avg_test_rmse_yield_bp"],
        yerr=summary["std_test_rmse_yield_bp"],
        marker="o",
        capsize=5,
    )
    plt.yscale("log")
    plt.xlabel("Signature truncation order N")
    plt.ylabel("Average test yield RMSE (bp)")
    plt.title("Experiment 9: Long-memory error versus signature order")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "experiment9_multiseed_error_vs_level.png"), dpi=300)
    plt.show()
    plt.close()


def plot_error_vs_dimension(summary):
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        summary["n_features"],
        summary["avg_test_rmse_yield_bp"],
        yerr=summary["std_test_rmse_yield_bp"],
        marker="o",
        capsize=5,
    )
    plt.yscale("log")
    plt.xlabel("Number of signature features")
    plt.ylabel("Average test yield RMSE (bp)")
    plt.title("Experiment 9: Error versus feature dimension")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "experiment9_multiseed_error_vs_dimension.png"), dpi=300)
    plt.show()
    plt.close()


def plot_condition_number(summary):
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        summary["signature_order"],
        summary["avg_condition_number"],
        yerr=summary["std_condition_number"],
        marker="o",
        capsize=5,
    )
    plt.yscale("log")
    plt.xlabel("Signature truncation order N")
    plt.ylabel("Condition number")
    plt.title("Experiment 9: Conditioning versus signature order")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "experiment9_multiseed_condition_number.png"), dpi=300)
    plt.show()
    plt.close()


def plot_runtime(summary):
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        summary["signature_order"],
        summary["avg_signature_runtime"],
        yerr=summary["std_signature_runtime"],
        marker="o",
        capsize=5,
    )
    plt.xlabel("Signature truncation order N")
    plt.ylabel("Signature computation runtime (seconds)")
    plt.title("Experiment 9: Runtime versus signature order")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "experiment9_multiseed_runtime.png"), dpi=300)
    plt.show()
    plt.close()



def main():
    start_time = time.time()

    print("\n================ Experiment 9: Multi-seed Failure Modes ================\n")

    all_results = []
    for seed in SEEDS:
        seed_results = run_single_seed(seed)
        all_results.append(seed_results)

    results_df = pd.concat(all_results, ignore_index=True)
    summary_df = build_summary(results_df)

    results_path = os.path.join(RESULTS_DIR, "experiment9_multiseed_results.csv")
    summary_path = os.path.join(RESULTS_DIR, "experiment9_multiseed_summary.csv")

    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\n================ Summary Results ================")
    print(summary_df)

    print("\nSaved results to:")
    print(results_path)
    print(summary_path)

    plot_error_vs_level(summary_df)
    plot_error_vs_dimension(summary_df)
    plot_condition_number(summary_df)
    plot_runtime(summary_df)

    print("\nSaved plots to:")
    print(RESULTS_DIR)

    print("\nTotal runtime:", time.time() - start_time)
    print("=============================================================\n")


if __name__ == "__main__":
    main()
