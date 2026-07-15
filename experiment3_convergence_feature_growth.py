import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

try:
    import iisignature
except ImportError:
    raise ImportError("Install iisignature with: pip install iisignature")


SEEDS = [11, 22, 33, 44, 55]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "results", "experiment3_multiseed"))
os.makedirs(OUTPUT_DIR, exist_ok=True)


kappa = 0.5
theta = 0.05
alpha_memory = 0.3
lam = 1.0
sigma = 0.01
R0 = 0.05
M0 = 0.0

T_max = 10.0
dt = 0.01
t0 = 1.0
maturities = np.array([2.0, 3.0, 5.0, 7.0, 10.0])

n_paths = 12000
n_train = 10000

signature_orders = [1, 2, 3, 4, 5]
ridge_alphas = np.logspace(-8, 4, 25)


def simulate_volterra_paths(seed):
    np.random.seed(seed)

    n_steps = int(T_max / dt)
    times = np.linspace(0.0, T_max, n_steps + 1)

    R = np.zeros((n_paths, n_steps + 1))
    M = np.zeros((n_paths, n_steps + 1))

    R[:, 0] = R0
    M[:, 0] = M0

    dW = np.sqrt(dt) * np.random.randn(n_paths, n_steps)

    for j in range(n_steps):
        R[:, j + 1] = (
            R[:, j]
            + kappa * (theta - R[:, j] - alpha_memory * M[:, j]) * dt
            + sigma * dW[:, j]
        )
        M[:, j + 1] = M[:, j] + (R[:, j] - lam * M[:, j]) * dt

    return times, R, M



def riccati_solution(max_tau):
    a_vec = np.array([kappa * theta, 0.0])
    K = np.array([
        [-kappa, -kappa * alpha_memory],
        [1.0, -lam],
    ])
    e_R = np.array([1.0, 0.0])

    def ode(tau, y):
        B = y[:2]
        A = y[2]

        dB = K.T @ B - e_R
        dA = a_vec @ B + 0.5 * sigma**2 * B[0]**2

        return np.array([dB[0], dB[1], dA])

    sol = solve_ivp(
        ode,
        t_span=(0.0, max_tau),
        y0=np.array([0.0, 0.0, 0.0]),
        dense_output=True,
        rtol=1e-10,
        atol=1e-12,
    )

    if not sol.success:
        raise RuntimeError("Riccati ODE solver failed.")

    return sol.sol


def compute_log_price_targets(R_t0, M_t0):
    taus = maturities - t0
    sol = riccati_solution(np.max(taus))

    Y = np.zeros((len(R_t0), len(maturities)))

    for k, tau in enumerate(taus):
        vals = sol(tau)
        B_R, B_M, A = vals[0], vals[1], vals[2]
        Y[:, k] = A + B_R * R_t0 + B_M * M_t0

    return Y



def build_time_augmented_paths(times, R_paths):
    idx0 = int(t0 / dt)

    times_window = times[:idx0 + 1]
    R_window = R_paths[:, :idx0 + 1]

    paths = np.zeros((n_paths, idx0 + 1, 2))
    paths[:, :, 0] = times_window[None, :]
    paths[:, :, 1] = R_window

    return paths


def compute_signature_features(paths, order):
    features = []

    for i in range(paths.shape[0]):
        features.append(iisignature.sig(paths[i], order))

    return np.asarray(features)


def condition_number_after_scaling(X_train):
    original_stds = X_train.std(axis=0)
    non_constant_cols = original_stds > 1e-10

    X_reduced = X_train[:, non_constant_cols]

    if X_reduced.shape[1] == 0:
        return np.nan

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_reduced)

    s = np.linalg.svd(Xs, compute_uv=False)
    s = s[s > 1e-10 * s.max()]

    if len(s) == 0:
        return np.nan

    return s.max() / s.min()


def evaluate_features(X, Y, order, runtime, seed):
    X_train = X[:n_train]
    X_test = X[n_train:]

    Y_train = Y[:n_train]
    Y_test = Y[n_train:]

    cond_num = condition_number_after_scaling(X_train)

    rows = []

    for j, T in enumerate(maturities):
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=ridge_alphas, cv=5)),
        ])

        model.fit(X_train, Y_train[:, j])

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_rmse_logP = np.sqrt(mean_squared_error(Y_train[:, j], y_train_pred))
        test_rmse_logP = np.sqrt(mean_squared_error(Y_test[:, j], y_test_pred))

        tau = T - t0

        train_rmse_yield_bp = train_rmse_logP / tau * 10000
        test_rmse_yield_bp = test_rmse_logP / tau * 10000

        rows.append({
            "seed": seed,
            "signature_order": order,
            "maturity": T,
            "tau": tau,
            "n_features": X.shape[1],
            "condition_number": cond_num,
            "signature_runtime": runtime,
            "train_rmse_logP": train_rmse_logP,
            "test_rmse_logP": test_rmse_logP,
            "train_rmse_yield_bp": train_rmse_yield_bp,
            "test_rmse_yield_bp": test_rmse_yield_bp,
            "selected_alpha": model.named_steps["ridge"].alpha_,
        })

    return rows



def run_single_seed(seed):
    print(f"\n================ Running seed {seed} ================\n")

    print("Simulating paths...")
    times, R_paths, M_paths = simulate_volterra_paths(seed)

    idx0 = int(t0 / dt)
    R_t0 = R_paths[:, idx0]
    M_t0 = M_paths[:, idx0]

    print("Computing exact affine labels...")
    Y = compute_log_price_targets(R_t0, M_t0)

    print("Building time-augmented paths...")
    paths = build_time_augmented_paths(times, R_paths)

    all_rows = []

    for order in signature_orders:
        print(f"\nComputing signatures of order {order}...")

        start = time.time()
        X = compute_signature_features(paths, order)
        runtime = time.time() - start

        print("Feature matrix shape:", X.shape)
        print("Signature runtime:", runtime)

        rows = evaluate_features(
            X=X,
            Y=Y,
            order=order,
            runtime=runtime,
            seed=seed,
        )

        all_rows.extend(rows)

    return pd.DataFrame(all_rows)



def build_summary(results_df):
    summary_df = (
        results_df
        .groupby("signature_order")
        .agg(
            n_features=("n_features", "first"),

            mean_condition_number=("condition_number", "mean"),
            std_condition_number=("condition_number", "std"),

            mean_signature_runtime=("signature_runtime", "mean"),
            std_signature_runtime=("signature_runtime", "std"),

            mean_train_rmse_logP=("train_rmse_logP", "mean"),
            std_train_rmse_logP=("train_rmse_logP", "std"),

            mean_test_rmse_logP=("test_rmse_logP", "mean"),
            std_test_rmse_logP=("test_rmse_logP", "std"),

            mean_train_rmse_yield_bp=("train_rmse_yield_bp", "mean"),
            std_train_rmse_yield_bp=("train_rmse_yield_bp", "std"),

            mean_test_rmse_yield_bp=("test_rmse_yield_bp", "mean"),
            std_test_rmse_yield_bp=("test_rmse_yield_bp", "std"),

            mean_selected_alpha=("selected_alpha", "mean"),
            std_selected_alpha=("selected_alpha", "std"),
        )
        .reset_index()
    )

    return summary_df


def build_maturity_summary(results_df):
    maturity_summary_df = (
        results_df
        .groupby(["signature_order", "maturity"])
        .agg(
            n_features=("n_features", "first"),

            mean_test_rmse_yield_bp=("test_rmse_yield_bp", "mean"),
            std_test_rmse_yield_bp=("test_rmse_yield_bp", "std"),

            mean_test_rmse_logP=("test_rmse_logP", "mean"),
            std_test_rmse_logP=("test_rmse_logP", "std"),

            mean_selected_alpha=("selected_alpha", "mean"),
            std_selected_alpha=("selected_alpha", "std"),
        )
        .reset_index()
    )

    return maturity_summary_df



def make_plots(summary_df):
    # Error versus feature dimension
    plt.figure(figsize=(9, 6))

    plt.errorbar(
        summary_df["n_features"],
        summary_df["mean_test_rmse_logP"],
        yerr=summary_df["std_test_rmse_logP"],
        marker="o",
        capsize=5,
        label="Test RMSE",
    )

    plt.errorbar(
        summary_df["n_features"],
        summary_df["mean_train_rmse_logP"],
        yerr=summary_df["std_train_rmse_logP"],
        marker="o",
        linestyle="--",
        capsize=5,
        label="Train RMSE",
    )

    plt.xlabel("Number of signature features")
    plt.ylabel("Average RMSE of log bond price")
    plt.title("Experiment 3: Error versus signature dimension")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "error_vs_dimension.png"), dpi=300)
    plt.show()
    plt.close()

    plt.figure(figsize=(9, 6))

    plt.errorbar(
        summary_df["signature_order"],
        summary_df["mean_test_rmse_yield_bp"],
        yerr=summary_df["std_test_rmse_yield_bp"],
        marker="o",
        capsize=5,
    )

    plt.xlabel("Signature truncation order N")
    plt.ylabel("Average test yield RMSE (bp)")
    plt.title("Experiment 3: Yield error versus signature order")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "yield_error_vs_order.png"), dpi=300)
    plt.show()
    plt.close()

    plt.figure(figsize=(9, 6))

    plt.errorbar(
        summary_df["signature_order"],
        summary_df["mean_condition_number"],
        yerr=summary_df["std_condition_number"],
        marker="o",
        capsize=5,
    )

    plt.yscale("log")
    plt.xlabel("Signature truncation order N")
    plt.ylabel("Condition number of standardized feature matrix")
    plt.title("Experiment 3: Condition number versus signature order")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "condition_number_vs_order.png"), dpi=300)
    plt.show()
    plt.close()

    plt.figure(figsize=(9, 6))

    plt.errorbar(
        summary_df["signature_order"],
        summary_df["mean_signature_runtime"],
        yerr=summary_df["std_signature_runtime"],
        marker="o",
        capsize=5,
    )

    plt.xlabel("Signature truncation order N")
    plt.ylabel("Signature computation runtime (seconds)")
    plt.title("Experiment 3: Runtime versus signature order")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "runtime_vs_order.png"), dpi=300)
    plt.show()
    plt.close()
    


def main():
    start_total = time.time()

    print("\n================ Experiment 3: Convergence and Feature Growth ================\n")

    all_results = []

    for seed in SEEDS:
        seed_results = run_single_seed(seed)
        all_results.append(seed_results)

    results_df = pd.concat(all_results, ignore_index=True)

    summary_df = build_summary(results_df)
    maturity_summary_df = build_maturity_summary(results_df)

    results_path = os.path.join(OUTPUT_DIR, "experiment3_multiseed_results.csv")
    summary_path = os.path.join(OUTPUT_DIR, "experiment3_multiseed_summary.csv")
    maturity_summary_path = os.path.join(
        OUTPUT_DIR,
        "experiment3_multiseed_maturity_summary.csv",
    )

    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    maturity_summary_df.to_csv(maturity_summary_path, index=False)

    make_plots(summary_df)

    print("\n================ Full Multi-seed Results ================")
    print(results_df)

    print("\n================ Summary Results ================")
    print(summary_df)

    print("\n================ Maturity Summary Results ================")
    print(maturity_summary_df)

    print("\nSaved results to:")
    print(results_path)
    print(summary_path)
    print(maturity_summary_path)

    print("\nSaved plots to:")
    print(OUTPUT_DIR)

    print("\nTotal runtime:", time.time() - start_total)
    print("==========================================================================\n")


if __name__ == "__main__":
    main()
