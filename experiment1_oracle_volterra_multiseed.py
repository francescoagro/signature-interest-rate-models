import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import iisignature

from scipy.integrate import solve_ivp
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score



SEEDS = [11, 22, 33, 44, 55]

RESULTS_DIR = "/Users/francescoagro/Desktop/TESI/code/results/experiment1_multiseed"
os.makedirs(RESULTS_DIR, exist_ok=True)

KAPPA = 0.5
THETA = 0.05
ALPHA = 0.3
LAMBDA = 1.0
SIGMA = 0.01
R0 = 0.05
M0 = 0.0

T_MAX = 10.0
DT = 0.01
T0 = 1.0

MATURITIES = np.array([2.0, 3.0, 5.0, 7.0, 10.0])

N_PATHS = 12000
N_TRAIN = 10000
N_TEST = N_PATHS - N_TRAIN

SIGNATURE_LEVELS = [1, 2, 3, 4]

RIDGE_ALPHAS = np.logspace(-8, 2, 21)


def simulate_volterra_paths(seed):
    np.random.seed(seed)

    n_steps = int(T_MAX / DT)
    times = np.linspace(0.0, T_MAX, n_steps + 1)

    R = np.zeros((N_PATHS, n_steps + 1))
    M = np.zeros((N_PATHS, n_steps + 1))

    R[:, 0] = R0
    M[:, 0] = M0

    dW = np.sqrt(DT) * np.random.randn(N_PATHS, n_steps)

    for k in range(n_steps):
        R_k = R[:, k]
        M_k = M[:, k]

        R[:, k + 1] = (
            R_k
            + KAPPA * (THETA - R_k - ALPHA * M_k) * DT
            + SIGMA * dW[:, k]
        )

        M[:, k + 1] = M_k + (R_k - LAMBDA * M_k) * DT

    return times, R, M



def solve_affine_coefficients(tau):
    K = np.array([
        [-KAPPA, -KAPPA * ALPHA],
        [1.0, -LAMBDA]
    ])

    a = np.array([KAPPA * THETA, 0.0])
    e_R = np.array([1.0, 0.0])

    def ode(s, y):
        B = y[:2]
        A = y[2]

        dB = K.T @ B - e_R
        dA = a @ B + 0.5 * SIGMA**2 * B[0]**2

        return np.array([dB[0], dB[1], dA])

    y0 = np.array([0.0, 0.0, 0.0])

    sol = solve_ivp(
        ode,
        [0.0, tau],
        y0,
        rtol=1e-10,
        atol=1e-12
    )

    B_R, B_M, A = sol.y[:, -1]

    return A, B_R, B_M


def compute_exact_labels(R_t0, M_t0):
    logP = np.zeros((len(R_t0), len(MATURITIES)))

    for j, T in enumerate(MATURITIES):
        tau = T - T0
        A, B_R, B_M = solve_affine_coefficients(tau)
        logP[:, j] = A + B_R * R_t0 + B_M * M_t0

    yields = -logP / (MATURITIES - T0)

    return logP, yields


def logP_to_yield(logP, T):
    return -logP / (T - T0)



def build_augmented_paths(times, R_paths):
    idx_t0 = int(T0 / DT)

    selected_times = times[:idx_t0 + 1]
    selected_R = R_paths[:, :idx_t0 + 1]

    augmented = np.zeros((R_paths.shape[0], idx_t0 + 1, 2))
    augmented[:, :, 0] = selected_times[None, :]
    augmented[:, :, 1] = selected_R

    return augmented


def compute_signature_features(augmented_paths, order):
    start = time.time()

    features = np.array([
        iisignature.sig(path, order)
        for path in augmented_paths
    ])

    runtime = time.time() - start

    # For order N=1, the first coordinate is the deterministic time increment.
    # It carries no cross-sectional information. I remove it so that the
    # N=1 signature contains only R_t0 - R_0, which is linearly equivalent
    # to R_t0 because R_0 is constant.
    if order == 1:
        features = features[:, 1:].copy()

    return features, runtime



def fit_ridge_cv(X_train, y_train):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(fit_intercept=True))
    ])

    grid = {
        "ridge__alpha": RIDGE_ALPHAS
    }

    model = GridSearchCV(
        pipe,
        grid,
        cv=5,
        scoring="neg_mean_squared_error"
    )

    model.fit(X_train, y_train)

    return model.best_estimator_, model.best_params_["ridge__alpha"]


def fit_oracle_linear(X_train, y_train):
    model = LinearRegression(fit_intercept=True)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, logP_test, maturity):
    pred_logP = model.predict(X_test)

    rmse_logP = np.sqrt(mean_squared_error(logP_test, pred_logP))
    r2_logP = r2_score(logP_test, pred_logP)

    true_price = np.exp(logP_test)
    pred_price = np.exp(pred_logP)

    rmse_price = np.sqrt(mean_squared_error(true_price, pred_price))

    true_yield = logP_to_yield(logP_test, maturity)
    pred_yield = logP_to_yield(pred_logP, maturity)

    rmse_yield_bp = np.sqrt(mean_squared_error(true_yield, pred_yield)) * 10000

    return rmse_logP, rmse_price, rmse_yield_bp, r2_logP, pred_logP



def run_experiment(seed):
    print(f"\n================ Experiment 1 | Seed {seed} ================\n")

    print("Simulating paths...")
    times, R_paths, M_paths = simulate_volterra_paths(seed)

    print(f"R_paths shape: {R_paths.shape}")
    print(f"M_paths shape: {M_paths.shape}")

    idx_t0 = int(T0 / DT)

    R_t0 = R_paths[:, idx_t0]
    M_t0 = M_paths[:, idx_t0]

    print("Computing exact affine labels...")
    logP, yields = compute_exact_labels(R_t0, M_t0)

    train_idx = np.arange(N_TRAIN)
    test_idx = np.arange(N_TRAIN, N_PATHS)

    logP_train = logP[train_idx]
    logP_test = logP[test_idx]

    results = []



    print("Evaluating oracle benchmark...")

    X_oracle = np.column_stack([R_t0, M_t0])
    X_oracle_train = X_oracle[train_idx]
    X_oracle_test = X_oracle[test_idx]

    for j, T in enumerate(MATURITIES):
        model = fit_oracle_linear(X_oracle_train, logP_train[:, j])

        rmse_logP, rmse_price, rmse_yield_bp, r2_logP, _ = evaluate_model(
            model,
            X_oracle_test,
            logP_test[:, j],
            T
        )

        results.append({
            "seed": seed,
            "method": "oracle",
            "signature_order": np.nan,
            "maturity": T,
            "rmse_logP": rmse_logP,
            "rmse_price": rmse_price,
            "rmse_yield_bp": rmse_yield_bp,
            "r2_logP": r2_logP,
            "selected_alpha": 0.0,
            "n_features": 2,
            "signature_runtime": np.nan
        })



    print("Evaluating Markovian benchmark...")

    X_markovian = R_t0.reshape(-1, 1)
    X_markovian_train = X_markovian[train_idx]
    X_markovian_test = X_markovian[test_idx]

    for j, T in enumerate(MATURITIES):
        model, alpha = fit_ridge_cv(X_markovian_train, logP_train[:, j])

        rmse_logP, rmse_price, rmse_yield_bp, r2_logP, _ = evaluate_model(
            model,
            X_markovian_test,
            logP_test[:, j],
            T
        )

        results.append({
            "seed": seed,
            "method": "markovian",
            "signature_order": np.nan,
            "maturity": T,
            "rmse_logP": rmse_logP,
            "rmse_price": rmse_price,
            "rmse_yield_bp": rmse_yield_bp,
            "r2_logP": r2_logP,
            "selected_alpha": alpha,
            "n_features": 1,
            "signature_runtime": np.nan
        })


    print("Building time-augmented paths...")
    augmented_paths = build_augmented_paths(times, R_paths)

    for order in SIGNATURE_LEVELS:
        print(f"Computing signatures of order {order}...")

        X_sig, runtime = compute_signature_features(augmented_paths, order)

        print(f"Signature feature matrix shape: {X_sig.shape}")
        print(f"Signature runtime: {runtime}")

        if order == 1:
            corr = np.corrcoef(R_t0, X_sig[:, 0])[0, 1]
            print(f"Diagnostic correlation R_t0 vs N=1 signature feature: {corr}")

        X_sig_train = X_sig[train_idx]
        X_sig_test = X_sig[test_idx]

        for j, T in enumerate(MATURITIES):
            model, alpha = fit_ridge_cv(X_sig_train, logP_train[:, j])

            rmse_logP, rmse_price, rmse_yield_bp, r2_logP, _ = evaluate_model(
                model,
                X_sig_test,
                logP_test[:, j],
                T
            )

            results.append({
                "seed": seed,
                "method": "signature",
                "signature_order": order,
                "maturity": T,
                "rmse_logP": rmse_logP,
                "rmse_price": rmse_price,
                "rmse_yield_bp": rmse_yield_bp,
                "r2_logP": r2_logP,
                "selected_alpha": alpha,
                "n_features": X_sig.shape[1],
                "signature_runtime": runtime
            })

    return pd.DataFrame(results)



def aggregate_by_method_order(results_df):
    summary = (
        results_df
        .groupby(["method", "signature_order"], dropna=False)
        .agg(
            mean_rmse_logP=("rmse_logP", "mean"),
            std_rmse_logP=("rmse_logP", "std"),

            mean_rmse_price=("rmse_price", "mean"),
            std_rmse_price=("rmse_price", "std"),

            mean_rmse_yield_bp=("rmse_yield_bp", "mean"),
            std_rmse_yield_bp=("rmse_yield_bp", "std"),

            mean_r2_logP=("r2_logP", "mean"),
            std_r2_logP=("r2_logP", "std"),

            mean_selected_alpha=("selected_alpha", "mean"),
            std_selected_alpha=("selected_alpha", "std"),

            n_features=("n_features", "first"),
            mean_signature_runtime=("signature_runtime", "mean"),
            std_signature_runtime=("signature_runtime", "std")
        )
        .reset_index()
    )

    return summary


def aggregate_by_method_order_maturity(results_df):
    summary = (
        results_df
        .groupby(["method", "signature_order", "maturity"], dropna=False)
        .agg(
            mean_rmse_logP=("rmse_logP", "mean"),
            std_rmse_logP=("rmse_logP", "std"),

            mean_rmse_price=("rmse_price", "mean"),
            std_rmse_price=("rmse_price", "std"),

            mean_rmse_yield_bp=("rmse_yield_bp", "mean"),
            std_rmse_yield_bp=("rmse_yield_bp", "std"),

            mean_r2_logP=("r2_logP", "mean"),
            std_r2_logP=("r2_logP", "std"),

            mean_selected_alpha=("selected_alpha", "mean"),
            std_selected_alpha=("selected_alpha", "std"),

            n_features=("n_features", "first"),
            mean_signature_runtime=("signature_runtime", "mean"),
            std_signature_runtime=("signature_runtime", "std")
        )
        .reset_index()
    )

    return summary



def create_multiseed_plots(results_df, summary_df):
    plot_rmse_vs_signature_order(summary_df)
    plot_yield_error_by_maturity(results_df)


def plot_rmse_vs_signature_order(summary_df):
    sig = summary_df[summary_df["method"] == "signature"].copy()

    oracle_avg = summary_df[summary_df["method"] == "oracle"]["mean_rmse_logP"].iloc[0]
    markovian_avg = summary_df[summary_df["method"] == "markovian"]["mean_rmse_logP"].iloc[0]

    plt.figure(figsize=(9, 6))

    plt.errorbar(
        sig["signature_order"],
        sig["mean_rmse_logP"],
        yerr=sig["std_rmse_logP"],
        marker="o",
        capsize=5,
        label="Signature"
    )

    plt.axhline(oracle_avg, linestyle="--", label="Oracle")
    plt.axhline(markovian_avg, linestyle="--", label="Markovian")

    plt.xlabel("Signature truncation order")
    plt.ylabel("Average RMSE of log bond price")
    plt.title("Experiment 1: Pricing error versus signature order")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(RESULTS_DIR, "multiseed_rmse_logP_vs_signature_order.png"), dpi=300)
    plt.show()


def plot_yield_error_by_maturity(results_df):
    maturity_summary = aggregate_by_method_order_maturity(results_df)

    plt.figure(figsize=(10, 6))

    plot_specs = [
        ("oracle", np.nan, "Oracle"),
        ("markovian", np.nan, "Markovian"),
        ("signature", 3.0, "Signature N=3"),
    ]

    for method, order, label in plot_specs:
        if np.isnan(order):
            df = maturity_summary[
                (maturity_summary["method"] == method)
                & (maturity_summary["signature_order"].isna())
            ]
        else:
            df = maturity_summary[
                (maturity_summary["method"] == method)
                & (maturity_summary["signature_order"] == order)
            ]

        plt.errorbar(
            df["maturity"],
            df["mean_rmse_yield_bp"],
            yerr=df["std_rmse_yield_bp"],
            marker="o",
            capsize=5,
            label=label
        )

    plt.xlabel("Maturity")
    plt.ylabel("Yield RMSE (basis points)")
    plt.title("Experiment 1: Yield error by maturity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join(RESULTS_DIR, "multiseed_yield_rmse_by_maturity.png"), dpi=300)
    plt.show()



def main():
    total_start = time.time()

    print("\n================ Experiment 1: Multi-seed Oracle Volterra Model ================\n")

    all_results = []

    for seed in SEEDS:
        seed_results = run_experiment(seed)
        all_results.append(seed_results)

    results_df = pd.concat(all_results, ignore_index=True)

    summary_df = aggregate_by_method_order(results_df)
    maturity_summary_df = aggregate_by_method_order_maturity(results_df)

    print("\n================ Multi-seed Full Results ================")
    print(results_df)

    print("\n================ Multi-seed Summary Results ================")
    print(summary_df)

    print("\n================ Multi-seed Maturity Summary Results ================")
    print(maturity_summary_df)

    results_path = os.path.join(RESULTS_DIR, "experiment1_multiseed_results.csv")
    summary_path = os.path.join(RESULTS_DIR, "experiment1_multiseed_summary.csv")
    maturity_summary_path = os.path.join(RESULTS_DIR, "experiment1_multiseed_maturity_summary.csv")

    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    maturity_summary_df.to_csv(maturity_summary_path, index=False)

    create_multiseed_plots(results_df, summary_df)

    print("\nSaved tables to:")
    print(results_path)
    print(summary_path)
    print(maturity_summary_path)

    print("\nSaved plots to:")
    print(RESULTS_DIR)

    print(f"\nTotal runtime: {time.time() - total_start:.2f} seconds")
    print("=====================================================================\n")


if __name__ == "__main__":
    main()