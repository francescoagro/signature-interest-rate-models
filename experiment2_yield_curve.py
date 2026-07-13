import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import iisignature

from scipy.integrate import solve_ivp
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



OUTPUT_DIR = "/Users/francescoagro/Desktop/TESI/code/results/experiment2_multiseed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEEDS = [11, 22, 33, 44, 55]


T_max = 10.0
dt = 0.01
n_steps = int(T_max / dt)
time_grid = np.linspace(0.0, T_max, n_steps + 1)


t0 = 1.0
t0_index = int(t0 / dt)


maturities = np.array([1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0])
taus = maturities - t0


n_paths = 12000
n_train = 10000
n_test = 2000

train_idx = np.arange(n_train)
test_idx = np.arange(n_train, n_train + n_test)


kappa = 0.5
theta = 0.05
sigma = 0.01
R0 = 0.05
M0 = 0.0

scenarios = {
    "weak_memory": {
        "alpha": 0.1,
        "lambda": 2.0,
    },
    "moderate_memory": {
        "alpha": 0.3,
        "lambda": 1.0,
    },
    "longer_memory": {
        "alpha": 0.3,
        "lambda": 0.3,
    },
}

signature_orders = [2, 3, 4]

ridge_alphas = np.logspace(-8, 4, 25)



def simulate_oracle_volterra(
    seed,
    n_paths,
    n_steps,
    dt,
    R0,
    M0,
    kappa,
    theta,
    alpha,
    lambda_memory,
    sigma,
):
    np.random.seed(seed)

    R = np.zeros((n_paths, n_steps + 1))
    M = np.zeros((n_paths, n_steps + 1))

    R[:, 0] = R0
    M[:, 0] = M0

    for i in range(n_steps):
        dW = np.sqrt(dt) * np.random.randn(n_paths)

        R[:, i + 1] = (
            R[:, i]
            + kappa * (theta - R[:, i] - alpha * M[:, i]) * dt
            + sigma * dW
        )

        M[:, i + 1] = M[:, i] + (R[:, i] - lambda_memory * M[:, i]) * dt

    return R, M




def solve_affine_coefficients(taus, kappa, theta, alpha, lambda_memory, sigma):
    max_tau = float(np.max(taus))

    a_vec = np.array([kappa * theta, 0.0])

    K = np.array([
        [-kappa, -kappa * alpha],
        [1.0, -lambda_memory],
    ])

    e_R = np.array([1.0, 0.0])

    def ode(tau, y):
        B = y[:2]
        A = y[2]

        dB = K.T @ B - e_R
        dA = a_vec @ B + 0.5 * sigma**2 * B[0]**2

        return np.array([dB[0], dB[1], dA])

    y0 = np.array([0.0, 0.0, 0.0])

    sol = solve_ivp(
        ode,
        t_span=(0.0, max_tau),
        y0=y0,
        t_eval=taus,
        rtol=1e-10,
        atol=1e-12,
    )

    if not sol.success:
        raise RuntimeError("Riccati ODE solver failed.")

    B_R = sol.y[0]
    B_M = sol.y[1]
    A = sol.y[2]

    return A, B_R, B_M


def compute_log_prices_and_yields(R_t0, M_t0, taus, A, B_R, B_M):
    n_paths = len(R_t0)
    n_maturities = len(taus)

    logP = np.zeros((n_paths, n_maturities))

    for j in range(n_maturities):
        logP[:, j] = A[j] + B_R[j] * R_t0 + B_M[j] * M_t0

    yields = -logP / taus.reshape(1, -1)

    return logP, yields




def build_time_augmented_paths_until_t0(R_paths, time_grid, t0_index, train_idx):
    R_cut = R_paths[:, : t0_index + 1]
    time_cut = time_grid[: t0_index + 1]

    n_paths, n_times = R_cut.shape

    time_scaled = time_cut / time_cut[-1]


    R_train_cut = R_cut[train_idx]
    R_mean = np.mean(R_train_cut)
    R_std = np.std(R_train_cut)

    if R_std == 0:
        R_std = 1.0

    paths = np.zeros((n_paths, n_times, 2))
    paths[:, :, 0] = time_scaled[None, :]
    paths[:, :, 1] = (R_cut - R_mean) / R_std

    return paths


def compute_signature_features(paths, order):
    n_paths = paths.shape[0]
    path_dim = paths.shape[2]

    sig_dim = iisignature.siglength(path_dim, order)
    X = np.zeros((n_paths, sig_dim))

    start = time.time()

    for i in range(n_paths):
        X[i, :] = iisignature.sig(paths[i], order)

    runtime = time.time() - start

    return X, runtime




def fit_predict_yields_ridge(X, y_matrix, train_idx, test_idx):
    predictions = np.zeros((len(test_idx), y_matrix.shape[1]))
    metrics = []

    for j in range(y_matrix.shape[1]):
        y_train = y_matrix[train_idx, j]
        y_test = y_matrix[test_idx, j]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=ridge_alphas, cv=5)),
        ])

        model.fit(X[train_idx], y_train)
        y_pred = model.predict(X[test_idx])

        predictions[:, j] = y_pred

        mae_bp = mean_absolute_error(y_test, y_pred) * 10000
        rmse_bp = np.sqrt(mean_squared_error(y_test, y_pred)) * 10000
        r2 = r2_score(y_test, y_pred)
        selected_alpha = model.named_steps["ridge"].alpha_

        metrics.append({
            "mae_yield_bp": mae_bp,
            "rmse_yield_bp": rmse_bp,
            "r2_yield": r2,
            "selected_alpha": selected_alpha,
        })

    return predictions, metrics


def fit_predict_yields_oracle(X, y_matrix, train_idx, test_idx):
    predictions = np.zeros((len(test_idx), y_matrix.shape[1]))
    metrics = []

    for j in range(y_matrix.shape[1]):
        y_train = y_matrix[train_idx, j]
        y_test = y_matrix[test_idx, j]

        model = LinearRegression(fit_intercept=True)
        model.fit(X[train_idx], y_train)
        y_pred = model.predict(X[test_idx])

        predictions[:, j] = y_pred

        mae_bp = mean_absolute_error(y_test, y_pred) * 10000
        rmse_bp = np.sqrt(mean_squared_error(y_test, y_pred)) * 10000
        r2 = r2_score(y_test, y_pred)

        metrics.append({
            "mae_yield_bp": mae_bp,
            "rmse_yield_bp": rmse_bp,
            "r2_yield": r2,
            "selected_alpha": 0.0,
        })

    return predictions, metrics




def run_single_seed(seed):
    all_results = []
    curve_rows = []

    print(f"\n================ Running seed {seed} ================")

    for scenario_name, params in scenarios.items():
        print(f"\nScenario: {scenario_name}")

        alpha = params["alpha"]
        lambda_memory = params["lambda"]

        print("Simulating paths...")

        R_paths, M_paths = simulate_oracle_volterra(
            seed=seed,
            n_paths=n_paths,
            n_steps=n_steps,
            dt=dt,
            R0=R0,
            M0=M0,
            kappa=kappa,
            theta=theta,
            alpha=alpha,
            lambda_memory=lambda_memory,
            sigma=sigma,
        )

        R_t0 = R_paths[:, t0_index]
        M_t0 = M_paths[:, t0_index]

        print("Solving Riccati equations...")

        A, B_R, B_M = solve_affine_coefficients(
            taus=taus,
            kappa=kappa,
            theta=theta,
            alpha=alpha,
            lambda_memory=lambda_memory,
            sigma=sigma,
        )

        logP_true, yields_true = compute_log_prices_and_yields(
            R_t0=R_t0,
            M_t0=M_t0,
            taus=taus,
            A=A,
            B_R=B_R,
            B_M=B_M,
        )

        print("Yield target shape:", yields_true.shape)


        print("Evaluating oracle benchmark...")

        X_oracle = np.column_stack([R_t0, M_t0])

        pred_oracle, metrics_oracle = fit_predict_yields_oracle(
            X_oracle,
            yields_true,
            train_idx,
            test_idx,
        )

        for j, T in enumerate(maturities):
            all_results.append({
                "seed": seed,
                "scenario": scenario_name,
                "method": "oracle",
                "signature_order": np.nan,
                "maturity": T,
                "tau": taus[j],
                "n_features": X_oracle.shape[1],
                "signature_runtime": np.nan,
                **metrics_oracle[j],
            })


        print("Evaluating Markovian benchmark...")

        X_markovian = R_t0.reshape(-1, 1)

        pred_markovian, metrics_markovian = fit_predict_yields_ridge(
            X_markovian,
            yields_true,
            train_idx,
            test_idx,
        )

        for j, T in enumerate(maturities):
            all_results.append({
                "seed": seed,
                "scenario": scenario_name,
                "method": "markovian",
                "signature_order": np.nan,
                "maturity": T,
                "tau": taus[j],
                "n_features": X_markovian.shape[1],
                "signature_runtime": np.nan,
                **metrics_markovian[j],
            })


        print("Building time-augmented paths...")

        augmented_paths = build_time_augmented_paths_until_t0(
            R_paths=R_paths,
            time_grid=time_grid,
            t0_index=t0_index,
            train_idx=train_idx,
        )

        signature_predictions = {}

        for order in signature_orders:
            print(f"Computing signatures of order {order}...")

            X_sig, runtime = compute_signature_features(
                augmented_paths,
                order,
            )

            print("Signature feature matrix shape:", X_sig.shape)
            print("Signature runtime:", runtime)

            pred_sig, metrics_sig = fit_predict_yields_ridge(
                X_sig,
                yields_true,
                train_idx,
                test_idx,
            )

            signature_predictions[order] = pred_sig

            for j, T in enumerate(maturities):
                all_results.append({
                    "seed": seed,
                    "scenario": scenario_name,
                    "method": "signature",
                    "signature_order": order,
                    "maturity": T,
                    "tau": taus[j],
                    "n_features": X_sig.shape[1],
                    "signature_runtime": runtime,
                    **metrics_sig[j],
                })


        true_mean_curve = np.mean(yields_true[test_idx], axis=0)
        oracle_mean_curve = np.mean(pred_oracle, axis=0)
        markovian_mean_curve = np.mean(pred_markovian, axis=0)

        for j, T in enumerate(maturities):
            curve_rows.append({
                "seed": seed,
                "scenario": scenario_name,
                "method": "true",
                "signature_order": np.nan,
                "maturity": T,
                "yield": true_mean_curve[j],
            })

            curve_rows.append({
                "seed": seed,
                "scenario": scenario_name,
                "method": "oracle",
                "signature_order": np.nan,
                "maturity": T,
                "yield": oracle_mean_curve[j],
            })

            curve_rows.append({
                "seed": seed,
                "scenario": scenario_name,
                "method": "markovian",
                "signature_order": np.nan,
                "maturity": T,
                "yield": markovian_mean_curve[j],
            })

            for order in signature_orders:
                sig_mean_curve = np.mean(signature_predictions[order], axis=0)

                curve_rows.append({
                    "seed": seed,
                    "scenario": scenario_name,
                    "method": "signature",
                    "signature_order": order,
                    "maturity": T,
                    "yield": sig_mean_curve[j],
                })

    return pd.DataFrame(all_results), pd.DataFrame(curve_rows)



def make_summary(results_df):
    summary_df = (
        results_df
        .groupby(["scenario", "method", "signature_order"], dropna=False)
        .agg(
            mean_mae_yield_bp=("mae_yield_bp", "mean"),
            std_mae_yield_bp=("mae_yield_bp", "std"),
            mean_rmse_yield_bp=("rmse_yield_bp", "mean"),
            std_rmse_yield_bp=("rmse_yield_bp", "std"),
            mean_r2_yield=("r2_yield", "mean"),
            std_r2_yield=("r2_yield", "std"),
            mean_selected_alpha=("selected_alpha", "mean"),
            std_selected_alpha=("selected_alpha", "std"),
            n_features=("n_features", "first"),
            mean_signature_runtime=("signature_runtime", "mean"),
            std_signature_runtime=("signature_runtime", "std"),
        )
        .reset_index()
    )

    return summary_df


def make_maturity_summary(results_df):
    maturity_summary_df = (
        results_df
        .groupby(
            ["scenario", "method", "signature_order", "maturity"],
            dropna=False,
        )
        .agg(
            mean_mae_yield_bp=("mae_yield_bp", "mean"),
            std_mae_yield_bp=("mae_yield_bp", "std"),
            mean_rmse_yield_bp=("rmse_yield_bp", "mean"),
            std_rmse_yield_bp=("rmse_yield_bp", "std"),
            mean_r2_yield=("r2_yield", "mean"),
            std_r2_yield=("r2_yield", "std"),
            mean_selected_alpha=("selected_alpha", "mean"),
            std_selected_alpha=("selected_alpha", "std"),
            n_features=("n_features", "first"),
            mean_signature_runtime=("signature_runtime", "mean"),
            std_signature_runtime=("signature_runtime", "std"),
        )
        .reset_index()
    )

    return maturity_summary_df


def make_curve_summary(curves_df):
    curve_summary_df = (
        curves_df
        .groupby(
            ["scenario", "method", "signature_order", "maturity"],
            dropna=False,
        )
        .agg(
            mean_yield=("yield", "mean"),
            std_yield=("yield", "std"),
        )
        .reset_index()
    )

    return curve_summary_df



def plot_average_error_by_scenario(summary_df):
    for scenario_name in scenarios.keys():
        scenario_summary = summary_df[summary_df["scenario"] == scenario_name].copy()

        labels = []
        values = []
        errors = []

        for _, row in scenario_summary.iterrows():
            if row["method"] == "signature":
                labels.append(f"Sig N={int(row['signature_order'])}")
            else:
                labels.append(row["method"])

            values.append(row["mean_mae_yield_bp"])
            errors.append(row["std_mae_yield_bp"])

        plt.figure(figsize=(9, 5))
        plt.bar(labels, values, yerr=errors, capsize=5)
        plt.ylabel("Average absolute yield error (bp)")
        plt.title(f"Yield-curve reconstruction error: {scenario_name}")
        plt.xticks(rotation=30)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                f"{scenario_name}_multiseed_avg_yield_error_bp.png",
            ),
            dpi=300,
        )

        plt.show()


def plot_mean_yield_curves(curve_summary_df):
    for scenario_name in scenarios.keys():
        plt.figure(figsize=(9, 5))

        scenario_curves = curve_summary_df[
            curve_summary_df["scenario"] == scenario_name
        ]

        plot_specs = [
            ("true", np.nan, "True"),
            ("oracle", np.nan, "Oracle"),
            ("markovian", np.nan, "Markovian"),
        ]

        for method, order, label in plot_specs:
            df = scenario_curves[
                (scenario_curves["method"] == method)
                & (scenario_curves["signature_order"].isna())
            ]

            plt.errorbar(
                df["maturity"],
                df["mean_yield"] * 100,
                yerr=df["std_yield"] * 100,
                marker="o",
                linestyle="--" if method != "true" else "-",
                capsize=4,
                label=label,
            )

        for order in signature_orders:
            df_sig = scenario_curves[
                (scenario_curves["method"] == "signature")
                & (scenario_curves["signature_order"] == order)
            ]

            plt.errorbar(
                df_sig["maturity"],
                df_sig["mean_yield"] * 100,
                yerr=df_sig["std_yield"] * 100,
                marker="o",
                linestyle="--",
                capsize=4,
                label=f"Signature N={order}",
            )

        plt.xlabel("Maturity T")
        plt.ylabel("Yield (%)")
        plt.title(f"Mean reconstructed yield curve: {scenario_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                f"{scenario_name}_multiseed_mean_yield_curve.png",
            ),
            dpi=300,
        )

        plt.show()


def plot_error_by_maturity(maturity_summary_df):
    for scenario_name in scenarios.keys():
        plt.figure(figsize=(9, 5))

        scenario_results = maturity_summary_df[
            maturity_summary_df["scenario"] == scenario_name
        ]

        for method in ["oracle", "markovian"]:
            df_method = scenario_results[
                (scenario_results["method"] == method)
                & (scenario_results["signature_order"].isna())
            ]

            plt.errorbar(
                df_method["maturity"],
                df_method["mean_mae_yield_bp"],
                yerr=df_method["std_mae_yield_bp"],
                marker="o",
                capsize=5,
                label=method,
            )

        for order in signature_orders:
            df_sig = scenario_results[
                (scenario_results["method"] == "signature")
                & (scenario_results["signature_order"] == order)
            ]

            plt.errorbar(
                df_sig["maturity"],
                df_sig["mean_mae_yield_bp"],
                yerr=df_sig["std_mae_yield_bp"],
                marker="o",
                capsize=5,
                label=f"Signature N={order}",
            )

        plt.xlabel("Maturity T")
        plt.ylabel("Absolute yield error (bp)")
        plt.title(f"Yield error by maturity: {scenario_name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                f"{scenario_name}_multiseed_yield_error_by_maturity.png",
            ),
            dpi=300,
        )

        plt.show()


def main():
    start_total = time.time()

    print("\n================ Experiment 2: Multi-seed Yield-Curve Reconstruction ================")

    all_results = []
    all_curves = []

    for seed in SEEDS:
        results_seed, curves_seed = run_single_seed(seed)
        all_results.append(results_seed)
        all_curves.append(curves_seed)

    results_df = pd.concat(all_results, ignore_index=True)
    curves_df = pd.concat(all_curves, ignore_index=True)

    summary_df = make_summary(results_df)
    maturity_summary_df = make_maturity_summary(results_df)
    curve_summary_df = make_curve_summary(curves_df)

    results_path = os.path.join(OUTPUT_DIR, "experiment2_multiseed_results.csv")
    curves_path = os.path.join(OUTPUT_DIR, "experiment2_multiseed_mean_yield_curves.csv")
    summary_path = os.path.join(OUTPUT_DIR, "experiment2_multiseed_summary.csv")
    maturity_summary_path = os.path.join(OUTPUT_DIR, "experiment2_multiseed_maturity_summary.csv")
    curve_summary_path = os.path.join(OUTPUT_DIR, "experiment2_multiseed_curve_summary.csv")

    results_df.to_csv(results_path, index=False)
    curves_df.to_csv(curves_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    maturity_summary_df.to_csv(maturity_summary_path, index=False)
    curve_summary_df.to_csv(curve_summary_path, index=False)

    print("\n================ Multi-seed Summary Results ================")
    print(summary_df)

    print("\n================ Multi-seed Maturity Summary Results ================")
    print(maturity_summary_df)

    print("\nSaved tables to:")
    print(results_path)
    print(curves_path)
    print(summary_path)
    print(maturity_summary_path)
    print(curve_summary_path)

    print("\nCreating plots...")

    plot_average_error_by_scenario(summary_df)
    plot_mean_yield_curves(curve_summary_df)
    plot_error_by_maturity(maturity_summary_df)

    print("\nSaved plots to:")
    print(OUTPUT_DIR)

    print(f"\nTotal runtime: {time.time() - start_total:.2f} seconds")
    print("\n==========================================================================\n")


if __name__ == "__main__":
    main()
