import os
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import iisignature

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score


SEEDS = [11, 22, 33, 44, 55]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "results", "experiment6_power_law"))
os.makedirs(OUTPUT_DIR, exist_ok=True)


KAPPA = 0.5
THETA = 0.05
ALPHA = 0.3
SIGMA = 0.01
R0 = 0.05

H = 0.10
KERNEL_EPS = 1e-4

DT = 0.02
T0 = 2.0
T_MAX = 5.0
MATURITIES = np.array([3.0, 4.0, 5.0])

N_PATHS = 6000
N_TRAIN = 5000
N_VALIDATION = 150
N_NESTED = 300

SIGNATURE_LEVELS = [1, 2, 3, 4]

RIDGE_ALPHAS = np.logspace(-6, 4, 25)


def integrate_trapezoid(values, dx, axis=1):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(values, dx=dx, axis=axis)
    return np.trapz(values, dx=dx, axis=axis)


def time_index(t):
    return int(round(t / DT))


def power_law_weights(n_steps):
    """
    Cell-averaged kernel weights: w_j = (1/DT) * \int_{j*DT}^{(j+1)*DT} K(u) du,
    where K(u) = (u + eps)^(H - 1/2) / Gamma(H + 1/2). Using the analytic
    antiderivative (u+eps)^(H+1/2) / Gamma(H+3/2) avoids evaluating the
    near-singular kernel pointwise, which materially overweights the most
    recent interval (see thesis discussion). Dividing by DT turns the cell
    integral into a cell average, matching the convention expected by
    compute_convolution_memory (which multiplies by DT again to recover a
    Riemann sum) -- this keeps the shared consumer function correct both
    here and for exponential_weights, which returns pointwise kernel values.
    """
    j = np.arange(n_steps + 1)
    lower = j * DT + KERNEL_EPS
    upper = (j + 1) * DT + KERNEL_EPS
    weights = (upper ** (H + 0.5) - lower ** (H + 0.5)) / math.gamma(H + 1.5) / DT
    return weights


def exponential_weights(n_steps, beta=1.0):
    lags = np.arange(n_steps + 1)
    u = lags * DT
    return np.exp(-beta * u)


def compute_convolution_memory(R_values, weights, current_idx):
    """
    Computes M_t ≈ dt * sum_{j=0}^{i} K(t_i - t_j) R_{t_j}.
    R_values has shape (n_paths, n_steps + 1).
    """
    kernel_slice = weights[:current_idx + 1][::-1]
    return DT * (R_values[:, :current_idx + 1] @ kernel_slice)


def simulate_power_law_paths(n_paths, t_max, seed):
    rng = np.random.default_rng(seed)

    n_steps = time_index(t_max)
    times = np.linspace(0.0, t_max, n_steps + 1)

    weights = power_law_weights(n_steps)

    R = np.zeros((n_paths, n_steps + 1))
    M = np.zeros((n_paths, n_steps + 1))

    R[:, 0] = R0
    M[:, 0] = 0.0

    sqrt_dt = np.sqrt(DT)

    for i in range(n_steps):
        M[:, i] = compute_convolution_memory(R, weights, i)

        drift = KAPPA * (THETA - R[:, i] - ALPHA * M[:, i])
        diffusion = SIGMA * sqrt_dt * rng.standard_normal(n_paths)

        R[:, i + 1] = R[:, i] + drift * DT + diffusion

    M[:, n_steps] = compute_convolution_memory(R, weights, n_steps)

    return times, R, M


def realized_discounted_payoffs(R_paths):
    t0_idx = time_index(T0)

    targets = []

    for T in MATURITIES:
        T_idx = time_index(T)

        integral = integrate_trapezoid(
            R_paths[:, t0_idx:T_idx + 1],
            dx=DT,
            axis=1,
        )

        targets.append(np.exp(-integral))

    return np.column_stack(targets)


def nested_mc_prices_from_histories(R_histories, seed):
    rng = np.random.default_rng(seed)

    t0_idx = time_index(T0)
    tmax_idx = time_index(T_MAX)

    weights = power_law_weights(tmax_idx)

    n_histories = R_histories.shape[0]
    nested_prices = np.zeros((n_histories, len(MATURITIES)))
    nested_prices_se = np.zeros((n_histories, len(MATURITIES)))

    sqrt_dt = np.sqrt(DT)

    for h in range(n_histories):
        R_nested = np.zeros((N_NESTED, tmax_idx + 1))

        R_nested[:, :t0_idx + 1] = R_histories[h, :t0_idx + 1][None, :]

        for i in range(t0_idx, tmax_idx):
            M_i = compute_convolution_memory(R_nested, weights, i)

            drift = KAPPA * (THETA - R_nested[:, i] - ALPHA * M_i)
            diffusion = SIGMA * sqrt_dt * rng.standard_normal(N_NESTED)

            R_nested[:, i + 1] = R_nested[:, i] + drift * DT + diffusion

        for j, T in enumerate(MATURITIES):
            T_idx = time_index(T)

            integral = integrate_trapezoid(
                R_nested[:, t0_idx:T_idx + 1],
                dx=DT,
                axis=1,
            )

            discounted = np.exp(-integral)
            nested_prices[h, j] = discounted.mean()
            # Standard error of the nested MC estimate itself
            nested_prices_se[h, j] = discounted.std(ddof=1) / np.sqrt(N_NESTED)

    return nested_prices, nested_prices_se


def price_to_yield(price, maturity):
    price = np.maximum(price, 1e-12)
    tau = maturity - T0
    return -np.log(price) / tau


def build_time_augmented_paths(R_paths):
    t0_idx = time_index(T0)

    path_times = np.linspace(0.0, T0, t0_idx + 1)

    n_paths = R_paths.shape[0]
    n_points = t0_idx + 1

    paths = np.zeros((n_paths, n_points, 2))
    paths[:, :, 0] = path_times[None, :]
    paths[:, :, 1] = R_paths[:, :t0_idx + 1]

    return paths


def compute_signature_features(paths, level):
    return np.array([
        iisignature.sig(path, level)
        for path in paths
    ])


def compute_exponential_proxy(R_paths, beta=1.0):
    t0_idx = time_index(T0)
    weights = exponential_weights(t0_idx, beta=beta)
    return compute_convolution_memory(R_paths, weights, t0_idx)


def compute_power_memory_proxy(R_paths):
    t0_idx = time_index(T0)
    weights = power_law_weights(t0_idx)
    return compute_convolution_memory(R_paths, weights, t0_idx)


def fit_predict_ridge(X_train, y_train, X_test):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=RIDGE_ALPHAS, cv=5)),
    ])

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    pred = np.maximum(pred, 1e-12)

    selected_alpha = model.named_steps["ridge"].alpha_

    return pred, selected_alpha


def evaluate_method(
    seed,
    method,
    signature_order,
    X_train,
    X_test,
    train_targets,
    validation_targets,
    signature_runtime=np.nan,
):
    rows = []
    predictions = np.zeros_like(validation_targets)

    for j, T in enumerate(MATURITIES):
        y_train = train_targets[:, j]
        y_true = validation_targets[:, j]

        pred, selected_alpha = fit_predict_ridge(
            X_train,
            y_train,
            X_test,
        )

        predictions[:, j] = pred

        price_rmse = np.sqrt(mean_squared_error(y_true, pred))
        r2_price = r2_score(y_true, pred)

        true_yield = price_to_yield(y_true, T)
        pred_yield = price_to_yield(pred, T)

        yield_rmse_bp = (
            np.sqrt(mean_squared_error(true_yield, pred_yield))
            * 10000
        )

        rows.append({
            "seed": seed,
            "method": method,
            "signature_order": signature_order,
            "maturity": T,
            "price_rmse": price_rmse,
            "yield_rmse_bp": yield_rmse_bp,
            "r2_price": r2_price,
            "selected_alpha": selected_alpha,
            "n_features": X_train.shape[1],
            "signature_runtime": signature_runtime,
        })

    return rows, predictions



def run_single_seed(seed):
    print(f"\n================ Experiment 6 | Seed {seed} ================\n")

    print("Simulating power-law Volterra paths...")
    times, R_paths, M_paths = simulate_power_law_paths(
        N_PATHS,
        T_MAX,
        seed=seed,
    )

    print("R_paths shape:", R_paths.shape)

    train_idx = np.arange(N_TRAIN)
    validation_idx = np.arange(N_TRAIN, N_TRAIN + N_VALIDATION)

    R_train = R_paths[train_idx]
    R_validation = R_paths[validation_idx]

    print("Computing realized discounted payoffs for training...")
    train_targets = realized_discounted_payoffs(R_train)

    print("Computing nested Monte Carlo validation labels...")
    validation_targets, validation_targets_se = nested_mc_prices_from_histories(
        R_validation,
        seed=seed + 1000,
    )

    t0_idx = time_index(T0)

    all_rows = []
    predictions = {}


    print("Evaluating constant benchmark...")

    X_const_train = np.ones((N_TRAIN, 1))
    X_const_validation = np.ones((N_VALIDATION, 1))

    rows, pred = evaluate_method(
        seed=seed,
        method="constant",
        signature_order=np.nan,
        X_train=X_const_train,
        X_test=X_const_validation,
        train_targets=train_targets,
        validation_targets=validation_targets,
    )

    all_rows.extend(rows)
    predictions["constant"] = pred


    print("Evaluating Markovian benchmark...")

    R_t0_train = R_train[:, t0_idx].reshape(-1, 1)
    R_t0_validation = R_validation[:, t0_idx].reshape(-1, 1)

    rows, pred = evaluate_method(
        seed=seed,
        method="markovian",
        signature_order=np.nan,
        X_train=R_t0_train,
        X_test=R_t0_validation,
        train_targets=train_targets,
        validation_targets=validation_targets,
    )

    all_rows.extend(rows)
    predictions["markovian"] = pred


    print("Evaluating exponential-memory proxy...")

    M_exp_train = compute_exponential_proxy(R_train, beta=1.0).reshape(-1, 1)
    M_exp_validation = compute_exponential_proxy(R_validation, beta=1.0).reshape(-1, 1)

    X_exp_train = np.column_stack([R_t0_train, M_exp_train])
    X_exp_validation = np.column_stack([R_t0_validation, M_exp_validation])

    rows, pred = evaluate_method(
        seed=seed,
        method="exponential_proxy",
        signature_order=np.nan,
        X_train=X_exp_train,
        X_test=X_exp_validation,
        train_targets=train_targets,
        validation_targets=validation_targets,
    )

    all_rows.extend(rows)
    predictions["exponential_proxy"] = pred


    print("Evaluating power-law memory proxy...")

    M_power_train = compute_power_memory_proxy(R_train).reshape(-1, 1)
    M_power_validation = compute_power_memory_proxy(R_validation).reshape(-1, 1)

    X_power_train = np.column_stack([R_t0_train, M_power_train])
    X_power_validation = np.column_stack([R_t0_validation, M_power_validation])

    rows, pred = evaluate_method(
        seed=seed,
        method="power_memory_proxy",
        signature_order=np.nan,
        X_train=X_power_train,
        X_test=X_power_validation,
        train_targets=train_targets,
        validation_targets=validation_targets,
    )

    all_rows.extend(rows)
    predictions["power_memory_proxy"] = pred


    print("Building time-augmented paths...")
    paths_train = build_time_augmented_paths(R_train)
    paths_validation = build_time_augmented_paths(R_validation)

    for level in SIGNATURE_LEVELS:
        print(f"Computing signatures of order {level}...")

        tic = time.time()

        sig_train = compute_signature_features(paths_train, level)
        sig_validation = compute_signature_features(paths_validation, level)

        runtime = time.time() - tic

        X_sig_train = np.column_stack([R_t0_train, sig_train])
        X_sig_validation = np.column_stack([R_t0_validation, sig_validation])

        print("Signature feature matrix shape:", X_sig_train.shape)
        print("Signature runtime:", runtime)

        rows, pred = evaluate_method(
            seed=seed,
            method="signature",
            signature_order=level,
            X_train=X_sig_train,
            X_test=X_sig_validation,
            train_targets=train_targets,
            validation_targets=validation_targets,
            signature_runtime=runtime,
        )

        all_rows.extend(rows)
        predictions[f"signature_N{level}"] = pred

    results_seed = pd.DataFrame(all_rows)

    se_summary_rows = [
        {
            "seed": seed,
            "maturity": T,
            "mean_nested_mc_se": validation_targets_se[:, j].mean(),
            "mean_nested_mc_se_yield_bp": (
                validation_targets_se[:, j].mean()
                / validation_targets[:, j].mean()
                * 10000
            ),
        }
        for j, T in enumerate(MATURITIES)
    ]

    return results_seed, predictions, validation_targets, se_summary_rows



def build_summary(results_df):
    per_seed = (
        results_df
        .groupby(["method", "signature_order", "seed"], dropna=False)
        .agg(
            price_rmse=("price_rmse", "mean"),
            yield_rmse_bp=("yield_rmse_bp", "mean"),
            r2_price=("r2_price", "mean"),
            selected_alpha=("selected_alpha", "mean"),
            n_features=("n_features", "first"),
            signature_runtime=("signature_runtime", "mean"),
        )
        .reset_index()
    )
    summary = (
        per_seed
        .groupby(["method", "signature_order"], dropna=False)
        .agg(
            mean_price_rmse=("price_rmse", "mean"),
            std_price_rmse=("price_rmse", "std"),
            mean_yield_rmse_bp=("yield_rmse_bp", "mean"),
            std_yield_rmse_bp=("yield_rmse_bp", "std"),
            mean_r2_price=("r2_price", "mean"),
            std_r2_price=("r2_price", "std"),
            mean_selected_alpha=("selected_alpha", "mean"),
            std_selected_alpha=("selected_alpha", "std"),
            n_features=("n_features", "first"),
            mean_signature_runtime=("signature_runtime", "mean"),
            std_signature_runtime=("signature_runtime", "std"),
        )
        .reset_index()
    )
    return summary


def build_maturity_summary(results_df):
    maturity_summary = (
        results_df
        .groupby(
            ["method", "signature_order", "maturity"],
            dropna=False,
        )
        .agg(
            mean_price_rmse=("price_rmse", "mean"),
            std_price_rmse=("price_rmse", "std"),

            mean_yield_rmse_bp=("yield_rmse_bp", "mean"),
            std_yield_rmse_bp=("yield_rmse_bp", "std"),

            mean_r2_price=("r2_price", "mean"),
            std_r2_price=("r2_price", "std"),

            mean_selected_alpha=("selected_alpha", "mean"),
            std_selected_alpha=("selected_alpha", "std"),

            n_features=("n_features", "first"),

            mean_signature_runtime=("signature_runtime", "mean"),
            std_signature_runtime=("signature_runtime", "std"),
        )
        .reset_index()
    )

    return maturity_summary



def plot_error_vs_signature_order(summary):
    plt.figure(figsize=(9, 6))

    sig = summary[summary["method"] == "signature"].copy()

    plt.errorbar(
        sig["signature_order"],
        sig["mean_yield_rmse_bp"],
        yerr=sig["std_yield_rmse_bp"],
        marker="o",
        capsize=5,
        label="Signature",
    )

    for method, label in [
        ("constant", "Constant"),
        ("markovian", "Markovian"),
        ("exponential_proxy", "Exponential proxy"),
        ("power_memory_proxy", "Power-law proxy"),
    ]:
        df = summary[summary["method"] == method]
        if len(df) > 0:
            y = df["mean_yield_rmse_bp"].iloc[0]
            yerr = df["std_yield_rmse_bp"].iloc[0]

            plt.axhline(
                y,
                linestyle="--",
                label=f"{label} ({y:.2f} ± {yerr:.2f} bp)",
            )

    plt.xlabel("Signature truncation order N")
    plt.ylabel("Average yield RMSE (bp)")
    plt.title("Experiment 6: Power-law memory model")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "experiment6_yield_error_vs_signature_order.png",
        ),
        dpi=300,
    )

    plt.show()
    plt.close()


def plot_error_by_maturity(maturity_summary):
    plt.figure(figsize=(10, 6))

    plot_specs = [
        ("markovian", np.nan, "Markovian"),
        ("exponential_proxy", np.nan, "Exponential proxy"),
        ("power_memory_proxy", np.nan, "Power-law proxy"),
        ("signature", 2.0, "Signature N=2"),
        ("signature", 3.0, "Signature N=3"),
        ("signature", 4.0, "Signature N=4"),
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

        if len(df) > 0:
            df = df.sort_values("maturity")

            plt.errorbar(
                df["maturity"],
                df["mean_yield_rmse_bp"],
                yerr=df["std_yield_rmse_bp"],
                marker="o",
                capsize=5,
                label=label,
            )

    plt.xlabel("Maturity T")
    plt.ylabel("Yield RMSE (bp)")
    plt.title("Experiment 6: Yield error by maturity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "experiment6_yield_error_by_maturity.png",
        ),
        dpi=300,
    )

    plt.show()
    plt.close()


def plot_predicted_vs_true_yields(predictions, validation_targets, seed):
    T = MATURITIES[-1]
    j = len(MATURITIES) - 1

    true_yield = price_to_yield(validation_targets[:, j], T)

    plt.figure(figsize=(7, 7))

    selected_methods = [
        ("markovian", "Markovian"),
        ("exponential_proxy", "Exponential proxy"),
        ("power_memory_proxy", "Power-law proxy"),
        ("signature_N3", "Signature N=3"),
    ]

    for key, label in selected_methods:
        pred_price = predictions[key][:, j]
        pred_yield = price_to_yield(pred_price, T)

        plt.scatter(
            true_yield,
            pred_yield,
            alpha=0.45,
            label=label,
        )

    min_val = true_yield.min()
    max_val = true_yield.max()

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
    )

    plt.xlabel("Nested Monte Carlo yield")
    plt.ylabel("Predicted yield")
    plt.title(f"Experiment 6: Predicted vs nested-MC yields, T={T:g}, seed={seed}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"experiment6_predicted_vs_nested_yields_T{T:g}_seed{seed}.png",
        ),
        dpi=300,
    )

    plt.show()
    plt.close()



def main():
    start_time = time.time()
    print("\n================ Experiment 6: Power-law Volterra Memory ================\n")
    all_results = []
    all_se_rows = []
    representative_predictions = None
    representative_targets = None
    representative_seed = SEEDS[0]
    for seed in SEEDS:
        results_seed, predictions_seed, validation_targets_seed, se_rows_seed = run_single_seed(seed)
        all_results.append(results_seed)
        all_se_rows.extend(se_rows_seed)
        if seed == representative_seed:
            representative_predictions = predictions_seed
            representative_targets = validation_targets_seed
    results = pd.concat(all_results, ignore_index=True)
    summary = build_summary(results)
    maturity_summary = build_maturity_summary(results)
    print("\n================ Full Multi-seed Results ================")
    print(results)
    print("\n================ Summary Results ================")
    print(summary)
    print("\n================ Maturity Summary Results ================")
    print(maturity_summary)
    results_path = os.path.join(
        OUTPUT_DIR,
        "experiment6_power_law_results.csv",
    )
    summary_path = os.path.join(
        OUTPUT_DIR,
        "experiment6_power_law_summary.csv",
    )
    maturity_summary_path = os.path.join(
        OUTPUT_DIR,
        "experiment6_power_law_maturity_summary.csv",
    )
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    maturity_summary.to_csv(maturity_summary_path, index=False)
    print("\nSaved results to:")
    print(results_path)
    print(summary_path)
    print(maturity_summary_path)

    nested_mc_se_df = pd.DataFrame(all_se_rows)
    nested_mc_se_summary = nested_mc_se_df.groupby("maturity", as_index=False).agg(
        mean_se=("mean_nested_mc_se", "mean"),
        mean_se_yield_bp=("mean_nested_mc_se_yield_bp", "mean"),
    )
    print("\n================ Nested MC Standard Error Summary ================")
    print(nested_mc_se_summary)
    nested_mc_se_summary.to_csv(
        os.path.join(OUTPUT_DIR, "experiment6_nested_mc_se_summary.csv"),
        index=False,
    )

    plot_error_vs_signature_order(summary)
    plot_error_by_maturity(maturity_summary)
    if representative_predictions is not None:
        plot_predicted_vs_true_yields(
            representative_predictions,
            representative_targets,
            representative_seed,
        )
    print("\nSaved plots to:")
    print(OUTPUT_DIR)
    print("\nTotal runtime:", time.time() - start_time)
    print("==========================================================================\n")


if __name__ == "__main__":
    main()
