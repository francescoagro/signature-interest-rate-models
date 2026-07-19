import os
import time
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
OUTPUT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "results", "experiment4"))
os.makedirs(OUTPUT_DIR, exist_ok=True)


KAPPA = 0.5
THETA = 0.05
ALPHA = 0.2
TAU = 1.0
SIGMA = 0.01
R0 = 0.05


DT = 0.01
T0 = 2.0
T_MAX = 6.0
MATURITIES = np.array([3.0, 4.0, 6.0])


N_PATHS = 12000
N_TRAIN = 10000
N_VALIDATION = 300
N_NESTED = 500


SIGNATURE_LEVELS = [1, 2, 3, 4]


RIDGE_ALPHAS = np.logspace(-6, 3, 25)



def time_index(t):
    return int(round(t / DT))


def integrate_trapezoid(values, dx, axis=1):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(values, dx=dx, axis=axis)
    return np.trapz(values, dx=dx, axis=axis)


def price_to_yield(price, maturity):
    price = np.maximum(price, 1e-12)
    tau = maturity - T0
    return -np.log(price) / tau



def simulate_delay_paths(n_paths, t_max, seed):
    rng = np.random.default_rng(seed)

    n_steps = time_index(t_max)
    delay_steps = time_index(TAU)

    times = np.linspace(0.0, t_max, n_steps + 1)

    R = np.zeros((n_paths, n_steps + 1))
    R[:, 0] = R0

    sqrt_dt = np.sqrt(DT)

    for i in range(n_steps):
        if i - delay_steps >= 0:
            R_delay = R[:, i - delay_steps]
        else:
            R_delay = R0

        drift = KAPPA * (THETA - R[:, i] - ALPHA * R_delay)
        diffusion = SIGMA * sqrt_dt * rng.standard_normal(n_paths)

        R[:, i + 1] = R[:, i] + drift * DT + diffusion

    return times, R


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
    delay_steps = time_index(TAU)

    n_histories = R_histories.shape[0]
    nested_prices = np.zeros((n_histories, len(MATURITIES)))

    sqrt_dt = np.sqrt(DT)

    for h in range(n_histories):
        R_nested = np.zeros((N_NESTED, tmax_idx + 1))

        R_nested[:, :t0_idx + 1] = R_histories[h, :t0_idx + 1][None, :]

        for i in range(t0_idx, tmax_idx):
            if i - delay_steps >= 0:
                R_delay = R_nested[:, i - delay_steps]
            else:
                R_delay = R0

            drift = KAPPA * (THETA - R_nested[:, i] - ALPHA * R_delay)
            diffusion = SIGMA * sqrt_dt * rng.standard_normal(N_NESTED)

            R_nested[:, i + 1] = R_nested[:, i] + drift * DT + diffusion

        for j, T in enumerate(MATURITIES):
            T_idx = time_index(T)

            integral = integrate_trapezoid(
                R_nested[:, t0_idx:T_idx + 1],
                dx=DT,
                axis=1,
            )

            nested_prices[h, j] = np.exp(-integral).mean()

    return nested_prices




def build_rolling_window_paths(R_paths):
    t0_idx = time_index(T0)
    window_steps = time_index(TAU)

    start_idx = t0_idx - window_steps

    window_times = np.linspace(T0 - TAU, T0, window_steps + 1)

    n_paths = R_paths.shape[0]
    n_points = window_steps + 1

    paths = np.zeros((n_paths, n_points, 2))
    paths[:, :, 0] = window_times[None, :]
    paths[:, :, 1] = R_paths[:, start_idx:t0_idx + 1]

    return paths


def compute_signature_features(paths, level):
    return np.array([
        iisignature.sig(path, level)
        for path in paths
    ])


def build_delay_features(R_paths):
    t0_idx = time_index(T0)
    delay_idx = time_index(T0 - TAU)

    R_t0 = R_paths[:, t0_idx].reshape(-1, 1)
    R_delay = R_paths[:, delay_idx].reshape(-1, 1)

    R_avg = np.mean(
        R_paths[:, delay_idx:t0_idx + 1],
        axis=1,
    ).reshape(-1, 1)

    return R_t0, R_delay, R_avg



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
    X_validation,
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
            X_validation,
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
    print(f"\n================ Running seed {seed} ================\n")

    print("Simulating delay-model paths...")
    times, R_paths = simulate_delay_paths(
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
    validation_targets = nested_mc_prices_from_histories(
        R_validation,
        seed=seed + 1000,
    )

    all_rows = []
    predictions = {}

    R_t0_train, R_delay_train, R_avg_train = build_delay_features(R_train)
    R_t0_val, R_delay_val, R_avg_val = build_delay_features(R_validation)

    print("Evaluating Markovian benchmark...")

    rows, pred = evaluate_method(
        seed=seed,
        method="markovian",
        signature_order=np.nan,
        X_train=R_t0_train,
        X_validation=R_t0_val,
        train_targets=train_targets,
        validation_targets=validation_targets,
    )

    all_rows.extend(rows)
    predictions["markovian"] = pred


    print("Evaluating delay-pair benchmark...")

    X_delay_pair_train = np.column_stack([
        R_t0_train,
        R_delay_train,
    ])

    X_delay_pair_val = np.column_stack([
        R_t0_val,
        R_delay_val,
    ])

    rows, pred = evaluate_method(
        seed=seed,
        method="delay_pair",
        signature_order=np.nan,
        X_train=X_delay_pair_train,
        X_validation=X_delay_pair_val,
        train_targets=train_targets,
        validation_targets=validation_targets,
    )

    all_rows.extend(rows)
    predictions["delay_pair"] = pred


    print("Evaluating delay-summary benchmark...")

    X_delay_summary_train = np.column_stack([
        R_t0_train,
        R_delay_train,
        R_avg_train,
    ])

    X_delay_summary_val = np.column_stack([
        R_t0_val,
        R_delay_val,
        R_avg_val,
    ])

    rows, pred = evaluate_method(
        seed=seed,
        method="delay_summary",
        signature_order=np.nan,
        X_train=X_delay_summary_train,
        X_validation=X_delay_summary_val,
        train_targets=train_targets,
        validation_targets=validation_targets,
    )

    all_rows.extend(rows)
    predictions["delay_summary"] = pred


    print("Building rolling-window time-augmented paths...")
    paths_train = build_rolling_window_paths(R_train)
    paths_validation = build_rolling_window_paths(R_validation)

    for level in SIGNATURE_LEVELS:
        print(f"Computing rolling-window signatures of order {level}...")

        tic = time.time()

        sig_train = compute_signature_features(paths_train, level)
        sig_validation = compute_signature_features(paths_validation, level)

        runtime = time.time() - tic

        X_sig_train = np.column_stack([
            R_t0_train,
            sig_train,
        ])

        X_sig_val = np.column_stack([
            R_t0_val,
            sig_validation,
        ])

        print("Signature feature matrix shape:", X_sig_train.shape)
        print("Signature runtime:", runtime)

        rows, pred = evaluate_method(
            seed=seed,
            method="signature",
            signature_order=level,
            X_train=X_sig_train,
            X_validation=X_sig_val,
            train_targets=train_targets,
            validation_targets=validation_targets,
            signature_runtime=runtime,
        )

        all_rows.extend(rows)
        predictions[f"signature_N{level}"] = pred

    results_seed = pd.DataFrame(all_rows)

    return results_seed, predictions, validation_targets



def build_summary(results_df):
    # Step 1: average over maturity WITHIN each seed
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

    # Step 2: mean/std ACROSS the seed-level values
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
        .groupby(["method", "signature_order", "maturity"], dropna=False)
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
        ("markovian", "Markovian"),
        ("delay_pair", "Delay pair"),
        ("delay_summary", "Delay summary"),
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
    plt.title("Experiment 4: Nested-MC validation error versus signature order")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "multiseed_yield_error_vs_signature_order.png",
        ),
        dpi=300,
    )

    plt.show()
    plt.close()


def plot_error_by_maturity(maturity_summary):
    plt.figure(figsize=(10, 6))

    plot_specs = [
        ("markovian", np.nan, "Markovian"),
        ("delay_pair", np.nan, "Delay pair"),
        ("delay_summary", np.nan, "Delay summary"),
        ("signature", 1.0, "Signature N=1"),
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
    plt.title("Experiment 4: Yield error by maturity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "multiseed_yield_error_by_maturity.png",
        ),
        dpi=300,
    )

    plt.show()
    plt.close()


def plot_selected_penalties(summary):
    plt.figure(figsize=(9, 6))

    labels = []
    means = []
    stds = []

    order_methods = [
        ("markovian", np.nan, "Markovian"),
        ("delay_pair", np.nan, "Delay pair"),
        ("delay_summary", np.nan, "Delay summary"),
        ("signature", 1.0, "Sig N=1"),
        ("signature", 2.0, "Sig N=2"),
        ("signature", 3.0, "Sig N=3"),
        ("signature", 4.0, "Sig N=4"),
    ]

    for method, order, label in order_methods:
        if np.isnan(order):
            df = summary[
                (summary["method"] == method)
                & (summary["signature_order"].isna())
            ]
        else:
            df = summary[
                (summary["method"] == method)
                & (summary["signature_order"] == order)
            ]

        if len(df) > 0:
            labels.append(label)
            means.append(df["mean_selected_alpha"].iloc[0])
            stds.append(df["std_selected_alpha"].iloc[0])

    plt.bar(labels, means, yerr=stds, capsize=5)
    plt.yscale("log")
    plt.ylabel("Selected ridge penalty")
    plt.title("Experiment 4: Selected ridge penalties")
    plt.xticks(rotation=35)
    plt.grid(True, which="both", axis="y")
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "multiseed_selected_ridge_penalties.png",
        ),
        dpi=300,
    )

    plt.show()
    plt.close()


def plot_predicted_vs_nested_yields(predictions, validation_targets, seed):
    T = MATURITIES[-1]
    j = len(MATURITIES) - 1

    true_yield = price_to_yield(validation_targets[:, j], T)

    plt.figure(figsize=(7, 7))

    plot_specs = [
        ("markovian", "Markovian"),
        ("delay_pair", "Delay pair"),
        ("delay_summary", "Delay summary"),
        ("signature_N2", "Signature N=2"),
    ]

    for key, label in plot_specs:
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
    plt.title(f"Experiment 4: Predicted vs nested-MC yields, T={T:g}, seed={seed}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            f"predicted_vs_nested_yields_T{T:g}_seed{seed}.png",
        ),
        dpi=300,
    )

    plt.show()
    plt.close()


def main():
    start_time = time.time()

    print("\n================ Experiment 4: Delay Model With Stronger Benchmarks ================\n")

    all_results = []

    representative_predictions = None
    representative_targets = None
    representative_seed = SEEDS[0]

    for seed in SEEDS:
        results_seed, predictions_seed, validation_targets_seed = run_single_seed(seed)

        all_results.append(results_seed)

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
        "experiment4_multiseed_results.csv",
    )

    summary_path = os.path.join(
        OUTPUT_DIR,
        "experiment4_multiseed_summary.csv",
    )

    maturity_summary_path = os.path.join(
        OUTPUT_DIR,
        "experiment4_multiseed_maturity_summary.csv",
    )

    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    maturity_summary.to_csv(maturity_summary_path, index=False)

    print("\nSaved results to:")
    print(results_path)
    print(summary_path)
    print(maturity_summary_path)

    plot_error_vs_signature_order(summary)
    plot_error_by_maturity(maturity_summary)
    plot_selected_penalties(summary)

    if representative_predictions is not None:
        plot_predicted_vs_nested_yields(
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
