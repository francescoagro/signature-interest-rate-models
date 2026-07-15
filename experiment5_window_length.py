import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import iisignature

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


SEEDS = [11, 22, 33, 44, 55]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "results", "experiment5_multiseed"))
os.makedirs(OUTPUT_DIR, exist_ok=True)


KAPPA = 0.5
THETA = 0.05
ALPHA = 0.2
TAU = 1.0
SIGMA = 0.01
R0 = 0.05


DT = 0.01
T0 = 3.0
T_MAX = 8.0
MATURITIES = np.array([4.0, 5.0, 7.0])


N_PATHS = 12000
N_TRAIN = 10000
N_VALIDATION = 300
N_NESTED = 500


SIGNATURE_ORDER = 3

WINDOWS = {
    "0.5tau": 0.5 * TAU,
    "tau": TAU,
    "2tau": 2.0 * TAU,
    "full_history": None,
}

WINDOW_ORDER = ["0.5tau", "tau", "2tau", "full_history"]

RIDGE_ALPHAS = np.logspace(-6, 4, 25)


def integrate_trapezoid(values, dx, axis=1):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(values, dx=dx, axis=axis)
    return np.trapz(values, dx=dx, axis=axis)


def simulate_delay_paths(n_paths, times, seed):
    rng = np.random.default_rng(seed)

    n_steps = len(times) - 1
    delay_steps = int(round(TAU / DT))

    R = np.zeros((n_paths, n_steps + 1))
    R[:, 0] = R0

    for i in range(n_steps):
        if i - delay_steps >= 0:
            R_delay = R[:, i - delay_steps]
        else:
            R_delay = R0

        drift = KAPPA * (THETA - R[:, i] - ALPHA * R_delay)
        dW = np.sqrt(DT) * rng.standard_normal(n_paths)

        R[:, i + 1] = R[:, i] + drift * DT + SIGMA * dW

    return R


def realized_discounted_payoffs(R_paths, t0, maturities):
    t0_idx = int(round(t0 / DT))
    payoffs = []

    for T in maturities:
        T_idx = int(round(T / DT))
        integral = integrate_trapezoid(
            R_paths[:, t0_idx:T_idx + 1],
            dx=DT,
            axis=1,
        )
        payoffs.append(np.exp(-integral))

    return np.column_stack(payoffs)


def nested_mc_prices_from_histories(R_histories, times, t0, maturities, seed):
    rng = np.random.default_rng(seed)

    t0_idx = int(round(t0 / DT))
    n_steps = len(times) - 1
    delay_steps = int(round(TAU / DT))

    n_histories = R_histories.shape[0]
    nested_prices = np.zeros((n_histories, len(maturities)))

    for h in range(n_histories):
        R_nested = np.zeros((N_NESTED, n_steps + 1))
        R_nested[:, :t0_idx + 1] = R_histories[h, :t0_idx + 1]

        for i in range(t0_idx, n_steps):
            if i - delay_steps >= 0:
                R_delay = R_nested[:, i - delay_steps]
            else:
                R_delay = R0

            drift = KAPPA * (THETA - R_nested[:, i] - ALPHA * R_delay)
            dW = np.sqrt(DT) * rng.standard_normal(N_NESTED)

            R_nested[:, i + 1] = R_nested[:, i] + drift * DT + SIGMA * dW

        for j, T in enumerate(maturities):
            T_idx = int(round(T / DT))
            integral = integrate_trapezoid(
                R_nested[:, t0_idx:T_idx + 1],
                dx=DT,
                axis=1,
            )
            nested_prices[h, j] = np.mean(np.exp(-integral))

    return nested_prices


def build_time_augmented_segment(R_path, times, start_idx, end_idx):
    t_segment = times[start_idx:end_idx + 1]
    r_segment = R_path[start_idx:end_idx + 1]

    t_relative = t_segment - t_segment[0]

    return np.column_stack([t_relative, r_segment])


def compute_signature_features(R_paths, times, t0, window_length, order):
    t0_idx = int(round(t0 / DT))

    if window_length is None:
        start_idx = 0
        effective_window = t0
    else:
        start_time = max(0.0, t0 - window_length)
        start_idx = int(round(start_time / DT))
        effective_window = t0 - start_time

    features = []

    for i in range(R_paths.shape[0]):
        path = build_time_augmented_segment(
            R_paths[i],
            times,
            start_idx,
            t0_idx,
        )

        sig = iisignature.sig(path, order)

        current_R = R_paths[i, t0_idx]

        feature_vector = np.concatenate([[current_R], sig])
        features.append(feature_vector)

    return np.asarray(features), effective_window


def fit_and_evaluate(
    seed,
    window_label,
    window_length,
    X_train,
    Y_train_price,
    X_val,
    Y_val_price,
    maturities,
    runtime,
):
    rows = []

    for j, T in enumerate(maturities):
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=RIDGE_ALPHAS, cv=5)),
        ])

        model.fit(X_train, Y_train_price[:, j])

        pred_price = model.predict(X_val)
        pred_price = np.clip(pred_price, 1e-8, 1.5)

        true_price = Y_val_price[:, j]

        tau = T - T0

        pred_yield = -np.log(pred_price) / tau
        true_yield = -np.log(true_price) / tau

        price_rmse = np.sqrt(mean_squared_error(true_price, pred_price))
        yield_rmse_bp = np.sqrt(mean_squared_error(true_yield, pred_yield)) * 10000
        r2_price = r2_score(true_price, pred_price)

        rows.append({
            "seed": seed,
            "window_label": window_label,
            "window_length": window_length,
            "signature_order": SIGNATURE_ORDER,
            "maturity": T,
            "price_rmse": price_rmse,
            "yield_rmse_bp": yield_rmse_bp,
            "r2_price": r2_price,
            "selected_alpha": model.named_steps["ridge"].alpha_,
            "n_features": X_train.shape[1],
            "signature_runtime": runtime,
        })

    return rows



def run_single_seed(seed):
    print(f"\n================ Experiment 5 | Seed {seed} ================\n")

    times = np.arange(0.0, T_MAX + DT, DT)

    print("Simulating delay-model paths...")
    R_paths = simulate_delay_paths(N_PATHS, times, seed)
    print("R_paths shape:", R_paths.shape)

    print("Computing realized discounted payoffs...")
    payoff_targets_all = realized_discounted_payoffs(
        R_paths,
        T0,
        MATURITIES,
    )

    R_train = R_paths[:N_TRAIN]
    Y_train_price = payoff_targets_all[:N_TRAIN]

    R_validation_histories = R_paths[N_TRAIN:N_TRAIN + N_VALIDATION]

    print("Computing nested Monte Carlo labels...")
    Y_validation_price = nested_mc_prices_from_histories(
        R_validation_histories,
        times,
        T0,
        MATURITIES,
        seed=seed + 1000,
    )

    all_rows = []

    for window_label, window_length in WINDOWS.items():
        print(f"\nWindow: {window_label}")

        tic = time.time()

        X_train, effective_window_train = compute_signature_features(
            R_train,
            times,
            T0,
            window_length,
            SIGNATURE_ORDER,
        )

        X_val, effective_window_val = compute_signature_features(
            R_validation_histories,
            times,
            T0,
            window_length,
            SIGNATURE_ORDER,
        )

        runtime = time.time() - tic

        effective_window = effective_window_train

        print("Feature matrix shape:", X_train.shape)
        print("Effective window length:", effective_window)
        print("Signature runtime:", runtime)

        rows = fit_and_evaluate(
            seed=seed,
            window_label=window_label,
            window_length=effective_window,
            X_train=X_train,
            Y_train_price=Y_train_price,
            X_val=X_val,
            Y_val_price=Y_validation_price,
            maturities=MATURITIES,
            runtime=runtime,
        )

        all_rows.extend(rows)

    return pd.DataFrame(all_rows)



def build_summary(results_df):
    summary_df = (
        results_df
        .groupby("window_label")
        .agg(
            window_length=("window_length", "first"),
            signature_order=("signature_order", "first"),
            n_features=("n_features", "first"),

            mean_price_rmse=("price_rmse", "mean"),
            std_price_rmse=("price_rmse", "std"),

            mean_yield_rmse_bp=("yield_rmse_bp", "mean"),
            std_yield_rmse_bp=("yield_rmse_bp", "std"),

            mean_r2_price=("r2_price", "mean"),
            std_r2_price=("r2_price", "std"),

            mean_selected_alpha=("selected_alpha", "mean"),
            std_selected_alpha=("selected_alpha", "std"),

            mean_signature_runtime=("signature_runtime", "mean"),
            std_signature_runtime=("signature_runtime", "std"),
        )
        .reset_index()
    )

    summary_df["window_label"] = pd.Categorical(
        summary_df["window_label"],
        categories=WINDOW_ORDER,
        ordered=True,
    )

    summary_df = summary_df.sort_values("window_label").reset_index(drop=True)

    return summary_df


def build_maturity_summary(results_df):
    maturity_summary_df = (
        results_df
        .groupby(["window_label", "maturity"])
        .agg(
            window_length=("window_length", "first"),
            signature_order=("signature_order", "first"),
            n_features=("n_features", "first"),

            mean_price_rmse=("price_rmse", "mean"),
            std_price_rmse=("price_rmse", "std"),

            mean_yield_rmse_bp=("yield_rmse_bp", "mean"),
            std_yield_rmse_bp=("yield_rmse_bp", "std"),

            mean_r2_price=("r2_price", "mean"),
            std_r2_price=("r2_price", "std"),

            mean_selected_alpha=("selected_alpha", "mean"),
            std_selected_alpha=("selected_alpha", "std"),

            mean_signature_runtime=("signature_runtime", "mean"),
            std_signature_runtime=("signature_runtime", "std"),
        )
        .reset_index()
    )

    maturity_summary_df["window_label"] = pd.Categorical(
        maturity_summary_df["window_label"],
        categories=WINDOW_ORDER,
        ordered=True,
    )

    maturity_summary_df = maturity_summary_df.sort_values(
        ["window_label", "maturity"]
    ).reset_index(drop=True)

    return maturity_summary_df


def plot_rmse_vs_window(summary_df):
    plt.figure(figsize=(8, 5))

    plt.errorbar(
        summary_df["window_label"].astype(str),
        summary_df["mean_yield_rmse_bp"],
        yerr=summary_df["std_yield_rmse_bp"],
        marker="o",
        capsize=5,
    )

    plt.xlabel("Memory window")
    plt.ylabel("Average yield RMSE (bp)")
    plt.title("Experiment 5: Rolling window versus full history")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "experiment5_multiseed_rmse_vs_window.png",
        ),
        dpi=300,
    )

    plt.show()
    plt.close()


def plot_error_by_maturity(maturity_summary_df):
    plt.figure(figsize=(8, 5))

    for label, group in maturity_summary_df.groupby(
        "window_label",
        observed=False,
    ):
        group = group.sort_values("maturity")

        plt.errorbar(
            group["maturity"],
            group["mean_yield_rmse_bp"],
            yerr=group["std_yield_rmse_bp"],
            marker="o",
            capsize=5,
            label=str(label),
        )

    plt.xlabel("Maturity T")
    plt.ylabel("Yield RMSE (bp)")
    plt.title("Experiment 5: Yield error by maturity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "experiment5_multiseed_error_by_maturity.png",
        ),
        dpi=300,
    )

    plt.show()
    plt.close()


def plot_runtime_vs_window(summary_df):
    plt.figure(figsize=(8, 5))

    plt.errorbar(
        summary_df["window_label"].astype(str),
        summary_df["mean_signature_runtime"],
        yerr=summary_df["std_signature_runtime"],
        marker="o",
        capsize=5,
    )

    plt.xlabel("Memory window")
    plt.ylabel("Signature runtime (seconds)")
    plt.title("Experiment 5: Runtime versus memory window")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "experiment5_multiseed_runtime_vs_window.png",
        ),
        dpi=300,
    )

    plt.show()
    plt.close()



def main():
    start_time = time.time()

    print("\n================ Experiment 5: Multi-seed Rolling Window Versus Full History ================\n")

    all_results = []

    for seed in SEEDS:
        seed_results = run_single_seed(seed)
        all_results.append(seed_results)

    results_df = pd.concat(all_results, ignore_index=True)

    summary_df = build_summary(results_df)
    maturity_summary_df = build_maturity_summary(results_df)

    results_path = os.path.join(
        OUTPUT_DIR,
        "experiment5_multiseed_results.csv",
    )

    summary_path = os.path.join(
        OUTPUT_DIR,
        "experiment5_multiseed_summary.csv",
    )

    maturity_summary_path = os.path.join(
        OUTPUT_DIR,
        "experiment5_multiseed_maturity_summary.csv",
    )

    results_df.to_csv(results_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    maturity_summary_df.to_csv(maturity_summary_path, index=False)

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

    plot_rmse_vs_window(summary_df)
    plot_error_by_maturity(maturity_summary_df)
    plot_runtime_vs_window(summary_df)

    print("\nSaved plots to:")
    print(OUTPUT_DIR)

    print("\nTotal runtime:", time.time() - start_time)
    print("==========================================================================\n")


if __name__ == "__main__":
    main()
