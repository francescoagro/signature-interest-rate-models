import os
import time
import numpy as np
import pandas as pd
import iisignature
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



os.makedirs("/Users/francescoagro/Desktop/TESI/code/results/experiment0", exist_ok=True)


seed = 42
np.random.seed(seed)

T_max = 10.0
n_steps = 1000
dt = T_max / n_steps
time_grid = np.linspace(0.0, T_max, n_steps + 1)

n_paths = 3000

t0 = 1.0
t0_index = int(t0 / dt)

maturities = np.array([2.0, 3.0, 5.0, 7.0, 10.0])

signature_order = 3
window_length = None


R0 = 0.05
kappa = 0.5
theta = 0.05
sigma = 0.01


ridge_alphas = np.logspace(-8, 4, 25)



def simulate_vasicek_paths(
    n_paths,
    n_steps,
    dt,
    R0,
    kappa,
    theta,
    sigma
):
    R = np.zeros((n_paths, n_steps + 1))
    R[:, 0] = R0

    for i in range(n_steps):
        dW = np.sqrt(dt) * np.random.randn(n_paths)
        R[:, i + 1] = (
            R[:, i]
            + kappa * (theta - R[:, i]) * dt
            + sigma * dW
        )

    return R



def build_time_augmented_path(rate_path, time_grid_slice):
    """
    Build Gamma_t = (t, R_t) for one path.
    The path is normalized before computing signatures.
    """

    t_scaled = time_grid_slice / time_grid_slice[-1]

    R_mean = np.mean(rate_path)
    R_std = np.std(rate_path)

    if R_std == 0:
        R_std = 1.0

    R_scaled = (rate_path - R_mean) / R_std

    return np.stack([t_scaled, R_scaled], axis=1)


def compute_signature_features(
    R_paths,
    time_grid,
    t0_index,
    order,
    window_length=None
):
    """
    Compute signature features at t0.

    If window_length is None, use full history [0,t0].
    Otherwise, use rolling window [t0-L,t0].
    """

    n_paths = R_paths.shape[0]
    path_dim = 2
    sig_dim = iisignature.siglength(path_dim, order)

    X = np.zeros((n_paths, sig_dim))

    if window_length is None:
        start_index = 0
    else:
        window_steps = int(window_length / dt)
        start_index = max(0, t0_index - window_steps)

    time_slice = time_grid[start_index:t0_index + 1]

    for p in range(n_paths):
        rate_slice = R_paths[p, start_index:t0_index + 1]
        gamma = build_time_augmented_path(rate_slice, time_slice)
        X[p, :] = iisignature.sig(gamma, order)

    return X


def compute_realized_log_bond_prices(
    R_paths,
    time_grid,
    t0_index,
    maturities
):
    """
    Compute realized log bond prices:

        log P(t0,T) approx - int_{t0}^{T} R_s ds

    This is not yet conditional pricing.
    It is used in Experiment 0 only to test the pipeline.
    """

    y_logP = []

    for T in maturities:
        T_index = int(T / dt)

        if T_index <= t0_index:
            raise ValueError("All maturities must be larger than t0.")

        integral = np.sum(R_paths[:, t0_index:T_index], axis=1) * dt
        logP = -integral
        y_logP.append(logP)

    return np.column_stack(y_logP)


def log_prices_to_yields(logP, t0, maturities):
    taus = maturities - t0
    return -logP / taus



def fit_ridge_model(X, y):
    """
    Ridge regression with:
    - train/test split;
    - standardization using training data only;
    - cross-validation over ridge penalty.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=ridge_alphas))
    ])

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "test_mae": mean_absolute_error(y_test, y_pred_test),
        "test_r2": r2_score(y_test, y_pred_test),
        "selected_alpha": model.named_steps["ridge"].alpha_
    }

    return model, metrics, y_test, y_pred_test



def plot_predicted_vs_true(y_true, y_pred, filename):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.4)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val])
    plt.xlabel("True value")
    plt.ylabel("Predicted value")
    plt.title("Predicted vs true")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()
    plt.close()



start_time = time.time()

print("\n================ Experiment 0: Pipeline Test ================")

print("\nSimulating paths...")
R_paths = simulate_vasicek_paths(
    n_paths=n_paths,
    n_steps=n_steps,
    dt=dt,
    R0=R0,
    kappa=kappa,
    theta=theta,
    sigma=sigma
)

print("R_paths shape:", R_paths.shape)

print("\nComputing signature features...")
X_sig = compute_signature_features(
    R_paths=R_paths,
    time_grid=time_grid,
    t0_index=t0_index,
    order=signature_order,
    window_length=window_length
)

print("Signature feature matrix shape:", X_sig.shape)

print("\nComputing targets...")
Y_logP = compute_realized_log_bond_prices(
    R_paths=R_paths,
    time_grid=time_grid,
    t0_index=t0_index,
    maturities=maturities
)

Y_yield = log_prices_to_yields(
    logP=Y_logP,
    t0=t0,
    maturities=maturities
)

print("Log-price target shape:", Y_logP.shape)
print("Yield target shape:", Y_yield.shape)

results = []

for j, T in enumerate(maturities):
    print(f"\n--- Maturity T = {T} ---")

    y = Y_logP[:, j]

    model, metrics, y_test, y_pred_test = fit_ridge_model(X_sig, y)

    tau = T - t0
    yield_true = -y_test / tau
    yield_pred = -y_pred_test / tau

    yield_rmse_bp = (
        np.sqrt(mean_squared_error(yield_true, yield_pred)) * 10000
    )

    result = {
        "maturity": T,
        "tau": tau,
        "signature_order": signature_order,
        "signature_dimension": X_sig.shape[1],
        "train_rmse_logP": metrics["train_rmse"],
        "test_rmse_logP": metrics["test_rmse"],
        "test_mae_logP": metrics["test_mae"],
        "test_r2_logP": metrics["test_r2"],
        "yield_rmse_bp": yield_rmse_bp,
        "selected_alpha": metrics["selected_alpha"]
    }

    results.append(result)

    print("Test RMSE log price:", metrics["test_rmse"])
    print("Test R2 log price:", metrics["test_r2"])
    print("Yield RMSE bp:", yield_rmse_bp)
    print("Selected ridge alpha:", metrics["selected_alpha"])

    plot_predicted_vs_true(
        y_true=y_test,
        y_pred=y_pred_test,
        filename=f"/Users/francescoagro/Desktop/TESI/code/results/experiment0/pred_vs_true_logP_T_{T}.png"
    )

results_df = pd.DataFrame(results)

results_df.to_csv(
    "/Users/francescoagro/Desktop/TESI/code/results/experiment0/experiment0_results.csv",
    index=False
)

elapsed_time = time.time() - start_time

print("\n================ Final Results ================")
print(results_df)

print("\nSaved table to:")
print("results/experiment0/experiment0_results.csv")

print("\nSaved plots to:")
print("results/experiment0/")

print("\nTotal runtime:", elapsed_time)
print("================================================\n")