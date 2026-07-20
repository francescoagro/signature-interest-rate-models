"""
Small step-size (Delta t) convergence check for the power-law Volterra
memory model of Experiment 6, requested during thesis review.

This script does NOT reimplement the simulation: it imports the actual
simulate_power_law_paths / realized_discounted_payoffs functions from
experiment6_power_law_kernel.py and reruns them at a few step sizes,
temporarily overriding the module-level DT (and T_MAX-dependent
constants derived from it) so that the exact production code path is
exercised at each resolution.

For each Delta t we report:
  - the mean and standard deviation of the memory variable M at t0
    (the object most directly affected by the cell-integration fix),
  - the mean nested-free (path-average) discounted bond price at each
    maturity.

These are independent Monte Carlo runs (different random draws per
Delta t, same representative seed), at the full N_PATHS used elsewhere
in Experiment 6, so residual differences reflect both discretization
bias and ordinary Monte Carlo noise. The purpose is not a formal
convergence-rate estimate but a sanity check that results are stable
well within the Monte Carlo noise floor as Delta t shrinks.
"""

import time

import numpy as np
import pandas as pd

import experiment6_power_law_kernel as e6

DT_VALUES = [0.02, 0.01, 0.005]
SEED = e6.SEEDS[0]
N_PATHS = e6.N_PATHS


def run_for_dt(dt):
    e6.DT = dt
    times, R_paths, M_paths = e6.simulate_power_law_paths(N_PATHS, e6.T_MAX, seed=SEED)
    t0_idx = e6.time_index(e6.T0)
    prices = e6.realized_discounted_payoffs(R_paths)
    row = {
        "dt": dt,
        "n_steps": R_paths.shape[1] - 1,
        "mean_M_t0": M_paths[:, t0_idx].mean(),
        "std_M_t0": M_paths[:, t0_idx].std(),
    }
    for j, T in enumerate(e6.MATURITIES):
        row[f"mean_price_T{T:g}"] = prices[:, j].mean()
        row[f"mean_yield_T{T:g}_bp"] = e6.price_to_yield(prices[:, j], T).mean() * 10000
    return row


def main():
    original_dt = e6.DT
    start = time.time()
    rows = [run_for_dt(dt) for dt in DT_VALUES]
    e6.DT = original_dt  # restore, in case this module is imported elsewhere

    df = pd.DataFrame(rows)
    print("\n================ Experiment 6: Delta t convergence check ================\n")
    print(df.to_string(index=False))
    print("\nRuntime:", time.time() - start, "seconds")

    out_path = f"{e6.OUTPUT_DIR}/experiment6_dt_convergence.csv"
    df.to_csv(out_path, index=False)
    print("Saved to:", out_path)


if __name__ == "__main__":
    main()
