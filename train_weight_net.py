#!/usr/bin/env python3
"""Weighted conformal prediction: train per-scenario weight networks and evaluate coverage.

For each task, loads pre-computed splits from splits/<task>.txt and
normalized profile data from profiles/<task>/normalized_data_delta=0.01.csv.
Test inputs are clustered into N scenarios by median execution time
(max insn_sum per constant-budget run). For each (task, scenario, target)
a WeightedConformalPredictor is trained on D_wt / D_cal from that scenario
and empirical coverage is evaluated on D_test_cp from that scenario.

Results are written to results/wcp_results.txt.

Usage:
    python train_weight_net.py
    python train_weight_net.py --tasks mcf xz --n-clusters 4 --delta 0.1
"""

import argparse
import os
import warnings

os.environ["MKL_VERBOSE"] = "0"
warnings.filterwarnings("ignore", message=".*Intel MKL.*")

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
import xgboost as xgb

from cp import WeightedConformalPredictor

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 12

DELTA_TAG = "delta=0.01"
TASK_TYPE = "spec"
OUTPUT_FEATURES = ["rem_time", "next_insn"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_features(task: str, features_file: str = "spec_input_feature_map.txt") -> list[str]:
    with open(features_file) as f:
        for line in f:
            parts = line.strip().split(",", 1)
            if parts[0] == task:
                return [x.strip() for x in parts[1].split(",")]
    raise ValueError(f"No features found for task '{task}'")


def load_splits(task: str) -> dict[str, list[str]]:
    path = os.path.join("splits", f"{task}.txt")
    result = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, names_str = line.split(":", 1)
            result[key] = [n.strip() for n in names_str.split(",") if n.strip()]
    return result


class _BoosterWrapper:
    """Wraps a single xgb.Booster; returns 1-D predictions."""

    def __init__(self, booster: xgb.Booster):
        self.booster = booster

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.booster.predict(xgb.DMatrix(X))


def load_xgb_model(task: str, target_idx: int) -> _BoosterWrapper:
    path = os.path.join("models", TASK_TYPE, task, DELTA_TAG,
                        f"xgboost_model.json_target_{target_idx}.model")
    b = xgb.Booster()
    b.load_model(path)
    return _BoosterWrapper(b)


def compute_exec_times(norm_df: pd.DataFrame) -> pd.Series:
    """Return per-input median execution time proxy.

    Uses max(insn_sum) per constant-budget run. insn_sum is globally
    normalized, so its maximum within a run reflects total execution length.
    Falls back to all runs if no constant-budget runs are found.
    """
    run_b = norm_df.groupby("run_id")["prev_budget"].agg(["min", "max"])
    const_runs = run_b[run_b["min"] == run_b["max"]].index
    ref = norm_df[norm_df["run_id"].isin(const_runs)] if len(const_runs) > 0 else norm_df
    et = (
        ref.groupby(["inp_name", "run_id"])["insn_sum"]
        .max()
        .reset_index(name="exec_time")
    )
    return et.groupby("inp_name")["exec_time"].median()


def cluster_by_exec_time(
    inp_names: list[str], exec_times: pd.Series, n_clusters: int
) -> dict[int, list[str]]:
    """Assign inp_names to quartile-based execution-time clusters.

    Returns {cluster_id: [inp_name, ...]} sorted by ascending execution time.
    """
    times = exec_times.reindex(inp_names).fillna(exec_times.median())
    df = pd.DataFrame({"inp_name": inp_names, "t": times.values}).sort_values("t")
    n = min(n_clusters, len(df))
    if n <= 1:
        return {0: df["inp_name"].tolist()}
    labels = pd.qcut(df["t"], n, labels=range(n), duplicates="drop").astype(int)
    df["k"] = labels.values
    return {k: df[df["k"] == k]["inp_name"].tolist() for k in sorted(df["k"].unique())}


def plot_cluster_histogram(
    task: str,
    norm_df: pd.DataFrame,
    clusters: dict[int, list[str]],
    save_dir: str,
) -> None:
    """Save a stacked insn_sum histogram with one color per scenario cluster."""
    colors = plt.cm.tab10.colors
    all_labels = [f"Scenario {k}" for k in clusters]
    patches = [None] * len(clusters)

    plt.figure(figsize=(6, 5))
    for k in reversed(range(len(clusters))):
        inp_set = set(clusters[k])
        data = norm_df[norm_df["inp_name"].isin(inp_set)]["insn_sum"].values
        alpha = 0.35 + 0.05 * (len(clusters) - k - 1)
        _, _, patch = plt.hist(data, bins=100, color=colors[k % len(colors)],
                               alpha=alpha, histtype="stepfilled", linewidth=0)
        patches[k] = patch[0]

    plt.xlabel("Cumulative Instruction Count", fontsize=18)
    plt.ylabel("Count", fontsize=18)
    ax = plt.gca()
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(fmt)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(left=0)
    plt.legend(patches, all_labels, fontsize=16)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f"{task}_cluster_insn_sum_hist.pdf")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved cluster histogram to: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Tasks to run (default: all in spec_input_feature_map.txt)")
    parser.add_argument("--features-file", default="spec_input_feature_map.txt")
    parser.add_argument("--n-clusters", type=int, default=4,
                        help="Number of execution-time scenarios (default: 4)")
    parser.add_argument("--delta", type=float, default=0.1,
                        help="Miscoverage level (default: 0.1)")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 32, 8])
    parser.add_argument("--lambda_", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--output", default=os.path.join("results", "wcp_results.txt"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.tasks is None:
        tasks = []
        with open(args.features_file) as f:
            for line in f:
                t = line.strip().split(",")[0]
                if t:
                    tasks.append(t)
    else:
        tasks = args.tasks

    os.makedirs("results", exist_ok=True)

    all_rows = []

    for task in tasks:
        print(f"\n{'='*60}\nTask: {task}\n{'='*60}")

        input_features = load_features(task, args.features_file)
        splits = load_splits(task)

        wt_set  = set(splits.get("weights", []))
        cal_set = set(splits.get("calibration", []))
        tcp_set = set(splits.get("test_cp", []))
        test_all = splits.get("test", [])

        data_path = os.path.join("profiles", task, f"normalized_data_{DELTA_TAG}.csv")
        print(f"Loading {data_path}...")
        norm_df = pd.read_csv(data_path)
        inp_name_col = norm_df["inp_name"].astype(str).values

        exec_times = compute_exec_times(norm_df)
        clusters = cluster_by_exec_time(test_all, exec_times, args.n_clusters)

        print(f"Scenarios ({len(clusters)}):")
        for k, names in clusters.items():
            lo = exec_times.reindex(names).min()
            hi = exec_times.reindex(names).max()
            print(f"  {k}: {len(names)} inputs  exec_time=[{lo:.3e}, {hi:.3e}]")

        plot_cluster_histogram(task, norm_df, clusters, f"results/{task}")

        models = [load_xgb_model(task, i) for i in range(len(OUTPUT_FEATURES))]

        X_all = norm_df[input_features].values.astype(np.float32)
        y_all = norm_df[OUTPUT_FEATURES].values.astype(np.float32)

        for k, cluster_names in clusters.items():
            cs = set(cluster_names)
            wt_k  = cs & wt_set
            cal_k = cs & cal_set
            tcp_k = cs & tcp_set

            if not wt_k or not cal_k or not tcp_k:
                print(f"\n  Scenario {k}: skipping "
                      f"(wt={len(wt_k)}, cal={len(cal_k)}, test={len(tcp_k)})")
                continue

            mask_wt  = np.isin(inp_name_col, list(wt_k))
            mask_cal = np.isin(inp_name_col, list(cal_k))
            mask_tcp = np.isin(inp_name_col, list(tcp_k))

            X_wt,  y_wt  = X_all[mask_wt],  y_all[mask_wt]
            X_cal, y_cal = X_all[mask_cal], y_all[mask_cal]
            X_tcp, y_tcp = X_all[mask_tcp], y_all[mask_tcp]

            lo = float(exec_times.reindex(cluster_names).min())
            hi = float(exec_times.reindex(cluster_names).max())
            exec_range = f"[{lo:.3e}, {hi:.3e}]"

            print(f"\n  Scenario {k}  exec={exec_range}"
                  f"  n_wt={mask_wt.sum()}, n_cal={mask_cal.sum()}, n_test={mask_tcp.sum()}")

            for target_idx, target_name in enumerate(OUTPUT_FEATURES):
                cp = WeightedConformalPredictor(models[target_idx], delta=args.delta)
                cp.calibrate(
                    X_cal, y_cal[:, target_idx],
                    X_wt=X_wt, y_wt=y_wt[:, target_idx],
                    lambda_=args.lambda_,
                    beta=args.beta,
                    lr=args.lr,
                    epochs=args.epochs,
                    hidden_dims=tuple(args.hidden_dims),
                    device=args.device,
                    verbose=args.verbose,
                    batch_size=args.batch_size,
                )
                cov = cp.evaluate_coverage(X_tcp, y_tcp[:, target_idx])
                coverage = float(cov["coverage"])
                print(f"    {target_name}: q={cp.q:.4f}  "
                      f"coverage={coverage:.4f}  "
                      f"(target={1-args.delta:.0%}, n={cov['n_total']})")

                all_rows.append({
                    "task":            task,
                    "scenario":        str(k),
                    "target":          target_name,
                    "exec_range":      exec_range,
                    "n_wt":            str(mask_wt.sum()),
                    "n_cal":           str(mask_cal.sum()),
                    "n_test":          str(cov["n_total"]),
                    "q":               f"{cp.q:.6f}",
                    "coverage":        f"{coverage:.4f}",
                    "target_coverage": f"{1 - args.delta:.2f}",
                })

    if not all_rows:
        print("\nNo results to write.")
        return

    headers = ["task", "scenario", "target", "exec_range",
               "n_wt", "n_cal", "n_test", "q", "coverage", "target_coverage"]
    col_w = {h: max(len(h), max(len(r[h]) for r in all_rows)) for h in headers}
    fmt = lambda r: " | ".join(str(r[h]).ljust(col_w[h]) for h in headers)
    sep = "-+-".join("-" * col_w[h] for h in headers)
    lines = ["", fmt({h: h for h in headers}), sep] + [fmt(r) for r in all_rows] + [""]
    out_str = "\n".join(lines)

    with open(args.output, "w") as f:
        f.write(out_str + "\n")
    print(f"\n✓ Saved results to: {args.output}")
    print(out_str)


if __name__ == "__main__":
    main()
