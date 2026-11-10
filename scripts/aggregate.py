"""Turn runs/ into the paper's main table: MAE per dataset, width and method, and RefAnchor-SOH against the strongest control.

  python scripts/aggregate.py

Uses whatever is present in runs/. Output: results/table_mae.csv and results/table_comparison.csv.
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from refanchor.stats import clip, compare  # noqa: E402

CONTROLS = ("b1", "b2", "cnn", "lstm", "transformer", "b5", "b1h")
OTHERS = ("b0", "cnn_residual", "v2", "v3", "v4", "m1h_nodelta", "m1h_meanref")


def load(run_dir, method, y):
    """Predictions per seed (trees: one file per seed; neural: one file per fold and seed)."""
    runs = []
    for s in range(5):
        f = os.path.join(run_dir, f"{method}_s{s}.npz")
        if os.path.exists(f):
            runs.append(np.load(f)["pred"])
            continue
        p = np.full_like(y, np.nan)
        for k in range(5):
            f = os.path.join(run_dir, f"{method}_f{k}_s{s}.npz")
            if os.path.exists(f):
                r = np.load(f)
                p[r["te"]] = r["pred"]
        if np.isfinite(p).all():
            runs.append(p)
    return clip(np.array(runs)) if runs else None


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--cache-dir", default="cache")
    a.add_argument("--runs", default="runs")
    a.add_argument("--out", default="results")
    a = a.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows, comp = [], []
    for ds in ("snl", "isu", "tongji"):
        path = os.path.join(a.cache_dir, ds + ".npz")
        if not os.path.exists(path):
            continue
        D = np.load(path)
        y, W, cells = D["y"], np.round(D["w"], 3), D["cells"]
        P = {m: load(os.path.join(a.runs, ds), m, y) for m in CONTROLS + OTHERS + ("m1h",)}
        P = {m: v for m, v in P.items() if v is not None}
        for w in sorted(set(W)):
            k = W == w
            maes = {m: float((100 * np.abs(v[:, k] - y[k]).mean(1)).mean()) for m, v in P.items()}
            rows += [[ds, w, m, round(x, 3)] for m, x in maes.items()]
            ctl = [m for m in CONTROLS if m in maes]
            if "m1h" in maes and ctl:
                best = min(ctl, key=maes.get)
                c = compare(P["m1h"][:, k].mean(0), P[best][:, k].mean(0), y[k], cells[k])
                comp.append([ds, w, best, round(maes["m1h"], 3), round(maes[best], 3), round(100 * (1 - maes["m1h"] / maes[best]), 1), round(c["ci_low"], 1), round(c["ci_high"], 1), f"{c['p']:.2e}", f"{c['better']}/{c['cells']}"])
                print(*comp[-1])
    with open(os.path.join(a.out, "table_mae.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "width_V", "method", "MAE_pct"])
        w.writerows(rows)
    with open(os.path.join(a.out, "table_comparison.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "width_V", "strongest_control", "RefAnchor_MAE", "control_MAE", "reduction_pct", "CI_low", "CI_high", "wilcoxon_p", "cells_better"])
        w.writerows(comp)


if __name__ == "__main__":
    main()
