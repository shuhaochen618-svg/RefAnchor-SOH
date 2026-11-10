"""The frozen test with conditions kept out of training: train without the target, test on all of it, five seeds.

  python scripts/run_unseen.py --dataset snl --data-dir <BatteryLife>/SNL
  python scripts/run_unseen.py --dataset isu --data-dir <BatteryLife>/ISU_ILCC

Prints, for windows of 0.2 V and wider, each target's MAE for RefAnchor-SOH and the controls, and the comparison with the
strongest of B1, B5 and B1h. Output: runs/unseen_<dataset>.json.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from refanchor.anchor import anchor  # noqa: E402
from refanchor.model import CLAIM_WIDTHS, RESIDUAL, fit_predict, held_out_targets, inputs, protocols  # noqa: E402
from refanchor.stats import cell_mae  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=("snl", "isu"))
    p.add_argument("--data-dir", required=True)
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--out", default="runs")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--jobs", type=int, default=-1)
    a = p.parse_args()
    D = dict(np.load(os.path.join(a.cache_dir, a.dataset + ".npz")))
    y, cells, W = D["y"], D["cells"], np.round(D["w"], 3)
    pr = protocols(a.dataset, a.data_dir, cells)
    X, base = inputs(D, pr), anchor(D["s"])
    res = {}
    for target, mask in held_out_targets(a.dataset, cells, pr).items():
        te, tr = np.where(mask)[0], np.where(~mask)[0]
        P = {m: np.mean([fit_predict(X[m], y, base if m in RESIDUAL else None, tr, te, s, a.jobs) for s in a.seeds], 0) for m in ("b1", "b5", "b1h", "m1h")}
        k = np.isin(W[te], CLAIM_WIDTHS)
        P["b0"] = base[te]
        u = np.unique(cells[te][k])
        cm = {m: cell_mae(v[k], y[te][k], cells[te][k], u) for m, v in P.items()}
        r = {m: round(100 * float(np.abs(v[k] - y[te][k]).mean()), 3) for m, v in P.items()}
        best = min(("b1", "b5", "b1h"), key=r.get)
        r.update(strongest=best, reduction_pct=round(100 * (1 - r["m1h"] / r[best]), 1), p=float(wilcoxon(cm["m1h"], cm[best], alternative="less").pvalue),
                 better=f"{int((cm['m1h'] < cm[best]).sum())}/{len(u)}")
        res[target] = r
        print(target, r, flush=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"unseen_{a.dataset}.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
