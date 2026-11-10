"""Five-fold cross-validation over condition groups for the tree methods. Finished files are skipped.

  python scripts/run_cv.py --dataset snl --data-dir <BatteryLife>/SNL                   # all tree methods, five seeds
  python scripts/run_cv.py --dataset snl --data-dir <BatteryLife>/SNL --methods m1h --seeds 0

Methods: b0 anchor alone, b1 window, b2 IC features + GP, b5 cycle + condition, b1h window + cycle + condition, v2 v3 v4
variants, m1h RefAnchor-SOH; m1h_nodelta and m1h_meanref use the caches of the ablations. Output: runs/<dataset>/<method>_s<seed>.npz.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from refanchor.model import cross_validate, gpr_cross_validate, protocols  # noqa: E402
from refanchor.stats import mae  # noqa: E402

ALL = ("b0", "b1", "b2", "b5", "b1h", "v2", "v3", "v4", "m1h", "m1h_nodelta", "m1h_meanref")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=("snl", "isu", "tongji"))
    p.add_argument("--data-dir", help="BatteryLife folder with the .pkl files (not needed for Tongji)")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--out", default="runs")
    p.add_argument("--methods", nargs="+", default=list(ALL), choices=ALL)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--jobs", type=int, default=-1)
    a = p.parse_args()
    D = dict(np.load(os.path.join(a.cache_dir, a.dataset + ".npz")))
    pr = protocols(a.dataset, a.data_dir, D["cells"])
    out = os.path.join(a.out, a.dataset)
    os.makedirs(out, exist_ok=True)
    for m in a.methods:
        for seed in ([0] if m == "b0" else a.seeds):
            fn = os.path.join(out, f"{m}_s{seed}.npz")
            if os.path.exists(fn):
                continue
            if m == "b2":
                pred = gpr_cross_validate(D, seed)
            elif m.startswith("m1h_"):
                T = dict(np.load(os.path.join(a.cache_dir, f"{a.dataset}_{m[4:]}.npz")))
                pred = cross_validate(T, pr, "m1h", seed, a.jobs)
            else:
                pred = cross_validate(D, pr, m, seed, a.jobs)
            np.savez_compressed(fn, pred=pred)
            print(a.dataset, m, seed, f"MAE {mae(pred, D['y']):.3f}", flush=True)


if __name__ == "__main__":
    main()
