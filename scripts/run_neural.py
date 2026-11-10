"""Neural controls on one or all folds. Finished files are skipped. A CUDA GPU is recommended.

  python scripts/run_neural.py --dataset snl                                  # 4 variants x 5 folds x 5 seeds
  python scripts/run_neural.py --dataset snl --variants cnn --folds 0 --seeds 0

Output: runs/<dataset>/<variant>_f<fold>_s<seed>.npz (test indices, predictions).
"""
import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from refanchor.neural import VARIANTS, run_fold  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=("snl", "isu", "tongji"))
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--out", default="runs")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    p.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--device", default="cuda")
    p.add_argument("--memory-fraction", type=float, help="limit on GPU memory, for running several jobs on one small GPU")
    a = p.parse_args()
    D = dict(np.load(os.path.join(a.cache_dir, a.dataset + ".npz")))
    out = os.path.join(a.out, a.dataset)
    os.makedirs(out, exist_ok=True)
    for v, f, s in itertools.product(a.variants, a.folds, a.seeds):
        fn = os.path.join(out, f"{v}_f{f}_s{s}.npz")
        if os.path.exists(fn):
            continue
        te, pred, val = run_fold(D, v, f, s, a.device, a.memory_fraction)
        np.savez_compressed(fn, te=te, pred=pred)
        print(a.dataset, v, f, s, f"val MAE {100 * val:.3f}, test MAE {100 * np.abs(pred - D['y'][te]).mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
