"""Build a window cache from the processed BatteryLife cells.

  python scripts/build_windows.py --dataset snl --data-dir <BatteryLife>/SNL
  python scripts/build_windows.py --dataset isu --data-dir <BatteryLife>/ISU_ILCC --rpt-dir rpt
  python scripts/build_windows.py --dataset tongji --data-dir <BatteryLife>/Tongji

Options select the ablations and the robustness study; the defaults give the method of the paper:
  --anchor nodelta | meanref       --soh-floor 0 (no cut at SOH 0.7)
  --perturb down --level 2         two points per 0.1 V
  --perturb vnoise --level 0.002   2 mV voltage noise
  --perturb qdrift --level 0.01    1 % charge gain drift
  --perturb reflate                reference from usable cycles 10-12

Only the cells in splits/cells.csv are used. A default build checks its arrays against splits/cache_checksums.json.
"""
import argparse
import csv
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from refanchor.data import Config, build, list_cells  # noqa: E402


def checksum(path):
    D = np.load(path)
    h = hashlib.sha256()
    for k in ("seq", "sca", "y", "s", "w", "cyc", "v1"):
        h.update(np.round(D[k].astype(np.float64), 5).tobytes())
    h.update("|".join(D["cells"].tolist()).encode())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=("snl", "isu", "tongji"))
    p.add_argument("--data-dir", required=True)
    p.add_argument("--rpt-dir", help="output of prepare_isu_rpt.py (ISU-ILCC only)")
    p.add_argument("--anchor", default="full", choices=("full", "nodelta", "meanref"))
    p.add_argument("--soh-floor", type=float, default=0.7)
    p.add_argument("--perturb", default="none", choices=("none", "down", "vnoise", "qdrift", "reflate"))
    p.add_argument("--level", type=float, default=0.0)
    p.add_argument("--out", help="default cache/<dataset>[_variant].npz")
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    if a.dataset == "isu" and not a.rpt_dir:
        p.error("--rpt-dir is required for ISU-ILCC")
    tag = "_".join(x for x in (a.anchor if a.anchor != "full" else "", f"{a.perturb}{a.level:g}" if a.perturb != "none" else "", "fullrange" if a.soh_floor != 0.7 else "") if x)
    out = a.out or os.path.join("cache", a.dataset + ("_" + tag if tag else "") + ".npz")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(os.path.join(ROOT, "splits", "cells.csv"), encoding="utf-8") as fh:
        keep = {r["cell"] for r in csv.DictReader(fh) if r["dataset"] == a.dataset}
    cfg = Config(a.dataset, a.anchor, a.soh_floor, a.perturb, a.level)
    n, m, skipped = build(cfg, list_cells(a.dataset, a.data_dir, a.rpt_dir, keep), out, a.workers)
    print(f"{out}: {n} cells, {m} windows", "skipped " + ", ".join(skipped) if skipped else "")
    if not tag:
        with open(os.path.join(ROOT, "splits", "cache_checksums.json"), encoding="utf-8") as fh:
            want = json.load(fh).get(a.dataset)
        got = checksum(out)
        print("matches the cache of the paper" if got == want else f"DIFFERS from the cache of the paper ({got})")


if __name__ == "__main__":
    main()
