"""Place the weekly C/5 tests of ISU-ILCC on the cycle indices of the processed BatteryLife cells.

Usage: python scripts/prepare_isu_rpt.py --rpt-json <RPT_json dir or .zip> --cycling-json <Cycling_json dir or .zip> --pkl-dir <ISU_ILCC> --out rpt

The original release gives each test's time and C/5 discharge capacity; BatteryLife gives the regular cycles. For each
cell, the cycling JSON capacities are matched in order to the BatteryLife capacities (difflib on values rounded to
0.1 mAh), so dropped or merged cycles in either source are found instead of assumed away. Output: rpt/<cell>.npz with
cap_dis_c5, and rpt_first_bl and rpt_last_bl, the BatteryLife cycles between one test and the next.
"""
import argparse
import difflib
import json
import os
import pickle
import zipfile
from multiprocessing import Pool

import numpy as np


def _time(s):
    if isinstance(s, str) and s not in ("", "[]"):
        return np.datetime64(s, "s").astype("int64").astype(float)
    return np.nan


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def _load(path):
    if "::" in path:                                # "<zip>::<member>": read straight from the downloaded archive
        archive, member = path.split("::", 1)
        with zipfile.ZipFile(archive) as z, z.open(member) as fh:
            return json.loads(json.load(fh))
    with open(path, "r") as fh:
        return json.loads(json.load(fh))


def _index(root):
    out = {}
    if root.endswith(".zip"):
        with zipfile.ZipFile(root) as z:
            for n in z.namelist():
                if n.endswith(".json"):
                    out[os.path.basename(n)[:-5]] = root + "::" + n
        return out
    for base, _, names in os.walk(root):
        for n in names:
            if n.endswith(".json"):
                out[n[:-5]] = os.path.join(base, n)
    return out


def align_cell(job):
    cell, rpt_path, cyc_path, pkl_path, out = job
    d = _load(rpt_path)
    stop = np.array([_time(s) for s in d["start_stop_time"]["stop"]])
    cap = np.array([_num(x) for x in d["capacity_discharge_C_5"]], float)
    c = _load(cyc_path)
    cap_j = np.array([_num(x) for x in c["capacity_discharge"]], float)
    t_j = np.array([_time(s) for s in c["time_series_discharge"]["start"]], float)
    with open(pkl_path, "rb") as fh:
        o = pickle.load(fh)
    cap_b = np.array([np.nanmax(x["discharge_capacity_in_Ah"]) if len(x["discharge_capacity_in_Ah"]) else np.nan for x in o["cycle_data"]], float)
    valid = np.where(np.isfinite(stop))[0]
    before = np.full(len(t_j), -1)                   # the test preceding each cycle of the JSON file
    for i, t in enumerate(t_j):
        if np.isfinite(t):
            k = valid[stop[valid] <= t]
            before[i] = k[-1] if len(k) else -1
    a = [round(x * 1e4) if np.isfinite(x) else -1 for x in cap_j]
    b = [round(x * 1e4) if np.isfinite(x) else -1 for x in cap_b]
    to_json = np.full(len(b), -1)
    for blk in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        to_json[blk.b:blk.b + blk.size] = np.arange(blk.a, blk.a + blk.size)
    matched = to_json >= 0
    rpt = np.full(len(b), -1)
    rpt[matched] = before[to_json[matched]]
    idx = np.where(matched)[0]
    for i in np.where(~matched)[0]:                  # unmatched cycles inherit a test only if both neighbours agree
        lo, hi = idx[idx < i], idx[idx > i]
        if len(lo) and len(hi) and rpt[lo[-1]] == rpt[hi[0]]:
            rpt[i] = rpt[lo[-1]]
    first, last = np.full(len(stop), -1), np.full(len(stop), -1)
    for k in range(len(stop)):
        w = np.where(rpt == k)[0]
        if len(w):
            first[k], last[k] = w[0], w[-1]
    np.savez_compressed(os.path.join(out, cell + ".npz"), cap_dis_c5=cap, rpt_first_bl=first, rpt_last_bl=last)
    return cell, float(matched.mean())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rpt-json", required=True)
    p.add_argument("--cycling-json", required=True)
    p.add_argument("--pkl-dir", required=True)
    p.add_argument("--out", default="rpt")
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rpt, cyc = _index(a.rpt_json), _index(a.cycling_json)
    jobs = []
    for f in sorted(os.listdir(a.pkl_dir)):
        if f.endswith(".pkl"):
            cell = f[len("ISU-ILCC_"):-4]
            if cell in rpt and cell in cyc:
                jobs.append((cell, rpt[cell], cyc[cell], os.path.join(a.pkl_dir, f), a.out))
            else:
                print("no original JSON for", cell)
    with Pool(a.workers) as pool:
        res = pool.map(align_cell, jobs)
    m = np.array([x for _, x in res])
    print(f"{len(res)} cells; share of cycles matched: median {np.median(m):.3f}, minimum {m.min():.3f}")


if __name__ == "__main__":
    main()
