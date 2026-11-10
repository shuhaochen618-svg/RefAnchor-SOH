"""From processed BatteryLife cells to charging windows: labels, sampling, anchor fit and model tensors.

Labels: Sandia uses 0.5C capacity tests (runs of densely sampled cycles), ISU-ILCC weekly C/5 RPTs placed by the
cycle alignment of scripts/prepare_isu_rpt.py, both interpolated between tests; Tongji has no tests and uses the cycle's
discharge capacity over the median of its first three usable cycles.
"""
import os
import pickle
from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np

from . import anchor as A

CURVE_POINTS = 32          # points per window curve
MAX_CYCLES = 150           # evenly spaced cycles per cell
WINDOWS_PER_WIDTH = 2      # windows per width per cycle
MIN_POINTS = 4             # records a window must contain
SOH_FLOOR = 0.7            # cycles after the label first falls below this value are dropped
WINDOW_SEED = 20261030
VOLTAGE_GRID = np.arange(2.0, 4.3, 0.0025)      # common grid of the chemistry-mean reference ablation
CHEMISTRY = {"LFP": 0, "NCA": 1, "NMC": 2}
WIDTHS = {"snl": (0.1, 0.2, 0.3, 0.5), "isu": (0.1, 0.2, 0.3), "tongji": (0.1, 0.2, 0.3, 0.5)}


@dataclass(frozen=True)
class Config:
    """How to build one cache. The defaults give the method of the paper.

    anchor:  full | nodelta (delta fixed at 0) | meanref (chemistry-mean reference of the other cells)
    perturb: none | down | vnoise | qdrift | reflate, with its level (test-time measurement errors of the robustness study)
    """
    kind: str
    anchor: str = "full"
    soh_floor: float = SOH_FLOOR
    perturb: str = "none"
    level: float = 0.0


def _perturbed(c, name, j, cfg):
    """A cycle's curve under a measurement error. Window positions still come from the clean curve."""
    v, q = c["v"], c["q"]
    if cfg.perturb == "down":
        keep, last = [], -9.0
        for i, x in enumerate(v):
            if x - last >= 0.1 / cfg.level:
                keep.append(i)
                last = x
        return v[keep], q[keep]
    if cfg.perturb == "vnoise":
        rng = np.random.default_rng(sum(map(ord, name)) + int(j))
        noisy = v + rng.normal(0, cfg.level, len(v))
        order = np.argsort(noisy)
        return noisy[order], q[order]
    if cfg.perturb == "qdrift":
        return v, q[0] + (q - q[0]) * (1 + cfg.level)
    return v, q


def _interp_labels(pos, cap, regular):
    """pos, cap: test positions (cycle index) and capacities; regular: cycle indices. Returns {cycle: SOH}."""
    labels = {}
    for k in range(len(pos) - 1):
        if not (np.isfinite(cap[k]) and np.isfinite(cap[k + 1])):
            continue
        for j in regular:
            if pos[k] < j < pos[k + 1]:
                f = (j - pos[k]) / (pos[k + 1] - pos[k])
                labels[j] = ((1 - f) * cap[k] + f * cap[k + 1]) / cap[0]
    return labels


def _windows(name, cycles, labels, q_nom, chem, temp, min_span, cfg, mean_ref=None, ref_only=False):
    """All windows of one cell, or None when too few cycles remain."""
    js = sorted(j for j in labels if j in cycles and len(cycles[j]["v"]) >= 5 and cycles[j]["v"][-1] - cycles[j]["v"][0] >= min_span)
    below = [j for j in js if labels[j] < cfg.soh_floor]
    if below:
        js = [j for j in js if j < below[0]]
    first = 9 if cfg.perturb == "reflate" else 0
    if len(js) < first + A.N_REF + 10:
        return None
    v_ref, q_ref = A.reference_curve([cycles[j] for j in js[first:first + A.N_REF]])
    if ref_only:
        q = np.interp(VOLTAGE_GRID, v_ref, q_ref, left=np.nan, right=np.nan) / q_nom
        return name, chem, np.gradient(q, VOLTAGE_GRID)
    # windows are always placed with the cell's own reference range, so every variant sees the same windows
    fit_v, fit_q = v_ref, q_ref
    if mean_ref is not None:
        ok = np.isfinite(mean_ref)
        fit_v = VOLTAGE_GRID[ok]
        fit_q = np.concatenate([[0], np.cumsum((mean_ref[ok][1:] + mean_ref[ok][:-1]) / 2 * np.diff(fit_v))]) * q_nom
    deltas = np.array([0.0]) if cfg.anchor == "nodelta" else A.DELTAS
    rest = js[first + A.N_REF:]
    chosen = [rest[i] for i in np.unique(np.linspace(0, len(rest) - 1, min(MAX_CYCLES, len(rest))).astype(int))]
    rng = np.random.default_rng(WINDOW_SEED + sum(map(ord, name)))
    one_hot = np.zeros(3)
    one_hot[CHEMISTRY[chem]] = 1
    seq, sca, meta = [], [], []
    for j in chosen:
        c = cycles[j]
        pv, pq = _perturbed(c, name, j, cfg)
        lo, hi = max(c["v"][0], v_ref[0]), min(c["v"][-1], v_ref[-1])
        for w in WIDTHS[cfg.kind]:
            if hi - lo < w:
                continue
            for _ in range(WINDOWS_PER_WIDTH):
                v1 = rng.uniform(lo, hi - w)
                m = (pv >= v1) & (pv <= v1 + w)
                if m.sum() < MIN_POINTS:
                    continue
                v, q = pv[m], pq[m]
                dq = q - q[0]
                e, s, d = A.anchor_fit(v, dq, fit_v, fit_q, deltas)
                if not np.isfinite(s):
                    continue
                g = np.linspace(v[0], v[-1], CURVE_POINTS)
                seg = np.interp(g, v, dq) / q_nom
                r0 = np.interp(g, fit_v, fit_q)
                r0 = (r0 - r0[0]) / q_nom
                rs = np.interp(g - d, fit_v, fit_q)
                rs = s * (rs - rs[0]) / q_nom
                dv = g[1] - g[0]
                # channels: window charge, reference, aligned reference, window dQ/dV, reference dQ/dV, voltage
                seq.append(np.stack([seg, r0, rs, np.gradient(seg, dv) * 0.1, np.gradient(r0, dv) * 0.1, g - 3.5]).astype(np.float32))
                t = c["t"] if np.isfinite(c["t"]) else temp
                # scalars: anchor, delta, fit error, width, start, points, C-rate, temperature, chemistry
                sca.append(np.r_[A.anchor(s), d * 10, np.sqrt(e) / q_nom * 100, v[-1] - v[0], v[0] - 3.5, m.sum() / 30, c["i"] / q_nom,
                                 (t - 25) / 10, one_hot].astype(np.float32))
                meta.append((j, w, labels[j], s, v[0]))
    if not meta:
        return None
    return name, np.array(seq), np.array(sca), np.array(meta)


def _snl_cell(job):
    path, cfg, mean_ref, ref_only = job
    name = os.path.basename(path)[:-4]
    with open(path, "rb") as fh:
        o = pickle.load(fh)
    cyc = o["cycle_data"]
    n = np.array([len(c["voltage_in_V"]) for c in cyc])
    qd = np.array([np.nanmax(c["discharge_capacity_in_Ah"]) if len(c["discharge_capacity_in_Ah"]) else np.nan for c in cyc])
    dense = n > 2.5 * np.median(n)             # capacity tests are logged far more densely than regular cycles
    idx = np.where(dense)[0]
    runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1) if len(idx) else []
    pos = np.array([r[0] for r in runs], float)
    cap = np.array([np.nanmax(qd[r]) for r in runs])
    if len(pos) < 2:
        return None
    cycles, last = {}, None
    for j in np.where(~dense)[0]:
        key = (n[j], float(np.nansum(cyc[j]["voltage_in_V"])), qd[j])
        if key == last:                        # exact duplicate of the previous cycle
            continue
        last = key
        c = A.cc_charge(cyc[j])
        if c is not None:
            cycles[j] = c
    labels = _interp_labels(pos, cap, list(cycles))
    parts = name.split("_")
    return _windows(name, cycles, labels, o["nominal_capacity_in_Ah"], parts[2], float(parts[3][:-1]), 0.3, cfg, mean_ref, ref_only)


def _isu_cell(job):
    (path, rpt_dir), cfg, mean_ref, ref_only = job
    name = os.path.basename(path)[:-4]
    cid = name[len("ISU-ILCC_"):]
    with open(path, "rb") as fh:
        o = pickle.load(fh)
    r = np.load(os.path.join(rpt_dir, cid + ".npz"))
    cap, first, last = r["cap_dis_c5"], r["rpt_first_bl"], r["rpt_last_bl"]
    if not np.isfinite(cap[0]):
        return None
    labels = {}
    for k in range(len(cap) - 1):              # RPT k sits just before cycle first[k]; week k is first[k]..last[k]
        if first[k] < 0 or not (np.isfinite(cap[k]) and np.isfinite(cap[k + 1])):
            continue
        n = last[k] - first[k] + 1
        for j in range(first[k], last[k] + 1):
            f = (j - first[k] + 0.5) / n
            labels[j] = ((1 - f) * cap[k] + f * cap[k + 1]) / cap[0]
    cycles = {}
    for j in labels:
        c = A.cc_charge(o["cycle_data"][j])
        if c is not None:
            cycles[j] = c
    return _windows(name, cycles, labels, o["nominal_capacity_in_Ah"], "NMC", 25.0, 0.15, cfg, mean_ref, ref_only)


def _tongji_cell(job):
    path, cfg, mean_ref, ref_only = job
    name = os.path.basename(path)[:-4]
    with open(path, "rb") as fh:
        o = pickle.load(fh)
    q_nom = o["nominal_capacity_in_Ah"]
    cycles, qd, last = {}, {}, None
    for j, c in enumerate(o["cycle_data"]):
        if not len(c["voltage_in_V"]) or not len(c["discharge_capacity_in_Ah"]):
            continue
        q = float(np.nanmax(c["discharge_capacity_in_Ah"]))
        key = (len(c["voltage_in_V"]), float(np.nansum(c["voltage_in_V"])), q)
        if key == last or not np.isfinite(q) or q < 0.3 * q_nom:
            continue
        last = key
        cc = A.cc_charge(c)
        if cc is not None:
            cycles[j], qd[j] = cc, q
    js = sorted(j for j in cycles if len(cycles[j]["v"]) >= 5 and cycles[j]["v"][-1] - cycles[j]["v"][0] >= 0.3)
    if len(js) < A.N_REF + 10:
        return None
    q0 = np.median([qd[j] for j in js[:A.N_REF]])
    labels = {j: qd[j] / q0 for j in js}
    chem = "NCA" if "NCA" in str(o.get("cathode_material")) else "NMC"
    return _windows(name, cycles, labels, q_nom, chem, float(name.split("_")[1][2:4]), 0.3, cfg, mean_ref, ref_only)


CELL_FN = {"snl": _snl_cell, "isu": _isu_cell, "tongji": _tongji_cell}


def list_cells(kind, data_dir, rpt_dir=None, cell_list=None):
    """Input files of a dataset. Sandia uses its full 0-100 % SOC cells, as in the paper."""
    names = sorted(f for f in os.listdir(data_dir) if f.endswith(".pkl") and (kind != "snl" or "_0-100_" in f))
    if cell_list is not None:
        names = [n for n in names if n[:-4] in cell_list]
    paths = [os.path.join(data_dir, n) for n in names]
    return [(p, rpt_dir) for p in paths] if kind == "isu" else paths


def _name(cell):
    return os.path.basename(cell[0] if isinstance(cell, tuple) else cell)[:-4]


def build(cfg, cells, out, workers=8):
    """Build one cache: seq (windows x 6 x 32), sca (windows x 11), cells, cyc, w, y, s, v1, cyc_id."""
    fn = CELL_FN[cfg.kind]
    with Pool(workers) as pool:
        mean_ref = {}
        if cfg.anchor == "meanref":
            refs = [r for r in pool.map(fn, [(c, cfg, None, True) for c in cells]) if r is not None]
            for name, chem, _ in refs:
                with np.errstate(all="ignore"):
                    mean_ref[name] = np.nanmean([x for n, ch, x in refs if ch == chem and n != name], axis=0)
            cells = [c for c in cells if _name(c) in mean_ref]
        res = pool.map(fn, [(c, cfg, mean_ref.get(_name(c)), False) for c in cells])
    skipped = [_name(c) for c, r in zip(cells, res) if r is None]
    res = [r for r in res if r is not None]
    names = np.concatenate([[n] * len(m) for n, _, _, m in res])
    meta = np.concatenate([m for *_, m in res])
    key = np.array([f"{c}#{int(k)}" for c, k in zip(names, meta[:, 0])])
    _, cyc_id = np.unique(key, return_inverse=True)
    np.savez_compressed(out, seq=np.concatenate([x[1] for x in res]), sca=np.concatenate([x[2] for x in res]), cells=names, cyc=meta[:, 0], w=meta[:, 1],
                        y=meta[:, 2], s=meta[:, 3], v1=meta[:, 4], cyc_id=cyc_id)
    return len(res), len(names), skipped
