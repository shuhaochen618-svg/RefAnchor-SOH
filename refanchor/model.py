"""RefAnchor-SOH and its tree-based controls: inputs, grouped cross-validation and held-out conditions."""
import os
import pickle
import re

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .anchor import anchor

N_TREES, MIN_LEAF = 300, 3
TREE_METHODS = ("b0", "b1", "b5", "b1h", "v2", "v3", "v4", "m1h")      # m1h is RefAnchor-SOH
RESIDUAL = ("m1h", "v2")                                               # these learn SOH minus the anchor
CLAIM_WIDTHS = (0.2, 0.3, 0.5)


def group(cell):
    """Condition group of a cell. Folds never split a group."""
    m = re.match(r"ISU-ILCC_(G\d+)C\d+", cell)
    if m:
        return m.group(1)
    return cell.split("--")[0] if cell.startswith("Tongji") else cell.rsplit("_", 1)[0]


def _tongji_protocol(cell):
    m = re.match(r"Tongji\d_CY\d+-(\d+)_(\d+)--", cell)
    a = m.group(1)
    return [float("0." + a[1:]) if a.startswith("0") else float(a), float(m.group(2)), 0.0, 1.0]


def _rate(steps):
    r = [x["rate_in_C"] for x in steps or [] if x.get("rate_in_C") is not None]
    return float(np.mean(r)) if r else -1.0


def protocols(kind, data_dir, cells):
    """{cell: [charge rate, discharge rate, SOC low, SOC high]}."""
    names = sorted(set(cells))
    if kind == "tongji":
        return {c: _tongji_protocol(c) for c in names}
    out = {}
    for c in names:
        with open(os.path.join(data_dir, c + ".pkl"), "rb") as fh:
            o = pickle.load(fh)
        lo, hi = (o.get("SOC_interval") or [0, 1])[:2]
        out[c] = [_rate(o.get("charge_protocol")), _rate(o.get("discharge_protocol")), float(lo), float(hi)]
    return out


def inputs(D, pr):
    """Input matrix per method. Channels and scalars are described in refanchor/data.py."""
    y, cells, seq, sca = D["y"], D["cells"], D["seq"], D["sca"]
    use = np.c_[D["cyc"], np.array([pr[c] for c in cells])]
    win = np.c_[seq[:, [0, 3]].reshape(len(y), -1), sca[:, 3:]]
    anc = np.c_[seq[:, [0, 2, 3]].reshape(len(y), -1), sca]
    full = np.c_[anc, use]
    return {"b1": win, "b5": np.c_[use, sca[:, 7], sca[:, 8:]], "b1h": np.c_[win, use], "v2": anc, "v3": anc, "v4": full, "m1h": full}


def ic_features(D):
    """Incremental-capacity features of the Gaussian-process control B2."""
    seq, sca = D["seq"], D["sca"]
    ic, g = seq[:, 3], seq[:, 5] + 3.5
    k = np.argmax(ic, 1)
    return np.c_[ic.max(1), g[np.arange(len(g)), k], ic.mean(1), ic.std(1), seq[:, 0, -1], sca[:, 3], sca[:, 4], sca[:, 6], sca[:, 7], sca[:, 8:]]


def folds(cells):
    return list(GroupKFold(5).split(cells, cells, np.array([group(c) for c in cells])))


def fit_predict(x, y, base, train, test, seed, jobs=-1):
    """Fit one forest. With base set, learn y - base and add it back (the residual form of Eq. 7)."""
    et = ExtraTreesRegressor(n_estimators=N_TREES, min_samples_leaf=MIN_LEAF, n_jobs=jobs, random_state=seed)
    if base is None:
        return et.fit(x[train], y[train]).predict(x[test])
    return base[test] + et.fit(x[train], y[train] - base[train]).predict(x[test])


def cross_validate(D, pr, method, seed, jobs=-1):
    """Out-of-fold predictions over condition groups."""
    y, cells = D["y"], D["cells"]
    a = anchor(D["s"])
    if method == "b0":
        return a
    x = inputs(D, pr)[method]
    pred = np.full_like(y, np.nan)
    for train, test in folds(cells):
        pred[test] = fit_predict(x, y, a if method in RESIDUAL else None, train, test, seed, jobs)
    return pred


def gpr_cross_validate(D, seed, max_train=3000):
    """Control B2. A Gaussian process cannot take all windows, so each fold keeps a random subset."""
    y = D["y"]
    x = ic_features(D)
    pred = np.full_like(y, np.nan)
    for train, test in folds(D["cells"]):
        sub = np.random.default_rng(seed).choice(train, min(max_train, len(train)), replace=False)
        sc = StandardScaler().fit(x[sub])
        gp = GaussianProcessRegressor(ConstantKernel() * RBF(np.ones(x.shape[1])) + WhiteKernel(), normalize_y=True, random_state=seed)
        pred[test] = gp.fit(sc.transform(x[sub]), y[sub]).predict(sc.transform(x[test]))
    return pred


def held_out_targets(kind, cells, pr):
    """The conditions kept out of training in the frozen test: {target: boolean mask of its windows}."""
    if kind == "snl":
        out = {f"T{t}": np.array([f"_{t}C_" in c for c in cells]) for t in (15, 25, 35)}
        out.update({ch: np.array([c.split("_")[2] == ch for c in cells]) for ch in ("LFP", "NCA", "NMC")})
        return out
    if kind == "isu":
        gs = np.array([group(c) for c in cells])
        rate = {g: np.mean([pr[c][0] for c in set(cells[gs == g])]) for g in set(gs)}
        top = sorted(rate, key=rate.get)[-max(1, len(rate) // 5):]          # fastest-charging fifth of the groups
        return {"ISUfast": np.isin(gs, top)}
    raise ValueError("held-out conditions are defined for snl and isu")
