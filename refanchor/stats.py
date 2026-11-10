"""Cell-level statistics: every test compares methods on cells, because the windows of a cell are not independent."""
import numpy as np
from scipy.stats import wilcoxon

from .anchor import S_MAX, S_MIN


def clip(pred):
    return np.clip(pred, S_MIN, S_MAX)


def mae(pred, y):
    """Mean absolute error in percentage points."""
    return 100 * float(np.abs(pred - y).mean())


def cell_mae(pred, y, cells, unique):
    e = np.abs(pred - y)
    return np.array([e[cells == c].mean() for c in unique])


def bootstrap_reduction(a, b, n=10000):
    """95 % percentile bootstrap CI over cells of the relative reduction of a against b, in %."""
    rng = np.random.default_rng(0)
    r = [1 - a[i].mean() / b[i].mean() for i in (rng.integers(0, len(a), len(a)) for _ in range(n))]
    return 100 * np.percentile(r, [2.5, 97.5])


def compare(pred, control, y, cells):
    """pred, control: one array per window, averaged over seeds. Returns the paper's comparison of two methods."""
    u = np.unique(cells)
    a, b = cell_mae(pred, y, cells, u), cell_mae(control, y, cells, u)
    lo, hi = bootstrap_reduction(a, b)
    return dict(cells=len(u), better=int((a < b).sum()), ci_low=float(lo), ci_high=float(hi), p=float(wilcoxon(a, b, alternative="less").pvalue))
