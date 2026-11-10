"""The reference-curve anchor: extract constant-current charging, build a cell's reference, align a window to it."""
import numpy as np

N_REF = 3                                  # usable early cycles averaged into the reference; they stand for SOH = 1
N_REF_POINTS = 400                         # voltages of the stored reference curve
DELTAS = np.linspace(-0.2, 0.2, 201)       # voltage shifts searched, in V (2 mV steps)
S_MIN, S_MAX = 0.5, 1.2                    # physically plausible range of the anchor and of every model output


def cc_charge(c):
    """First contiguous constant-current charging run of one cycle, sorted by voltage, or None.

    c: dict with current_in_A, voltage_in_V, charge_capacity_in_Ah and optionally temperature_in_C.
    Returns dict(v, q, i, t): voltage, charge capacity, typical charging current, mean temperature (NaN if not logged).
    """
    current = np.asarray(c["current_in_A"], float)
    voltage = np.asarray(c["voltage_in_V"], float)
    charge = np.asarray(c["charge_capacity_in_Ah"], float)
    temp = np.asarray(c["temperature_in_C"], float) if c.get("temperature_in_C") is not None else np.full_like(voltage, np.nan)
    pos = current > 0.02
    if pos.sum() < 5:
        return None
    imax = np.median(current[pos][current[pos] >= np.percentile(current[pos], 50)])
    idx = np.where(pos & (current > 0.9 * imax) & (voltage < np.nanmax(voltage[pos]) - 0.005))[0]
    if len(idx) < 5:
        return None
    breaks = np.where(np.diff(idx) > 1)[0]
    if len(breaks):
        idx = idx[:breaks[0] + 1]
    v, q = voltage[idx], charge[idx]
    order = np.argsort(v, kind="stable")
    v, q = v[order], q[order]
    keep = np.concatenate([[True], np.diff(v) > 1e-4])
    return dict(v=v[keep], q=q[keep], i=float(imax), t=float(np.nanmean(temp[idx])) if np.isfinite(temp[idx]).any() else np.nan)


def reference_curve(cycles):
    """Mean of the cycles' charging curves, each starting at zero charge, on their common voltage range."""
    v_ref = np.linspace(max(x["v"][0] for x in cycles), min(x["v"][-1] for x in cycles), N_REF_POINTS)
    q_ref = np.mean([np.interp(v_ref, x["v"], x["q"] - x["q"][0]) for x in cycles], axis=0)
    return v_ref, q_ref


def anchor_fit(v, dq, v_ref, q_ref, deltas=DELTAS):
    """Least-squares alignment dq(v) ~ s * [q_ref(v - delta) - q_ref(v[0] - delta)].

    The reference is never extrapolated: a shift that moves the window outside its range is skipped.
    Returns (mean squared error, s, delta); NaNs if no shift is admissible.
    """
    best = (np.inf, np.nan, np.nan)
    for d in deltas:
        if v[0] - d < v_ref[0] or v[-1] - d > v_ref[-1]:
            continue
        r = np.interp(v - d, v_ref, q_ref)
        r = r - r[0]
        den = r @ r
        if den <= 0:
            continue
        s = (r @ dq) / den
        e = np.mean((dq - s * r) ** 2)
        if e < best[0]:
            best = (e, s, d)
    return best


def anchor(s):
    """The health anchor: the scale limited to the plausible range."""
    return np.clip(s, S_MIN, S_MAX)
