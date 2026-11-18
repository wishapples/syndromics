"""Held-out decoding comparison on real repetition-code data.

The question a hardware team cares about: does a decoder given the Syndromics noise model make fewer
logical errors than one given the device calibration? We answer it by cross-validation on the same shots:
fit on one half, decode the other half, for several random splits plus a time split (first half -> second half).

Four noise models are compared, all decoded with the same matching decoder (PyMatching):
  uniform        one median value per error type (no device knowledge)
  calibration    the device's published per-qubit / per-gate values
  reweighting    standard model-free correlation re-weighting from detection events (literature baseline)
  syndromics     Syndromics physical-noise fit, starting from the calibration

Crosstalk candidates are left out of the decoder model: on a repetition code a fitted two-data-qubit term
can add a logical-flipping shortcut to the matching graph (seen on real ibm_kingston data, Nov 2026).
"""
from __future__ import annotations

import numpy as np
import pymatching

from . import ibm
from .model import fit_noise

MODELS = ("uniform", "calibration", "reweighting", "syndromics", "reweighting_la", "syndromics_la")
LABELS = {"uniform": "Uniform (no device data)",
          "calibration": "Device calibration",
          "reweighting": "Standard correlation re-weighting",
          "syndromics": "Syndromics noise model",
          "reweighting_la": "Re-weighting + leakage-aware",
          "syndromics_la": "Syndromics + leakage-aware"}


def _detectors(circ, meas):
    d, o = circ.compile_m2d_converter().convert(measurements=meas.astype(bool), separate_observables=True)
    return d, o[:, 0]


def _matching_edges(circ):
    """Graph-like edges of the matching graph -> whether the edge flips the logical observable."""
    edges = {}
    for inst in circ.detector_error_model(decompose_errors=True).flattened():
        if inst.type != "error":
            continue
        parts, cur = [], []
        for t in inst.targets_copy():
            if t.is_separator():
                parts.append(cur); cur = []
            else:
                cur.append(t)
        parts.append(cur)
        for part in parts:
            ds = tuple(sorted(t.val for t in part if t.is_relative_detector_id()))
            if ds:
                edges.setdefault(ds, any(t.is_logical_observable_id() for t in part))
    return edges


def reweighting_matching(circ, det_train, boost_pos=None, boost=1.0, pos=None):
    """Model-free baseline: edge probabilities estimated from detection-event correlations (p_ij).
    boost_pos: multiply the time-like edges of that measure qubit by `boost` (leakage-aware decoding)."""
    edges = _matching_edges(circ)
    X = det_train.astype(float)
    p1 = X.mean(0)
    P11 = (X.T @ X) / len(X)

    def pij(i, j):
        a = 1 - 2 * p1[i] - 2 * p1[j] + 4 * P11[i, j]
        x = 1 - 4 * (P11[i, j] - p1[i] * p1[j]) / max(a, 1e-12)
        return float(min(max(0.5 - 0.5 * np.sqrt(max(x, 0.0)), 1e-6), 0.49))

    pe = {e: pij(*e) for e in edges if len(e) == 2}
    for e in edges:
        if len(e) == 1:
            i = e[0]
            prod = np.prod([1 - 2 * pe[f] for f in pe if i in f]) or 1e-12
            pe[e] = float(min(max(0.5 - 0.5 * (1 - 2 * p1[i]) / prod, 1e-6), 0.49))
    m = pymatching.Matching()
    for e, p in pe.items():
        if boost_pos is not None and len(e) == 2 and pos[e[0]] == pos[e[1]] == boost_pos:
            p = min(p * boost, 0.45)
        kw = dict(fault_ids={0} if edges[e] else set(), weight=float(np.log((1 - p) / p)),
                  error_probability=p, merge_strategy="independent")
        if len(e) == 2:
            m.add_edge(e[0], e[1], **kw)
        elif len(e) == 1:
            m.add_boundary_edge(e[0], **kw)
    return m


def _matching(circ):
    return pymatching.Matching.from_detector_error_model(circ.detector_error_model(decompose_errors=True))


def compare(meas, n_data, rounds, uniform, calibrated, *, splits=5, seed=100, sparsity=0.5, leakage_aware=True):
    """meas: (shots, n_meas) raw repetition-code measurements. uniform/calibrated: (readout, gate2, idle) dicts.
    Returns {"tests": [...], "mean": {model: ler}, "shots": n, "raw_flip_rate": x}."""
    uni_c = ibm.rep_code_stim(n_data, rounds, *uniform)
    cal_c = ibm.rep_code_stim(n_data, rounds, *calibrated)
    D, O = _detectors(uni_c, meas)            # detectors do not depend on the noise values
    n = len(D)
    h = n // 2
    tests = []
    rng = np.random.default_rng(seed)
    for s in range(splits):
        p = rng.permutation(n)
        tests.append(("random split %d" % (s + 1), p[:h], p[h:]))
    tests.append(("time split (first half -> second half)", np.arange(h), np.arange(h, n)))
    m_uni, m_cal = _matching(uni_c), _matching(cal_c)
    out = []
    for name, tr, te in tests:
        def ler(m):
            return float((m.decode_batch(D[te])[:, 0] != O[te]).mean())
        syn = fit_noise(cal_c, D[tr].astype(np.uint8), sparsity=sparsity, crosstalk=False).calibrated_circuit()
        row = {"test": name, "uniform": ler(m_uni), "calibration": ler(m_cal),
               "reweighting": ler(reweighting_matching(uni_c, D[tr])), "syndromics": ler(_matching(syn))}
        if leakage_aware:
            from .adaptive import fit_leakage_aware
            for key, meth in (("syndromics_la", "syndromics"), ("reweighting_la", "reweighting")):
                dec = fit_leakage_aware(cal_c, D[tr], O[tr], method=meth, sparsity=sparsity)
                row[key] = ler(dec)
        out.append(row)
    mean = {k: float(np.mean([t[k] for t in out])) for k in MODELS if k in out[0]}
    return {"tests": out, "mean": mean, "shots": n, "test_shots": n - h,
            "raw_flip_rate": float(O.mean()), "sparsity": sparsity}


def changed_components(meas, n_data, rounds, uniform, calibrated, line, *, sparsity=0.5, min_ratio=2.0, z=3.0):
    """Fit on all shots, starting from the calibration; list components clearly worse than calibrated."""
    cal_c = ibm.rep_code_stim(n_data, rounds, *calibrated)
    D, _ = _detectors(cal_c, meas)
    fit = fit_noise(cal_c, D.astype(np.uint8), sparsity=sparsity, crosstalk=False)
    ro, g2, idle = calibrated
    rows = []
    for f in fit.fits:
        c = f.component
        if c[0] == "xtalk" or f.scale < min_ratio or (f.err > 0 and (f.scale - 1) / f.err < z):
            continue
        if c[0] == "spam":
            what, base = f"q{line[c[1]]} measurement", ro[c[1]]
        elif c[0] == "gate":
            what, base = f"q{line[c[1]]}-q{line[c[2]]} two-qubit gate", g2[(c[1], c[2])]
        elif c[0] == "qubit":
            what, base = f"q{line[c[1]]} idle/single-qubit", idle[c[1]]
        else:
            continue
        rows.append({"component": what, "ratio": float(f.scale), "err": float(f.err),
                     "calibration": float(base), "measured": float(base * f.scale)})
    return sorted(rows, key=lambda r: -r["ratio"])


def format_comparison(res):
    m = res["mean"]
    lines = [f"Held-out decoding test: {len(res['tests'])} tests, each fits on {res['shots'] - res['test_shots']:,} shots "
             f"and decodes the other {res['test_shots']:,}.",
             f"  {'Noise model given to the decoder':38s} {'logical error':>13s}  {'vs calibration':>14s}"]
    for k in [k for k in MODELS if k in m]:
        ch = "baseline" if k == "calibration" else f"{(m[k] / m['calibration'] - 1) * 100:+.0f}%"
        lines.append(f"  {LABELS[k]:38s} {m[k] * 100:12.3f}%  {ch:>14s}")
    lines.append("  per test (calibration -> Syndromics): " +
                 ", ".join(f"{t['calibration'] * 100:.3f}->{t['syndromics'] * 100:.3f}" for t in res["tests"]))
    wins = sum(t["syndromics"] < t["calibration"] for t in res["tests"])
    lines.append(f"  Syndromics beat the calibration in {wins} of {len(res['tests'])} tests.")
    if "syndromics_la" in m:
        w2 = sum(t["syndromics_la"] < t["syndromics"] for t in res["tests"])
        lines.append(f"  Leakage-aware decoding improved on Syndromics in {w2} of {len(res['tests'])} tests "
                     "(per shot: distrust a measure qubit that fires >= k detection events; settings chosen on training data).")
    n_err = m["calibration"] * res["test_shots"]
    if n_err < 20:
        lines.append(f"  NOTE: only ~{n_err:.0f} logical errors per test with the calibration model: too few to compare models.")
        lines.append("  This is expected on the simulator (its noise is exactly the published calibration and is low).")
        lines.append("  On hardware, use more shots or --rounds 20 if you see this note.")
    return "\n".join(lines)
