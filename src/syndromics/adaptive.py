"""Adaptive decoding and drift tracking.

Two things learned on real ibm_kingston data (job dbukcjvf8rhs73bmf9l0, Nov 2026):

1. Per-shot adaptation (leakage-aware decoding). In about 2% of real shots one measure qubit
   produces near-random outcomes for much of the run (62x more often than a Pauli noise model
   predicts), the signature of leakage or a failed reset. Those shots cause about half of all
   logical failures. The decoder can see this from the shot's own syndrome: when one measure
   qubit fires >= k detection events, its measurement errors are boosted for that shot only, so
   the decoder stops trusting it. Held-out result on the real data: 0.198% -> 0.103%. The trick
   works for any decoder model (correlation re-weighting gains the same).

2. Drift tracking across windows. Each window of shots gets a fresh fit from the calibration;
   a component is flagged when it is significantly above both its calibrated value and the
   inverse-variance mean of earlier windows, in at least `confirm` windows. Simulated drift
   (gate x4, measurement x3) is caught within one window with no false alarms; idle-error drift
   on a data qubit is not observable in a repetition code.

A naive "memory" chain (each fit starting from the previous fitted model) was tested and
rejected: scale factors compound, mechanisms vanish and decoding becomes unstable.
"""
from __future__ import annotations

import numpy as np

from . import decodecheck as dc
from .model import fit_noise, rebuild


def _measure_detectors(circ):
    """Index of the measure qubit (line position) for each detector, from detector coordinates."""
    co = circ.get_detector_coordinates()
    pos = np.array([int(co[d][0]) for d in range(circ.num_detectors)])
    return pos, sorted(set(pos.tolist()))


class PerShotDecoder:
    """Per shot: if one measure qubit fires >= k detection events, decode with the matching that
    distrusts that qubit's measurements; otherwise with the base matching."""

    def __init__(self, base, per, pos, anc, k):
        self.base, self.per, self.pos, self.anc, self.k = base, per, pos, anc, k

    def flags(self, det):
        cnt = np.stack([det[:, self.pos == a].sum(1) for a in self.anc], 1)
        return cnt.max(1) >= self.k, cnt.argmax(1)

    def decode_batch(self, det):
        pred = self.base.decode_batch(det).copy()
        flag, which = self.flags(det)
        for j, m in enumerate(self.per):
            sel = flag & (which == j)
            if sel.any():
                pred[sel] = m.decode_batch(det[sel])
        return pred


def _syndromics_builder(circ, det, sparsity):
    fit = fit_noise(circ, det.astype(np.uint8), sparsity=sparsity, crosstalk=False)
    sc = {f.component: max(0.0, f.scale) for f in fit.fits if f.component[0] != "xtalk"}
    pos, anc = _measure_detectors(circ)

    def build(boost, k):
        per = []
        for a in anc:
            s2 = dict(sc); s2[("spam", a)] = s2.get(("spam", a), 1.0) * boost
            per.append(dc._matching(rebuild(fit.circuit, lambda c, s2=s2: s2.get(c, 1.0))))
        return PerShotDecoder(dc._matching(fit.calibrated_circuit(crosstalk=False)), per, pos, anc, k)
    return build


def _reweighting_builder(circ, det, sparsity):
    pos, anc = _measure_detectors(circ)

    def build(boost, k):
        base = dc.reweighting_matching(circ, det)
        per = [dc.reweighting_matching(circ, det, boost_pos=a, boost=boost, pos=pos) for a in anc]
        return PerShotDecoder(base, per, pos, anc, k)
    return build


def fit_leakage_aware(circ, det_train, obs_train, *, method="syndromics", sparsity=0.5,
                      boosts=(5, 10, 20, 40), ks=(4, 5, 6)):
    """Fit on det_train and choose (boost, k) by 2-fold CV inside the training data only."""
    maker = _syndromics_builder if method == "syndromics" else _reweighting_builder
    n = len(det_train); h = n // 2
    halves = (np.arange(h), np.arange(h, n))
    builders = [maker(circ, det_train[i], sparsity) for i in halves]
    best = None
    for b in boosts:
        for k in ks:
            e = 0
            for bld, te in ((builders[0], halves[1]), (builders[1], halves[0])):
                e += int((bld(b, k).decode_batch(det_train[te])[:, 0] != obs_train[te]).sum())
            if best is None or e < best[0]:
                best = (e, b, k)
    dec = maker(circ, det_train, sparsity)(best[1], best[2])
    dec.boost, dec.k = best[1], best[2]
    return dec


def burst_stats(circ, det, k=6):
    """How often one measure qubit fires >= k detection events in a shot: real data vs the fitted model."""
    pos, anc = _measure_detectors(circ)
    cnt = np.stack([det[:, pos == a].sum(1) for a in anc], 1)
    fit = fit_noise(circ, det.astype(np.uint8), sparsity=0.5, crosstalk=False)
    sim = fit.calibrated_circuit(crosstalk=False).compile_detector_sampler(seed=7).sample(len(det))
    cs = np.stack([sim[:, pos == a].sum(1) for a in anc], 1)
    return {"k": k, "real": float((cnt >= k).any(1).mean()), "model": float((cs >= k).any(1).mean()),
            "per_qubit_real": [float((cnt[:, j] >= k).mean()) for j in range(len(anc))], "anc": anc}


def track_drift(start_circ, windows, line, *, z=4.0, min_ratio=1.5, confirm=2):
    """Fresh fit per window; returns (per-window scales, confirmed alerts)."""
    hist, raw, table = {}, [], []
    for k, w in enumerate(windows):
        f = fit_noise(start_circ, w.astype(np.uint8), sparsity=0.0, crosstalk=False)
        row = {}
        for x in f.fits:
            c = x.component
            if c[0] == "xtalk":
                continue
            e = max(x.err, 0.1, 0.1 * x.scale)
            row[c] = x.scale
            if k >= 2 and c in hist:
                v = np.array(hist[c]); wt = 1 / v[:, 1] ** 2
                mu = (wt * v[:, 0]).sum() / wt.sum(); emu = wt.sum() ** -0.5
                zz = (x.scale - mu) / np.hypot(e, emu)
                if zz > z and x.scale > min_ratio * max(mu, 1.0):
                    raw.append((k, c, mu, x.scale, zz))
            hist.setdefault(c, []).append((x.scale, e))
        table.append(row)
    alerts, seen = [], {}
    for k, c, mu, s, zz in raw:
        seen.setdefault(c, []).append(k)
        if len(seen[c]) == confirm:
            alerts.append({"window": k + 1, "first_seen": seen[c][0] + 1, "component": _name(c, line),
                           "before": float(mu), "now": float(s), "z": float(zz)})
    return table, alerts


def _name(c, line):
    if c[0] == "spam":
        return f"q{line[c[1]]} measurement"
    if c[0] == "gate":
        return f"q{line[c[1]]}-q{line[c[2]]} two-qubit gate"
    return f"q{line[c[1]]} idle/single-qubit"


def compare_runs(prev_rep, prev_cfg, new_rep, new_cfg, models_fn, *, sparsity=0.5):
    """Prospective test: freeze everything learned on run 1 (noise model + adaptive settings) and decode run 2.
    Also reports how components changed between the runs. Both runs must use the same qubits and circuit."""
    from . import ibm
    for k in ("line", "n_data", "rounds"):
        if prev_cfg[k] != new_cfg[k]:
            raise ValueError(f"runs differ in {k}: {prev_cfg[k]} vs {new_cfg[k]} (use --line to run on the same qubits)")
    nd, R, line = new_cfg["n_data"], new_cfg["rounds"], new_cfg["line"]
    _, cal1 = models_fn(prev_cfg)
    _, cal2 = models_fn(new_cfg)
    c1, c2 = ibm.rep_code_stim(nd, R, *cal1), ibm.rep_code_stim(nd, R, *cal2)
    D1, O1 = dc._detectors(c1, prev_rep)
    D2, O2 = dc._detectors(c2, new_rep)

    def ler(m):
        e = int((m.decode_batch(D2)[:, 0] != O2).sum())
        return {"errors": e, "rate": e / len(D2)}
    fit1 = fit_noise(c1, D1.astype(np.uint8), sparsity=sparsity, crosstalk=False)
    la1 = fit_leakage_aware(c1, D1, O1, sparsity=sparsity)
    res = {"shots_run2": len(D2),
           "run2_calibration": ler(dc._matching(c2)),
           "run1_calibration": ler(dc._matching(c1)),
           "frozen_syndromics": ler(dc._matching(fit1.calibrated_circuit(crosstalk=False))),
           "frozen_syndromics_la": ler(la1),
           "frozen_settings": {"boost": la1.boost, "k": la1.k}}
    # component changes between runs: same regularised fit as the main report, from the same starting model.
    # (An unregularised fit cannot separate a measurement error from its neighbouring gate, so it can report
    # a "change" that is only the fit moving error between the two; see CHANGELOG 0.5.2.)
    f1 = fit1
    f2 = fit_noise(c1, D2.astype(np.uint8), sparsity=sparsity, crosstalk=False)
    s2 = {x.component: x for x in f2.fits}
    changes = []
    for x in f1.fits:
        c = x.component
        if c[0] == "xtalk" or c not in s2:
            continue
        y = s2[c]
        e = np.hypot(max(x.err, 0.1, 0.1 * x.scale), max(y.err, 0.1, 0.1 * y.scale))
        z = (y.scale - x.scale) / e
        if abs(z) > 4 and max(x.scale, y.scale) > 1.5 and (y.scale > 1.5 * max(x.scale, 0.2) or x.scale > 1.5 * max(y.scale, 0.2)):
            changes.append({"component": _name(c, line), "run1": float(x.scale), "run2": float(y.scale), "z": float(z)})
    res["component_changes"] = sorted(changes, key=lambda r: -abs(r["z"]))
    res["bursts_run2"] = burst_stats(c2, D2)
    return res
