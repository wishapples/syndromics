"""
Chip diagnostics from syndrome data.

Given
  * the nominal (designed) Stim circuit of a surface-code memory experiment, and
  * detection events recorded from the chip,
estimate which physical components are worse than spec:
  * data qubits with elevated X- or Z-type error,
  * neighbouring data-qubit pairs with correlated (crosstalk) errors,
  * measurement qubits with elevated readout/reset error.

Method
  1. Estimate the probability of every error mechanism that flips a detector pair from
     the pairwise correlations of detection events, and subtract a baseline simulated
     from the nominal model.
  2. Compute each candidate fault's detector fingerprint by injecting that single fault
     into a noiseless copy of the circuit.
  3. Fit all candidates jointly (non-negative least squares) so that faults compete to
     explain the data, which suppresses false suspects around a real defect.
  4. Group candidates whose fingerprints are identical. Their product is a stabilizer, so
     no syndrome data can separate them; the group is reported as a whole.

Supported today: rotated surface-code memory circuits in the layout produced by
``stim.Circuit.generated("surface_code:rotated_memory_[x|z]", ...)``.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import stim

NOISE = {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR", "Y_ERROR",
         "PAULI_CHANNEL_1", "PAULI_CHANNEL_2", "CORRELATED_ERROR", "E", "ELSE_CORRELATED_ERROR"}


@dataclass
class Finding:
    """One group of syndrome-equivalent candidate faults."""
    kind: str                       # "data_qubit" | "crosstalk" | "measure_qubit"
    members: list                   # [(kind, qubits tuple, pauli), ...]
    extra_rate: float               # estimated extra error probability per round
    times_spec: float               # rough multiple of the nominal error rate
    flagged: bool

    @property
    def exact(self) -> bool:
        return len({m[1] for m in self.members}) == 1

    def describe(self) -> str:
        who = " or ".join(f"{k.replace('_', ' ')} {list(q)} ({pa})" for k, q, pa in self.members)
        return f"{self.times_spec:4.1f}x spec  {who}"


@dataclass
class Report:
    findings: list = field(default_factory=list)
    fit_rms: float = 0.0

    @property
    def flagged(self):
        return [f for f in self.findings if f.flagged]

    def summary(self) -> str:
        lines = [f"{len(self.flagged)} finding(s) above threshold:"]
        lines += ["  " + f.describe() for f in self.flagged] or ["  none"]
        return "\n".join(lines)


# ---------------------------------------------------------------- statistics
def pair_stats(det: np.ndarray, chunk: int = 200_000):
    """Marginals <x_i> and the pairwise mechanism probability p_ij for all detector pairs."""
    n, D = det.shape
    s1, s2 = np.zeros(D), np.zeros((D, D))
    for i in range(0, n, chunk):
        x = det[i:i + chunk].astype(np.float32)
        s1 += x.sum(0)
        s2 += x.T @ x
    m1, m2 = s1 / n, s2 / n
    num = m2 - np.outer(m1, m1)
    den = 1 - 2 * m1[:, None] - 2 * m1[None, :] + 4 * m2
    pij = 0.5 - 0.5 * np.sqrt(np.clip(1 - 4 * num / den, 0, None))
    np.fill_diagonal(pij, 0)
    return m1, pij


# ---------------------------------------------------------------- layout
def qubit_roles(circ: stim.Circuit):
    coords = circ.get_final_qubit_coordinates()
    data = sorted(q for q, c in coords.items() if int(c[0]) % 2 == 1 and int(c[1]) % 2 == 1)
    meas = sorted(q for q in coords if q not in data)
    return coords, data, meas


def _probe(base: stim.Circuit, data_set, inject_data=None, inject_meas=None) -> stim.Circuit:
    """Noiseless copy of `base` with one injected fault per round."""
    out = stim.Circuit()
    for inst in base:
        if inst.name in NOISE:
            tg = {t.value for t in inst.targets_copy()}
            if inst.name == "DEPOLARIZE1" and tg == data_set and inject_data:
                inject_data(out)
            continue
        if inst.name == "MR" and inject_meas:
            inject_meas(out)
        out.append(inst)
    return out


def _fingerprint(circ: stim.Circuit):
    sig = []
    dem = circ.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)
    for inst in dem.flattened():
        if inst.type == "error":
            dets = tuple(sorted(t.val for t in inst.targets_copy() if t.is_relative_detector_id()))
            if dets:
                sig.append(dets)
    return tuple(sorted(sig))


def candidate_faults(nominal: stim.Circuit):
    base = nominal.flattened()
    coords, data, meas = qubit_roles(base)
    ds = set(data)
    cands = []
    for q in data:
        for pa in ("X", "Z"):
            c = _probe(base, ds, inject_data=lambda o, q=q, pa=pa: o.append(f"{pa}_ERROR", [q], 0.01))
            cands.append(("data_qubit", (q,), pa, _fingerprint(c)))
    for a, b in itertools.combinations(data, 2):
        if abs(coords[a][0] - coords[b][0]) + abs(coords[a][1] - coords[b][1]) == 2:
            for pa, tgt in (("XX", stim.target_x), ("ZZ", stim.target_z)):
                c = _probe(base, ds, inject_data=lambda o, a=a, b=b, t=tgt: o.append(
                    "CORRELATED_ERROR", [t(a), t(b)], 0.01))
                cands.append(("crosstalk", (a, b), pa, _fingerprint(c)))
    for m in meas:
        c = _probe(base, ds, inject_meas=lambda o, m=m: o.append("X_ERROR", [m], 0.01))
        cands.append(("measure_qubit", (m,), "readout", _fingerprint(c)))
    # drop faults the experiment cannot see at all (e.g. ZZ in a Z-memory first round)
    return [c for c in cands if c[3]], coords, data, meas


def _joint_fit(groups, ex_pair, ex_single, w_single=0.3):
    from scipy.optimize import nnls
    rows, entries = {}, []
    def row(k):
        return rows.setdefault(k, len(rows))
    for j, sig in enumerate(groups):
        for s in sig:
            for i in s:
                entries.append((row(("s", i)), j, w_single))
            for a, b in itertools.combinations(s, 2):
                entries.append((row(("p", a, b)), j, 1.0))
    A = np.zeros((len(rows), len(groups)))
    for r, j, v in entries:
        A[r, j] += v
    y = np.array([w_single * ex_single[k[1]] if k[0] == "s" else ex_pair[k[1], k[2]] for k in rows])
    rates, _ = nnls(A, y, maxiter=50 * A.shape[1])
    return rates, float(np.sqrt(np.mean((y - A @ rates) ** 2)))


# ---------------------------------------------------------------- main entry
def diagnose(detections: np.ndarray, nominal: stim.Circuit, p_nominal: float,
             baseline_shots: int = 2_000_000, threshold: float = 0.5, seed: int = 11) -> Report:
    """
    detections   : (shots, num_detectors) array of 0/1 detection events from the chip
    nominal      : the designed circuit with its nominal noise model
    p_nominal    : nominal physical error rate, used to express findings as "x times spec"
    threshold    : flag a finding when its extra rate exceeds threshold * p_nominal
    """
    if detections.shape[1] != nominal.num_detectors:
        raise ValueError(f"detections have {detections.shape[1]} columns, circuit has "
                         f"{nominal.num_detectors} detectors")
    base_det = nominal.compile_detector_sampler(seed=seed).sample(baseline_shots).astype(np.uint8)
    m_r, p_r = pair_stats(detections)
    m_n, p_n = pair_stats(base_det)
    cands, *_ = candidate_faults(nominal)
    groups = {}
    for kind, qs, pa, sig in cands:
        groups.setdefault(sig, []).append((kind, qs, pa))
    sigs = list(groups)
    rates, rms = _joint_fit(sigs, p_r - p_n, m_r - m_n)
    findings = [Finding(kind=groups[s][0][0], members=groups[s], extra_rate=float(r),
                        times_spec=1 + float(r) / p_nominal, flagged=bool(r > threshold * p_nominal))
                for s, r in zip(sigs, rates)]
    findings.sort(key=lambda f: -f.extra_rate)
    return Report(findings=findings, fit_rms=rms)


def score_against_truth(report: Report, truth: dict) -> dict:
    """Benchmark helper for simulated chips with known hidden defects."""
    want = [("data_qubit", (q,)) for q in truth["bad_data_qubits"]]
    want += [("crosstalk", tuple(sorted(p))) for p in truth["crosstalk_pairs"]]
    want += [("measure_qubit", (q,)) for q in truth["bad_readout_qubits"]]
    def hit(f, w):
        return any((k, q) == w for k, q, _ in f.members)
    fl = report.flagged
    found = [w for w in want if any(hit(f, w) for f in fl)]
    exact = [w for w in want if any(hit(f, w) and f.exact for f in fl)]
    false = [f for f in fl if not any(hit(f, w) for w in want)]
    return {"true_defects": len(want), "localised": len(found), "exact": len(exact),
            "false_alarms": len(false), "missed": [str(w) for w in want if w not in found]}
