"""
Circuit-agnostic noise calibration from syndrome data.

Works on any Stim circuit with a noise model (Google's SI1000 circuits, repetition codes,
Stim-generated codes, ...). The nominal noise is split into physical *components*:

  ("qubit", q)       single-qubit gate / idle noise on qubit q
  ("spam", q)        state-preparation and measurement error on qubit q
  ("gate", a, b)     two-qubit gate noise on the pair (a, b)
  ("xtalk", a, b, P) optional: correlated PP error on a next-nearest-neighbour pair that has
                     no gate between them (absent from the nominal model)

For each component we compute, with Stim, the probability it contributes to every graph-like
error mechanism (detector pair or boundary edge). The same mechanism probabilities are
estimated from the recorded detection events, and a regularised non-negative least-squares fit
finds one scale factor per component:

      p_measured(edge) ~= sum_c  scale_c * p_nominal(edge, c)

scale_c = 1 means "as specified", 3 means "three times worse", 0.5 "twice as good".
Crosstalk components have no nominal rate; their fitted value is an absolute rate.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import stim

_P1 = {"DEPOLARIZE1", "X_ERROR", "Y_ERROR", "Z_ERROR", "PAULI_CHANNEL_1"}
_P2 = {"DEPOLARIZE2", "PAULI_CHANNEL_2"}
_MEAS = {"M", "MZ", "MX", "MY", "MR", "MRZ", "MRX", "MRY"}
_RESET = {"R", "RZ", "RX", "RY", "MR", "MRZ", "MRX", "MRY"}
_MAXP = {"DEPOLARIZE1": 0.75, "DEPOLARIZE2": 0.9375, "X_ERROR": 0.5, "Y_ERROR": 0.5,
         "Z_ERROR": 0.5}


# ---------------------------------------------------------------- circuit surgery
def _classify(circ: stim.Circuit):
    """Yield (instruction, [(component, target_group), ...]) over a flattened circuit."""
    insts = list(circ.flattened())
    out = []
    for k, inst in enumerate(insts):
        name, tg = inst.name, inst.targets_copy()
        if name in _P1:
            # skip over neighbouring single-qubit noise instructions: with per-qubit rates, Stim
            # keeps one X_ERROR per qubit in a row before the M, and all of them are readout.
            j = k - 1
            while j >= 0 and insts[j].name in _P1:
                j -= 1
            prev = insts[j] if j >= 0 else None
            j = k + 1
            while j < len(insts) and insts[j].name in _P1:
                j += 1
            nxt = insts[j] if j < len(insts) else None
            prevq = {t.value for t in prev.targets_copy()} if prev and prev.name in _RESET else set()
            nxtq = {t.value for t in nxt.targets_copy()} if nxt and nxt.name in _MEAS else set()
            groups = []
            for t in tg:
                kind = "spam" if (t.value in prevq or t.value in nxtq) and name != "DEPOLARIZE1" else "qubit"
                groups.append(((kind, t.value), [t]))
            out.append((inst, groups))
        elif name in _P2:
            groups = []
            for a, b in zip(tg[::2], tg[1::2]):
                groups.append((("gate",) + tuple(sorted((a.value, b.value))), [a, b]))
            out.append((inst, groups))
        elif name in _MEAS and inst.gate_args_copy() and inst.gate_args_copy()[0] > 0:
            out.append((inst, [(("spam", t.value), [t]) for t in tg]))
        else:
            out.append((inst, None))
    return out


def rebuild(circ: stim.Circuit, scale, extra=None) -> stim.Circuit:
    """Copy of `circ` with each component's noise multiplied by scale(component).
    `extra` = {component: rate} adds correlated crosstalk at every layer where both qubits
    are hit by the same single-qubit noise instruction."""
    extra = extra or {}
    xt_by_pair = {}
    for c, r in extra.items():
        xt_by_pair.setdefault((c[1], c[2]), []).append((c[3], r))
    out = stim.Circuit()
    for inst, groups in _classify(circ):
        if groups is None:
            out.append(inst)
            continue
        name, args = inst.name, inst.gate_args_copy()
        if name in _MEAS:
            for comp, tg in groups:
                f = scale(comp)
                out.append(name, tg, [min(0.5, args[0] * f)])
            continue
        for comp, tg in groups:
            f = scale(comp)
            if f <= 0:
                continue
            if name.startswith("PAULI_CHANNEL"):
                a = [x * f for x in args]
                s = sum(a)
                if s > 0.999:
                    a = [x * 0.999 / s for x in a]
            else:
                a = [min(_MAXP.get(name, 0.5), args[0] * f)]
            out.append(name, tg, a)
        if name == "DEPOLARIZE1" and xt_by_pair:
            hit = {t.value for _, tg in groups for t in tg}
            for (qa, qb), lst in xt_by_pair.items():
                if qa in hit and qb in hit:
                    for pauli, r in lst:
                        if r > 0:
                            mk = {"X": stim.target_x, "Z": stim.target_z}[pauli[0]]
                            out.append("CORRELATED_ERROR", [mk(qa), mk(qb)], [min(0.5, r)])
    return out


def components(circ: stim.Circuit):
    seen = []
    for _, groups in _classify(circ):
        for comp, _ in groups or []:
            if comp not in seen:
                seen.append(comp)
    return seen


def crosstalk_candidates(circ: stim.Circuit):
    """Next-nearest neighbours: pairs that share a two-qubit-gate partner but no gate."""
    gates = {c[1:] for c in components(circ) if c[0] == "gate"}
    nbr = {}
    for a, b in gates:
        nbr.setdefault(a, set()).add(b)
        nbr.setdefault(b, set()).add(a)
    pairs = set()
    for m, ns in nbr.items():
        for a, b in itertools.combinations(sorted(ns), 2):
            if (a, b) not in gates:
                pairs.add((a, b))
    return [("xtalk", a, b, p) for a, b in sorted(pairs) for p in ("XX", "ZZ")]


# ---------------------------------------------------------------- DEM -> observables
# Two kinds of observable, both additive over independent error mechanisms:
#   ("m", i)    : -1/2 ln(1 - 2<x_i>)  = sum over mechanisms touching detector i of ~p
#   ("p", i, j) : pairwise mechanism estimate p_ij = sum over mechanisms touching i AND j of ~p
# Hyperedges (mechanisms flipping 3+ detectors) contribute to every pair they contain, which
# is exactly what the p_ij estimator sees, so nothing is dropped.
def signature(circ: stim.Circuit) -> dict:
    dem = circ.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)
    sig = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        q = -0.5 * np.log(max(1e-12, 1 - 2 * p))
        S = sorted(t.val for t in inst.targets_copy() if t.is_relative_detector_id())
        for i in S:
            sig[("m", i)] = sig.get(("m", i), 0.0) + q
        for i, j in itertools.combinations(S, 2):
            sig[("p", i, j)] = sig.get(("p", i, j), 0.0) + p
    return sig


def measured_obs(det: np.ndarray, keys, chunk: int = 200_000) -> np.ndarray:
    n, D = det.shape
    s1, s2 = np.zeros(D), np.zeros((D, D))
    for i in range(0, n, chunk):
        x = det[i:i + chunk].astype(np.float32)
        s1 += x.sum(0)
        s2 += x.T @ x
    m1, m2 = s1 / n, s2 / n
    num = m2 - np.outer(m1, m1)
    den = 1 - 2 * m1[:, None] - 2 * m1[None, :] + 4 * m2
    with np.errstate(invalid="ignore", divide="ignore"):
        pij = np.nan_to_num(0.5 - 0.5 * np.sqrt(np.clip(1 - 4 * num / den, 0, None)))
    mo = -0.5 * np.log(np.clip(1 - 2 * m1, 1e-9, None))
    return np.array([mo[k[1]] if k[0] == "m" else pij[k[1], k[2]] for k in keys])


# ---------------------------------------------------------------- fit
@dataclass
class ComponentFit:
    component: tuple
    scale: float              # x times nominal (absolute rate for crosstalk)
    err: float                # bootstrap standard deviation
    nominal_weight: float     # how much detector signal this component carries
    members: list = field(default_factory=list)   # syndrome-equivalent alternatives

    @property
    def label(self) -> str:
        names = [self._name(c) for c in (self.members or [self.component])]
        return " or ".join(names)

    @staticmethod
    def _name(c) -> str:
        if c[0] == "qubit":
            return f"qubit {c[1]} gate/idle noise"
        if c[0] == "spam":
            return f"qubit {c[1]} prep/readout"
        if c[0] == "gate":
            return f"2-qubit gate {c[1]}-{c[2]}"
        return f"crosstalk {c[1]}-{c[2]} ({c[3]})"


@dataclass
class NoiseFit:
    fits: list = field(default_factory=list)
    circuit: stim.Circuit | None = None
    shots: int = 0
    p_typical: float = 1e-3      # median two-qubit-gate error in the nominal model

    def get(self, component):
        return next((f for f in self.fits if f.component == component), None)

    def calibrated_circuit(self, floor: float = 0.0, crosstalk: bool = True) -> stim.Circuit:
        """floor: lowest allowed scale factor. A component fitted to zero removes edges from the decoding
        graph, which a decoder cannot recover from if that error then happens.
        crosstalk=False leaves fitted crosstalk terms out. On a repetition code a fitted two-data-qubit term can
        add a logical-flipping shortcut to the decoding graph, so decoders should use crosstalk=False there;
        the terms are still reported as diagnostics."""
        sc = {f.component: max(floor, f.scale) for f in self.fits if f.component[0] != "xtalk"}
        xt = {f.component: f.scale for f in self.fits if f.component[0] == "xtalk" and f.scale > 0} if crosstalk else {}
        return rebuild(self.circuit, lambda c: sc.get(c, 1.0), extra=xt)

    def outliers(self, worse=1.5, z=3.0, xtalk_rate=None):
        """Components worse than `worse` x nominal, and crosstalk above `xtalk_rate`
        (default: half the typical two-qubit gate error), both with z-score above `z`."""
        xtalk_rate = 0.5 * self.p_typical if xtalk_rate is None else xtalk_rate
        res = []
        for f in self.fits:
            if f.component[0] == "xtalk":
                if f.scale > xtalk_rate and f.scale > z * f.err:
                    res.append(f)
            elif f.scale > worse and (f.scale - 1) > z * f.err:
                res.append(f)
        return sorted(res, key=lambda f: -(f.scale if f.component[0] != "xtalk" else 1 + f.scale * 1e3))

    def table(self, only_outliers=False):
        rows = self.outliers() if only_outliers else sorted(self.fits, key=lambda f: -f.scale)
        lines = []
        for f in rows:
            if f.component[0] == "xtalk":
                lines.append(f"  {f.label:34s} rate {f.scale:.4f} +/- {f.err:.4f}")
            else:
                lines.append(f"  {f.label:34s} {f.scale:5.2f}x nominal +/- {f.err:.2f}")
        return "\n".join(lines) or "  none"


def _design(circ, comps, xtalk, probe_rate=1e-3):
    cols = []
    for c in comps:
        cols.append(signature(rebuild(circ, lambda x, c=c: 1.0 if x == c else 0.0)))
    for c in xtalk:
        probe = rebuild(circ, lambda x: 0.0, extra={c: probe_rate})
        e = signature(probe)
        cols.append({k: v / probe_rate for k, v in e.items()})   # per unit rate
    keys = sorted(set().union(*[c.keys() for c in cols]))
    idx = {k: i for i, k in enumerate(keys)}
    A = np.zeros((len(keys), len(cols)))
    for j, col in enumerate(cols):
        for k, v in col.items():
            A[idx[k], j] = v
    return keys, A


_IS_M = None


def _weights(A, y, nominal_cols, shots):
    """~ inverse standard deviation of each observable's estimate."""
    pnom = A[:, :nominal_cols].sum(1)
    w = np.sqrt(shots / np.maximum(pnom + y, 1e-5))
    if _IS_M is not None:                                     # marginals: var ~ m/N
        m = np.clip(0.5 * (1 - np.exp(-2 * np.maximum(pnom, y))), 1e-5, 0.49)
        w = np.where(_IS_M, np.sqrt(shots) * (1 - 2 * m) / np.sqrt(m * (1 - m)), w)
    return w


def _select(A, y, nominal_cols, shots, strength):
    """Sparse step: L1 penalty on deviations from nominal (scale 1) and on crosstalk (0).
    Assumes most components are as specified; returns the set allowed to deviate."""
    from scipy.optimize import minimize
    w = _weights(A, y, nominal_cols, shots)
    Aw, yw = A * w[:, None], y * w
    n = A.shape[1]
    base = np.zeros(n)
    base[:nominal_cols] = 1.0
    r0 = yw - Aw @ base
    lam = strength * np.linalg.norm(Aw, axis=0) * np.sqrt(2 * np.log(max(n, 2)))
    # variables: u (n, >=0) and v (nominal only, 0..1); delta = u - v
    def f(z):
        u, v = z[:n], z[n:]
        d = u.copy()
        d[:nominal_cols] -= v
        r = r0 - Aw @ d
        g = -Aw.T @ r
        val = 0.5 * r @ r + lam @ u + lam[:nominal_cols] @ v
        return val, np.concatenate([g + lam, -g[:nominal_cols] + lam[:nominal_cols]])
    z0 = np.zeros(n + nominal_cols)
    bounds = [(0, None)] * n + [(0, 1)] * nominal_cols
    z = minimize(f, z0, jac=True, method="L-BFGS-B", bounds=bounds,
                 options={"maxiter": 5000, "ftol": 1e-14, "gtol": 1e-10}).x
    d = z[:n].copy()
    d[:nominal_cols] -= z[n:]
    return np.abs(d) > 1e-9


def _refit(A, y, nominal_cols, shots, support):
    """Unpenalised fit of the selected components; everything else stays nominal."""
    from scipy.optimize import lsq_linear
    w = _weights(A, y, nominal_cols, shots)
    n = A.shape[1]
    x = np.zeros(n)
    x[:nominal_cols] = 1.0
    if support.any():
        fixed = ~support
        r = (y - A[:, fixed] @ x[fixed]) * w
        x[support] = lsq_linear(A[:, support] * w[:, None], r, bounds=(0, np.inf), method="bvls").x
    return x


def fit_noise(circuit: stim.Circuit, detections: np.ndarray, *, crosstalk: bool = True,
              bootstrap: int = 5, sparsity: float = 2.0) -> NoiseFit:
    """Fit per-component scale factors of `circuit`'s noise model to recorded detections.

    sparsity : strength of the "most components are as specified" assumption. 2.0 was best in
               benchmarks (zero false alarms at medium and weak defect strengths); 0 lets every
               component move freely.
    """
    circuit = circuit.flattened()
    if detections.shape[1] != circuit.num_detectors:
        raise ValueError(f"detections have {detections.shape[1]} columns, circuit has "
                         f"{circuit.num_detectors} detectors")
    comps = components(circuit)
    xt = crosstalk_candidates(circuit) if crosstalk else []
    keys, A = _design(circuit, comps, xt)
    keep = A.sum(0) > 0
    comps_all = comps + xt
    # merge crosstalk candidates whose signatures are identical: no data can separate them
    members = {j: [comps_all[j]] for j in range(len(comps_all))}
    norm = A / np.maximum(np.linalg.norm(A, axis=0), 1e-30)
    xt_idx = [j for j in range(len(comps), len(comps_all)) if keep[j]]
    for a_i, a in enumerate(xt_idx):
        if not keep[a]:
            continue
        for b in xt_idx[a_i + 1:]:
            if keep[b] and norm[:, a] @ norm[:, b] > 0.9999:
                members[a].append(comps_all[b])
                keep[b] = False
    n = len(detections)

    Ak = A[:, keep]
    nk = int(keep[:len(comps)].sum())

    global _IS_M
    _IS_M = np.array([k[0] == "m" for k in keys])

    def edges_of(det):
        return measured_obs(det, keys)

    y_all = edges_of(detections)
    support = _select(Ak, y_all, nk, n, sparsity) if sparsity > 0 else np.ones(Ak.shape[1], bool)

    def run(det, y=None):
        y = edges_of(det) if y is None else y
        x = np.zeros(A.shape[1])
        x[keep] = _refit(Ak, y, nk, len(det), support)
        return x

    est = run(detections, y_all)
    if bootstrap > 1:
        parts = np.array_split(np.random.default_rng(0).permutation(n), bootstrap)
        boots = np.array([run(detections[p]) for p in parts])
        err = boots.std(0) / np.sqrt(bootstrap)       # std of the mean of independent halves
    else:
        err = np.zeros_like(est)
    weight = A.sum(0)
    fits = [ComponentFit(c, float(s), float(e), float(wt), members[j] if len(members[j]) > 1 else [])
            for j, (c, s, e, wt, k) in enumerate(zip(comps_all, est, err, weight, keep)) if k]
    p2 = [inst.gate_args_copy()[0] for inst in circuit if inst.name == "DEPOLARIZE2"]
    p_typ = float(np.median(p2)) if p2 else 1e-3
    return NoiseFit(fits=fits, circuit=circuit, shots=n, p_typical=p_typ)
