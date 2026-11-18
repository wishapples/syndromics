"""
Repetition-code diagnostics on IBM Quantum hardware (or a noisy simulation of it).

IBM chips use a heavy-hex layout, so a rotated surface code does not fit natively; a
repetition code on a line of 2n-1 qubits does. Each round, ancilla a_i measures the parity of
data qubits d_i and d_{i+1}, and is measured and reset mid-circuit.

Why this is a good validation: IBM publishes per-qubit readout errors and per-pair gate errors
for every device, updated with each calibration. We give Syndromics a *uniform* nominal model
(one median value for all qubits) and check whether the per-qubit variation it finds from
syndrome data alone matches IBM's published calibration.

Requires: qiskit, qiskit-ibm-runtime (and qiskit-aer for simulation).
"""
from __future__ import annotations

import numpy as np
import stim

from .model import fit_noise


# ---------------------------------------------------------------- layout
def find_line(backend, length: int, exclude=()):
    """A path of `length` physical qubits on the coupling map, preferring low readout and
    gate errors. Greedy DFS from every start; returns the best path found."""
    cmap = backend.coupling_map
    nbr = {}
    for a, b in cmap.get_edges():
        nbr.setdefault(a, set()).add(b)
        nbr.setdefault(b, set()).add(a)
    ro = _readout_errors(backend)
    g2 = _gate_errors(backend)
    def cost(path):
        c = sum(ro.get(q, 0.05) for q in path)
        c += sum(g2.get(tuple(sorted(p)), 0.05) for p in zip(path, path[1:]))
        return c
    best, best_c = None, np.inf
    for start in sorted(nbr):
        if start in exclude:
            continue
        stack = [[start]]
        tried = 0
        while stack and tried < 2000:
            path = stack.pop()
            tried += 1
            if len(path) == length:
                c = cost(path)
                if c < best_c:
                    best, best_c = path, c
                continue
            nxt = sorted((q for q in nbr.get(path[-1], ()) if q not in path and q not in exclude),
                         key=lambda q: -ro.get(q, 0.05))
            for q in nxt:
                stack.append(path + [q])
    if best is None:
        raise ValueError(f"no path of {length} qubits found")
    return best


def _two_qubit_gate_name(backend):
    for g in ("cz", "ecr", "cx"):
        if g in backend.target.operation_names:
            return g
    raise ValueError("backend has no cz/ecr/cx gate")


def _readout_errors(backend):
    t = backend.target
    out = {}
    for qargs, props in (t["measure"].items() if "measure" in t else []):
        if props is not None and props.error is not None:
            out[qargs[0]] = props.error
    return out


def _gate_errors(backend):
    t = backend.target
    g = _two_qubit_gate_name(backend)
    out = {}
    for qargs, props in t[g].items():
        if props is not None and props.error is not None:
            k = tuple(sorted(qargs))
            out[k] = min(out.get(k, 1.0), props.error)
    return out


def _sq_errors(backend):
    t = backend.target
    name = "sx" if "sx" in t.operation_names else "x"
    return {q[0]: p.error for q, p in t[name].items() if p is not None and p.error is not None}


# ---------------------------------------------------------------- circuits
def rep_code_stim(n_data: int, rounds: int, readout, gate2, idle) -> stim.Circuit:
    """Repetition-code memory on a line. Line index k: data at even k, ancilla at odd k.
    readout[k], idle[k]: per-line-index probabilities; gate2[(k, k+1)]: CX depolarizing."""
    L = 2 * n_data - 1
    data = list(range(0, L, 2)); anc = list(range(1, L, 2))
    c = stim.Circuit()
    for k in range(L):
        c.append("QUBIT_COORDS", [k], [k, 0])
    c.append("R", range(L))
    c.append("TICK")
    nm = 0
    for r in range(rounds):
        for layer in (0, 1):
            pairs = [(a - 1, a) if layer == 0 else (a + 1, a) for a in anc]
            for d, a in pairs:
                c.append("CX", [d, a])
            for d, a in pairs:
                c.append("DEPOLARIZE2", [d, a], [gate2[tuple(sorted((d, a)))]])
            c.append("TICK")
        for a in anc:
            c.append("X_ERROR", [a], [readout[a]])
        c.append("M", anc)
        c.append("R", anc)
        for d in data:
            c.append("DEPOLARIZE1", [d], [idle[d]])
        nm += len(anc)
        for i, a in enumerate(anc):
            back = len(anc) - i
            if r == 0:
                c.append("DETECTOR", [stim.target_rec(-back)], [a, r])
            else:
                c.append("DETECTOR", [stim.target_rec(-back), stim.target_rec(-back - len(anc))], [a, r])
        c.append("TICK")
    for d in data:
        c.append("X_ERROR", [d], [readout[d]])
    c.append("M", data)
    for i, a in enumerate(anc):
        dl, dr = i, i + 1
        c.append("DETECTOR", [stim.target_rec(-n_data + dl), stim.target_rec(-n_data + dr),
                              stim.target_rec(-n_data - len(anc) + i)], [a, rounds])
    c.append("OBSERVABLE_INCLUDE", [stim.target_rec(-n_data)], [0])
    return c


def rep_code_qiskit(n_data: int, rounds: int):
    """Same circuit in Qiskit. Classical bits are laid out in the same order as Stim's
    measurement record: round 0 ancillas, round 1 ancillas, ..., final data."""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    L = 2 * n_data - 1
    data = list(range(0, L, 2)); anc = list(range(1, L, 2))
    q = QuantumRegister(L, "q")
    regs = [ClassicalRegister(len(anc), f"r{r}") for r in range(rounds)]
    fin = ClassicalRegister(n_data, "final")
    qc = QuantumCircuit(q, *regs, fin)
    for r in range(rounds):
        for a in anc:
            qc.cx(a - 1, a)
        for a in anc:
            qc.cx(a + 1, a)
        for i, a in enumerate(anc):
            qc.measure(a, regs[r][i])
        for a in anc:
            qc.reset(a)
    for i, d in enumerate(data):
        qc.measure(d, fin[i])
    return qc


def nominal_model(backend, line, mode="uniform", meas_time_ns=None):
    """Per-line-index noise. mode='uniform' uses the median of the line's calibration values
    for every qubit (what Syndromics starts from); mode='calibrated' uses IBM's per-qubit values."""
    ro_all, g2_all, sq_all = _readout_errors(backend), _gate_errors(backend), _sq_errors(backend)
    L = len(line)
    ro = {k: ro_all.get(line[k], 0.02) for k in range(L)}
    g2 = {(k, k + 1): g2_all.get(tuple(sorted((line[k], line[k + 1]))), 0.01) for k in range(L - 1)}
    # idle error of a data qubit during the mid-circuit measurement + reset of the ancillas
    t = backend.target
    meas_ns = meas_time_ns
    if meas_ns is None:
        try:
            meas_ns = float(np.median([p.duration for p in t["measure"].values() if p and p.duration])) * 1e9
        except Exception:
            meas_ns = 1500.0
    idle = {}
    for k in range(L):
        qp = t.qubit_properties[line[k]] if t.qubit_properties else None
        T1 = getattr(qp, "t1", None) or 150e-6
        T2 = getattr(qp, "t2", None) or 100e-6
        tt = meas_ns * 1e-9
        p_x = (1 - np.exp(-tt / T1)) / 4
        p_z = max(0.0, (1 - np.exp(-tt / T2)) / 2 - p_x)
        idle[k] = float(min(0.5, 3 * p_x + p_z)) if True else 0.0
        idle[k] = max(idle[k], sq_all.get(line[k], 1e-4))
    if mode == "uniform":
        mro, mg, mi = np.median(list(ro.values())), np.median(list(g2.values())), np.median(list(idle.values()))
        ro = {k: float(mro) for k in ro}; g2 = {k: float(mg) for k in g2}; idle = {k: float(mi) for k in idle}
    return ro, g2, idle


def run_sampler(backend, n_data, rounds, line, shots, sampler=None):
    """Execute on hardware (or a simulator backend) and return measurements in Stim order."""
    from qiskit import transpile
    qc = rep_code_qiskit(n_data, rounds)
    tqc = transpile(qc, backend, initial_layout=line, optimization_level=1, seed_transpiler=1)
    if sampler is None:
        from qiskit_ibm_runtime import SamplerV2
        sampler = SamplerV2(mode=backend)
    res = sampler.run([tqc], shots=shots).result()[0]
    cols = []
    for r in range(rounds):
        cols.append(getattr(res.data, f"r{r}").to_bool_array(order="little"))
    cols.append(res.data.final.to_bool_array(order="little"))
    return np.hstack(cols).astype(np.uint8)


def analyse(meas, nominal_circuit: stim.Circuit, **kw):
    conv = nominal_circuit.compile_m2d_converter()
    det, obs = conv.convert(measurements=meas.astype(bool), separate_observables=True)
    fit = fit_noise(nominal_circuit, det.astype(np.uint8), **kw)
    return fit, det, obs


def compare_with_calibration(fit, backend, line, uniform):
    """Fitted per-qubit readout and per-pair gate errors vs IBM's published values."""
    ro_u, g2_u, _ = uniform
    ro_cal, g2_cal, _ = nominal_model(backend, line, mode="calibrated")
    rows = []
    for k in range(len(line)):
        f = fit.get(("spam", k))
        if f is not None:
            rows.append(("readout", f"q{line[k]}", ro_cal[k], f.scale * ro_u[k], f.err * ro_u[k]))
    for (a, b), pu in g2_u.items():
        f = fit.get(("gate", a, b))
        if f is not None:
            rows.append(("gate", f"q{line[a]}-q{line[b]}", g2_cal[(a, b)], f.scale * pu, f.err * pu))
    out = {"rows": rows}
    for kind in ("readout", "gate"):
        xs = [r[2] for r in rows if r[0] == kind]; ys = [r[3] for r in rows if r[0] == kind]
        if len(xs) >= 3 and np.std(xs) > 0 and np.std(ys) > 0:
            out[f"{kind}_pearson"] = float(np.corrcoef(xs, ys)[0, 1])
            rk = lambda v: np.argsort(np.argsort(v))
            out[f"{kind}_spearman"] = float(np.corrcoef(rk(xs), rk(ys))[0, 1])
    return out
