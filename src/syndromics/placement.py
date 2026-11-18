"""
Noise-aware placement: put a circuit on the qubits that are healthy *today*.

Compilers such as Qiskit already choose qubits using the device's published calibration.
That calibration is a snapshot: qubits drift, two-level-system defects appear and disappear,
and readout degrades between calibrations. Syndromics measures the chip from syndrome data
(repetition-code runs on lines of qubits), turns the result into a ChipHealth record, and
feeds the corrected error rates back into the compiler's Target, so the compiler's own
noise-aware layout avoids what is broken now.

    health = diagnose_chip(backend, executor)          # minutes of machine time
    result = place(circuit, backend, health)            # best layout under today's errors
"""
from __future__ import annotations

import copy
import datetime as _dt
import json
from dataclasses import dataclass, field

import numpy as np

from . import ibm


# ---------------------------------------------------------------- health record
@dataclass
class ChipHealth:
    backend: str
    created: str
    readout: dict = field(default_factory=dict)     # physical qubit -> readout error
    sq: dict = field(default_factory=dict)          # physical qubit -> single-qubit gate error
    cz: dict = field(default_factory=dict)          # (a, b) sorted -> two-qubit gate error
    flagged: dict = field(default_factory=dict)     # changed since calibration: "q12"/"q12-q13" -> reason
    known_bad: dict = field(default_factory=dict)   # bad, but the published calibration already says so
    lines: list = field(default_factory=list)       # diagnostic lines that were run
    source: str = "syndromics"

    @property
    def measured_qubits(self):
        return set(self.readout)

    @property
    def measured_pairs(self):
        return set(self.cz)

    def bad_qubits(self):
        return sorted({int(k[1:]) for k in self.flagged if "-" not in k})

    def bad_pairs(self):
        out = []
        for k in self.flagged:
            if "-" in k:
                a, b = k.split("-")
                out.append((int(a[1:]), int(b[1:])))
        return sorted(out)

    def to_json(self) -> str:
        return json.dumps({
            "format": "syndromics.chiphealth/1", "backend": self.backend, "created": self.created,
            "source": self.source, "lines": self.lines,
            "readout": {str(k): v for k, v in sorted(self.readout.items())},
            "sq": {str(k): v for k, v in sorted(self.sq.items())},
            "cz": {f"{a}-{b}": v for (a, b), v in sorted(self.cz.items())},
            "flagged": self.flagged, "known_bad": self.known_bad,
        }, indent=1)

    @classmethod
    def from_json(cls, text: str) -> "ChipHealth":
        d = json.loads(text)
        if d.get("format") != "syndromics.chiphealth/1":
            raise ValueError("not a Syndromics chip-health file")
        cz = {}
        for k, v in d["cz"].items():
            a, b = k.split("-")
            cz[(int(a), int(b))] = v
        return cls(backend=d["backend"], created=d["created"], source=d.get("source", ""),
                   lines=d.get("lines", []),
                   readout={int(k): v for k, v in d["readout"].items()},
                   sq={int(k): v for k, v in d["sq"].items()}, cz=cz, flagged=d["flagged"],
                   known_bad=d.get("known_bad", {}))


# ---------------------------------------------------------------- diagnosis
def unusable_qubits(backend, max_readout=0.2, max_gate=0.05):
    """Qubits the published calibration already rules out; compilers avoid them anyway."""
    ro, g2 = ibm._readout_errors(backend), ibm._gate_errors(backend)
    bad = {q for q, e in ro.items() if e > max_readout}
    per_q = {}
    for (a, b), e in g2.items():
        per_q.setdefault(a, []).append(e); per_q.setdefault(b, []).append(e)
    bad |= {q for q, es in per_q.items() if min(es) > max_gate}
    return bad


def cover_with_lines(backend, length=19, min_length=7, max_lines=12):
    """Disjoint lines of qubits that together cover as much of the usable chip as possible."""
    used, lines = set(unusable_qubits(backend)), []
    L = length
    while L >= min_length and len(lines) < max_lines:
        try:
            line = ibm.find_line(backend, L, exclude=used)
        except ValueError:
            L -= 2
            continue
        lines.append(line)
        used |= set(line)
    return lines


def diagnose_chip(backend, executor, lines=None, rounds=6, shots=200_000,
                  ratio=2.0, z=4.0, known_bad_factor=4.0, log=print) -> ChipHealth:
    """Run a repetition-code diagnostic on each line and merge the fits into a ChipHealth.

    executor(line, n_data, rounds, shots) -> measurement array in Stim order. On hardware use
    `hardware_executor(backend)`; in simulation use `SimulatedChip.executor`."""
    lines = lines or cover_with_lines(backend)
    pub_ro, pub_cz, pub_sq = ibm._readout_errors(backend), ibm._gate_errors(backend), ibm._sq_errors(backend)
    h = ChipHealth(backend=getattr(backend, "name", str(backend)),
                   created=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                   lines=[list(map(int, l)) for l in lines])
    med = {"readout": float(np.median(list(pub_ro.values()))) if pub_ro else 0.01,
           "gate": float(np.median(list(pub_cz.values()))) if pub_cz else 0.003}
    for line in lines:
        n_data = (len(line) + 1) // 2
        # start from the published per-qubit calibration, so the fit only explains what changed
        ro_u, g2_u, idle_u = ibm.nominal_model(backend, line, mode="calibrated")
        idle_cal = idle_u
        nominal = ibm.rep_code_stim(n_data, rounds, ro_u, g2_u, idle_u)
        meas = executor(line, n_data, rounds, shots)
        fit, _, _ = ibm.analyse(meas, nominal)
        found = 0

        def judge(key, est, sig, pub, what):
            # "changed since calibration": clearly above this qubit's own published value
            nonlocal found
            if est > ratio * pub and est - pub > z * sig:
                h.flagged[key] = f"{what} {est:.4f}, calibration says {pub:.4f} ({est/pub:.1f}x)"
                found += 1
            elif pub > known_bad_factor * med[what] and key not in h.known_bad:
                h.known_bad[key] = f"{what} {pub:.4f} (already in calibration)"

        for k, q in enumerate(line):
            f = fit.get(("spam", k))
            if f is not None:
                est = float(max(0.0, f.scale) * ro_u[k])
                h.readout[q] = est
                judge(f"q{q}", est, f.err * ro_u[k], pub_ro.get(q, ro_u[k]), "readout")
            f = fit.get(("qubit", k))
            if f is not None:
                factor = max(0.0, f.scale) * idle_u[k] / max(idle_cal[k], 1e-9)
                h.sq[q] = float(factor * pub_sq.get(q, 2.5e-4))
                if f"q{q}" not in h.flagged and factor > ratio and \
                        (f.scale * idle_u[k] - idle_cal[k]) > z * f.err * idle_u[k]:
                    h.flagged[f"q{q}"] = f"coherence/idle error {factor:.1f}x calibration (TLS defect or drift?)"
                    found += 1
        for (a, b) in g2_u:
            f = fit.get(("gate", a, b))
            if f is None:
                continue
            pa, pb = sorted((line[a], line[b]))
            est = float(max(0.0, f.scale) * g2_u[(a, b)])
            h.cz[(pa, pb)] = est
            judge(f"q{pa}-q{pb}", est, f.err * g2_u[(a, b)], pub_cz.get((pa, pb), g2_u[(a, b)]), "gate")
        log(f"  line of {len(line):2d} qubits starting q{line[0]:<3d}: {found} change(s)")
    return h


def hardware_executor(backend, sampler=None):
    def run(line, n_data, rounds, shots):
        return ibm.run_sampler(backend, n_data, rounds, line, shots, sampler=sampler)
    return run


# ---------------------------------------------------------------- corrected target
def corrected_target(backend, health: ChipHealth):
    """A copy of the backend's Target with today's measured error rates written in."""
    from qiskit.transpiler import InstructionProperties
    t = copy.deepcopy(backend.target)
    g2 = ibm._two_qubit_gate_name(backend)

    def upd(name, qargs, err):
        if name not in t or qargs not in t[name]:
            return
        old = t[name][qargs]
        t.update_instruction_properties(name, qargs, InstructionProperties(
            duration=getattr(old, "duration", None), error=float(min(err, 0.75))))

    for q, e in health.readout.items():
        upd("measure", (q,), e)
    for q, e in health.sq.items():
        for name in ("sx", "x"):
            upd(name, (q,), e)
    for (a, b), e in health.cz.items():
        upd(g2, (a, b), e)
        upd(g2, (b, a), e)
    return t


def estimate_success(tqc, target) -> float:
    """Product of (1 - error) over every operation, using the Target's error rates."""
    logp = 0.0
    for inst in tqc.data:
        name = inst.operation.name
        if name in ("barrier", "delay", "rz", "id"):
            continue
        qargs = tuple(tqc.find_bit(q).index for q in inst.qubits)
        try:
            props = target[name][qargs]
        except KeyError:
            continue
        if props is not None and props.error:
            logp += np.log1p(-min(props.error, 0.999))
    return float(np.exp(logp))


def used_pairs(tqc):
    return sorted({tuple(sorted(tqc.find_bit(q).index for q in inst.qubits))
                   for inst in tqc.data if len(inst.qubits) == 2})


def used_qubits(tqc):
    s = set()
    for inst in tqc.data:
        if inst.operation.name in ("barrier", "delay"):
            continue
        s.update(tqc.find_bit(q).index for q in inst.qubits)
    return sorted(s)


@dataclass
class Placement:
    circuit: object             # transpiled circuit
    qubits: list
    estimated_success: float
    avoided: list               # flagged qubits/pairs that this placement does not touch
    touched_flags: list         # flagged items it could not avoid


def _transpile_best(qc, target, seeds, score_target):
    from qiskit import transpile
    best = None
    for s in seeds:
        tqc = transpile(qc, target=target, optimization_level=3, seed_transpiler=int(s))
        sc = estimate_success(tqc, score_target)
        if best is None or sc > best[1]:
            best = (tqc, sc)
    return best


def place(qc, backend, health: ChipHealth | None = None, seeds=range(6)) -> Placement:
    """Compile `qc` for `backend`. With a ChipHealth record, layouts are chosen and scored on
    today's measured errors; without one, on the published calibration (the default)."""
    target = corrected_target(backend, health) if health else backend.target
    tqc, sc = _transpile_best(qc, target, seeds, target)
    used = set(used_qubits(tqc))
    pairs = set()
    for inst in tqc.data:
        if len(inst.qubits) == 2:
            pairs.add(tuple(sorted(tqc.find_bit(q).index for q in inst.qubits)))
    avoided, touched = [], []
    if health:
        for k in health.flagged:
            if "-" in k:
                a, b = (int(x[1:]) for x in k.split("-"))
                (touched if (a, b) in pairs else avoided).append(k)
            else:
                (touched if int(k[1:]) in used else avoided).append(k)
    return Placement(tqc, sorted(used), sc, sorted(avoided), sorted(touched))


# ---------------------------------------------------------------- simulated chip (for demos/tests)
class SimulatedChip:
    """A backend's published calibration plus hidden drift: what the chip is really like today.

    drift: {"readout": {q: err}, "sq": {q: err}, "cz": {(a,b): err}} absolute values that
    override the published ones. Used to (1) generate diagnostic data and (2) score compiled
    circuits by noisy simulation, so placements can be compared against the truth."""

    def __init__(self, backend, drift=None):
        self.backend = backend
        self.drift = drift or {"readout": {}, "sq": {}, "cz": {}}
        self.ro = dict(ibm._readout_errors(backend))
        self.cz = dict(ibm._gate_errors(backend))
        self.sqe = dict(ibm._sq_errors(backend))
        self.ro.update(self.drift.get("readout", {}))
        self.sqe.update(self.drift.get("sq", {}))
        for k, v in self.drift.get("cz", {}).items():
            self.cz[tuple(sorted(k))] = v

    def truth_target(self):
        h = ChipHealth(backend="truth", created="", readout=self.ro, sq=self.sqe, cz=self.cz)
        return corrected_target(self.backend, h)

    def executor(self, line, n_data, rounds, shots, seed=None):
        _, _, idle_cal = ibm.nominal_model(self.backend, line, mode="calibrated")
        pub_sq = ibm._sq_errors(self.backend)
        L = len(line)
        ro = {k: self.ro.get(line[k], 0.02) for k in range(L)}
        g2 = {(k, k + 1): self.cz.get(tuple(sorted((line[k], line[k + 1]))), 0.01) for k in range(L - 1)}
        idle = {k: idle_cal[k] * self.sqe.get(line[k], 1) / max(pub_sq.get(line[k], 1), 1e-9)
                for k in range(L)}
        idle = {k: float(min(v, 0.5)) for k, v in idle.items()}
        circ = ibm.rep_code_stim(n_data, rounds, ro, g2, idle)
        if seed is None:
            seed = abs(hash(tuple(line))) % (2**31)
        return circ.compile_sampler(seed=seed).sample(shots).astype(np.uint8)

    def noise_model(self):
        from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
        nm = NoiseModel(basis_gates=["cz", "sx", "x", "rz", "measure", "reset"])
        for q, e in self.ro.items():
            e = min(e, 0.5)
            nm.add_readout_error(ReadoutError([[1 - e, e], [e, 1 - e]]), [q])
        for q, e in self.sqe.items():
            lam = min(4 / 3 * e, 1.0)
            err = depolarizing_error(lam, 1)
            nm.add_quantum_error(err, "sx", [q])
            nm.add_quantum_error(err, "x", [q])
        for (a, b), e in self.cz.items():
            lam = min(16 / 15 * e, 1.0)
            err = depolarizing_error(lam, 2)
            nm.add_quantum_error(err, "cz", [a, b])
            nm.add_quantum_error(err, "cz", [b, a])
        return nm

    def run(self, tqc, shots=4000, seed=7):
        from qiskit_aer import AerSimulator
        sim = AerSimulator(noise_model=self.noise_model(), seed_simulator=seed)
        return sim.run(tqc, shots=shots).result().get_counts()


def ideal_distribution(qc):
    """Noiseless output distribution of a logical circuit (measured bits, Qiskit key order)."""
    from qiskit_aer import AerSimulator
    from qiskit import transpile
    sim = AerSimulator(method="statevector")
    counts = sim.run(transpile(qc, sim), shots=20000, seed_simulator=3).result().get_counts()
    tot = sum(counts.values())
    return {k: v / tot for k, v in counts.items()}


def fidelity(counts, ideal) -> float:
    """Hellinger fidelity between measured counts and the ideal distribution (1 = perfect)."""
    tot = sum(counts.values())
    keys = set(counts) | set(ideal)
    bc = sum(np.sqrt(counts.get(k, 0) / tot * ideal.get(k, 0.0)) for k in keys)
    return float(bc ** 2)
