"""
Self-service demo for IBM Quantum users: run Syndromics on an IBM backend with your own account.

    syndromics-ibm-demo check                                   # environment + account check, no QPU use
    syndromics-ibm-demo rehearse                                # everything on a local simulator, no account
    syndromics-ibm-demo validate --backend ibm_kingston --yes   # A: recover today's calibration from syndromes
    syndromics-ibm-demo health   --backend ibm_kingston --yes   # B: what changed since calibration
    syndromics-ibm-demo ab       --backend ibm_kingston --yes   # C: default vs Syndromics placement
    syndromics-ibm-demo reanalyse                               # re-analyse saved validation data, no QPU
    syndromics-ibm-demo report                                  # one HTML page from everything in ./results

Nothing is sent anywhere except to IBM Quantum through your own Qiskit Runtime account. Results are
written to ./results so you can inspect them, and share them only if you choose. Hardware steps show an
estimate of the shots they will use and do nothing unless --yes is given.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import os
import platform
import sys
import time

import numpy as np

from . import __version__, adaptive, copilot, decodecheck, ibm, placement as pl

RESULTS = "results"


# ---------------------------------------------------------------- backend + sampler
def get_backend(a):
    """Returns (backend, sampler, label). --fake uses Qiskit's fake backend and a local noisy simulator,
    exercising exactly the same code path as hardware."""
    if a.fake:
        from qiskit_ibm_runtime import fake_provider
        from qiskit_aer.primitives import SamplerV2 as AerSampler
        name = a.backend.replace("ibm_", "")
        backend = getattr(fake_provider, "Fake" + name.capitalize())()
        return backend, AerSampler.from_backend(backend, seed=11), f"{backend.name} (local simulator with IBM's published noise)"
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    kw = {}
    if a.instance:
        kw["instance"] = a.instance
    try:
        service = QiskitRuntimeService(**kw)
        backend = service.backend(a.backend)
    except Exception as e:
        sys.exit(f"Could not open {a.backend} with your IBM Quantum account: {str(e).splitlines()[0]}\n"
                 "Run 'syndromics-ibm-demo check' for help, or add --fake to rehearse on a local simulator.")
    sampler = SamplerV2(mode=backend)
    return backend, sampler, f"{backend.name} (IBM Quantum hardware)"


def calibration_time(backend):
    for getter in (lambda: backend.properties().last_update_date, lambda: backend.target.properties_last_updated):
        try:
            t = getter()
            if t:
                return str(t)
        except Exception:
            pass
    return "unknown"


def confirm(a, what, circuits, shots_each, backend):
    total = circuits * shots_each
    rep = 250e-6
    try:
        rep = float(backend.configuration().default_rep_delay) or rep
    except Exception:
        pass
    est = total * (rep + 30e-6)
    print(f"\n{what}: {circuits} circuit(s) x {shots_each:,} shots = {total:,} shots on {backend.name}.")
    if not a.fake:
        print(f"Rough QPU-time estimate: {est/60:.1f} min (actual usage is shown in your IBM Quantum dashboard).")
        if not a.yes:
            print("Nothing submitted. Re-run with --yes to run it.")
            sys.exit(0)


def run_pubs(a, sampler, pubs, shots):
    """One job on hardware. In --fake mode each circuit runs separately: Qiskit Aer, when it batches
    several circuits in one run, can carry state from a mid-circuit-reset circuit into the next one
    (seen with qiskit-aer 0.18.2), which real hardware does not do."""
    if a.fake:
        res = [sampler.run([p], shots=shots).result()[0] for p in pubs]
        return res, "local"
    job = sampler.run(pubs, shots=shots)
    jid = getattr(job, "job_id", lambda: "local")()
    print(f"Submitted job {jid}; waiting for results (queue time depends on the backend)...")
    return job.result(), jid


def run_lines(backend, sampler, lines, rounds, shots, a=None):
    """All diagnostic lines in ONE job (one PUB per line). Returns {tuple(line): measurements}."""
    from qiskit import transpile
    pubs, keys = [], []
    for line in lines:
        n_data = (len(line) + 1) // 2
        qc = ibm.rep_code_qiskit(n_data, rounds)
        tqc = transpile(qc, backend, initial_layout=list(line), optimization_level=1, seed_transpiler=1)
        pubs.append(tqc); keys.append(tuple(line))
    if a is not None:
        res, jid = run_pubs(a, sampler, pubs, shots)
    else:
        job = sampler.run(pubs, shots=shots)
        jid = getattr(job, "job_id", lambda: "local")()
        print(f"Submitted job {jid}; waiting for results (queue time depends on the backend)...")
        res = job.result()
    out = {}
    for key, r in zip(keys, res):
        cols = [getattr(r.data, f"r{k}").to_bool_array(order="little") for k in range(rounds)]
        cols.append(r.data.final.to_bool_array(order="little"))
        out[key] = np.hstack(cols).astype(np.uint8)
    return out, jid


def save(name, obj):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)
    return path


def meta(a, backend, label):
    return {"syndromics_version": __version__, "backend": backend.name, "label": label, "fake": bool(a.fake),
            "run_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "calibration_last_updated": calibration_time(backend), "python": platform.python_version()}


# ---------------------------------------------------------------- steps
def step_check(a):
    print(f"Syndromics {__version__}, Python {platform.python_version()}")
    import qiskit, stim
    print(f"qiskit {qiskit.__version__}, stim {stim.__version__}")
    try:
        import qiskit_ibm_runtime
        print(f"qiskit-ibm-runtime {qiskit_ibm_runtime.__version__}")
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService(**({"instance": a.instance} if a.instance else {}))
        names = sorted(b.name for b in service.backends(operational=True, simulator=False))
        print("Account OK. Operational backends you can use:", ", ".join(names) or "(none)")
    except Exception as e:
        print("IBM account not reachable yet:", str(e).splitlines()[0])
        print("Save your account once with:  QiskitRuntimeService.save_account(token='...', instance='...')")
        print("You can still run:  syndromics-ibm-demo rehearse")


def step_rehearse(a):
    """Reproduces the proposal's simulation results locally (fast Stim-based simulator, planted drift)."""
    from qiskit_ibm_runtime.fake_provider import FakeKingston
    backend = FakeKingston()
    lines = pl.cover_with_lines(backend)
    cz = {(142, 143): 0.06}
    for p in ibm._gate_errors(backend):
        if 104 in p:
            cz[p] = 0.04
    drift = {"readout": {103: 0.15, 105: 0.09}, "sq": {104: 0.01}, "cz": cz}
    print("Rehearsal on simulated ibm_kingston with planted drift:", "readout q103=15%, q105=9%; defect on q104; bad CZ q142-q143")
    chip = pl.SimulatedChip(backend, drift)
    h = pl.diagnose_chip(backend, chip.executor, lines=lines, log=lambda s: None)
    print(f"Health check: {len(lines)} lines, {len({q for l in lines for q in l})} qubits; changes found:")
    for k, v in sorted(h.flagged.items()):
        print(f"  {k:10s} {v}")
    clean = pl.diagnose_chip(backend, pl.SimulatedChip(backend).executor, lines=lines, log=lambda s: None)
    print(f"Control (same chip, no drift): {len(clean.flagged)} false alarm(s)")
    rows = []
    for prompt in ("GHZ state on 12 qubits", "Grover search for 101", "QFT on 5 qubits"):
        r = copilot.generate(prompt, allow_llm=False)
        ideal = pl.ideal_distribution(r.circuit)
        d = pl.place(r.circuit, backend, None); s = pl.place(r.circuit, backend, h)
        fd = pl.fidelity(chip.run(d.circuit), ideal); fs = pl.fidelity(chip.run(s.circuit), ideal)
        rows.append({"circuit": r.name, "default": fd, "syndromics": fs})
        print(f"  {r.name:30s} default {fd:.3f}  syndromics {fs:.3f}")
    save("rehearsal.json", {"meta": {"syndromics_version": __version__, "backend": "fake_kingston (Stim simulator)",
                                     "run_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")},
                            "flagged": h.flagged, "control_false_alarms": len(clean.flagged), "placement": rows})
    print(f"Saved {RESULTS}/rehearsal.json")


def reference_circuits(nq, rounds):
    """Same-day reference measurements on the line's qubits:
    t0/t1: prepare all |0> / |1>, one terminal measurement (ordinary readout error);
    m0/m1: `rounds` times (prepare |0> / |1>, measure mid-circuit, reset): the measure+reset
           error that the repetition code's measurement qubits experience every round."""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    out = {}
    for name, flip in (("t0", False), ("t1", True)):
        q = QuantumRegister(nq, "q"); c = ClassicalRegister(nq, "t")
        qc = QuantumCircuit(q, c, name=name)
        if flip:
            qc.x(range(nq))
        qc.measure(range(nq), range(nq))
        out[name] = qc
    for name, flip in (("m0", False), ("m1", True)):
        q = QuantumRegister(nq, "q")
        regs = [ClassicalRegister(nq, f"m{r}") for r in range(rounds)]
        qc = QuantumCircuit(q, *regs, name=name)
        for r in range(rounds):
            if flip:
                qc.x(range(nq))
            qc.measure(range(nq), regs[r])
            qc.reset(range(nq))
        out[name] = qc
    return out


def reference_rates(raw, rounds):
    """Per-qubit terminal readout error and mid-circuit measure+reset error from the reference data."""
    term = (raw["t0"].mean(0) + (1 - raw["t1"]).mean(0)) / 2
    # skip round 0 (it follows the initial reset, not a mid-circuit one)
    m0 = raw["m0"][:, 1:, :].mean((0, 1)); m1 = (1 - raw["m1"][:, 1:, :]).mean((0, 1))
    return term, (m0 + m1) / 2


def analyse_validation(raw, cfg, sparsity):
    line, rounds, n_data = cfg["line"], cfg["rounds"], cfg["n_data"]
    ro_u = {int(k): v for k, v in cfg["uniform"]["readout"].items()}
    g2_u = {tuple(map(int, k.split("-"))): v for k, v in cfg["uniform"]["gate2"].items()}
    id_u = {int(k): v for k, v in cfg["uniform"]["idle"].items()}
    nominal = ibm.rep_code_stim(n_data, rounds, ro_u, g2_u, id_u)
    fit, det, _ = ibm.analyse(raw["rep"], nominal, sparsity=sparsity)
    term, mid = reference_rates(raw, rounds)
    rows = []
    for k, q in enumerate(line):
        f = fit.get(("spam", k))
        role = "data" if k % 2 == 0 else "measure"
        ref = term[k] if role == "data" else mid[k]
        rows.append({"qubit": f"q{q}", "role": role, "ibm_calibration": cfg["ibm_readout"][k],
                     "today_terminal": float(term[k]), "today_midcircuit": float(mid[k]),
                     "reference": float(ref), "syndromics": float(f.scale * ro_u[k]) if f else None,
                     "err": float(f.err * ro_u[k]) if f else None})
    gates = []
    for (a, b), pu in g2_u.items():
        f = fit.get(("gate", a, b))
        if f:
            gates.append({"pair": f"q{line[a]}-q{line[b]}", "ibm_calibration": cfg["ibm_gate"][f"{a}-{b}"],
                          "syndromics": float(f.scale * pu), "err": float(f.err * pu)})

    def r(xs, ys):
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        return float(np.corrcoef(xs, ys)[0, 1]) if len(xs) >= 3 and xs.std() > 0 and ys.std() > 0 else float("nan")
    ok = [x for x in rows if x["syndromics"] is not None]
    stats = {
        "syndromics_vs_today_reference": r([x["reference"] for x in ok], [x["syndromics"] for x in ok]),
        "syndromics_vs_ibm_calibration": r([x["ibm_calibration"] for x in ok], [x["syndromics"] for x in ok]),
        "ibm_calibration_vs_today_terminal": r([x["ibm_calibration"] for x in rows], [x["today_terminal"] for x in rows]),
        "gate_syndromics_vs_ibm": r([g["ibm_calibration"] for g in gates], [g["syndromics"] for g in gates]),
        "detection_rate": float(det.mean()), "sparsity": sparsity}
    return rows, gates, stats


def print_validation(rows, gates, stats):
    print("\nReadout / measurement error per qubit (today's reference = terminal readout for data qubits,")
    print("mid-circuit measure+reset for measurement qubits; both measured in the same job):")
    print(f"  {'qubit':6s} {'role':8s} {'IBM calib':>9s} {'today ref':>9s} {'Syndromics':>18s}   (today terminal / mid-circuit)")
    for x in rows:
        sy = f"{x['syndromics']:.4f} +/- {x['err']:.4f}" if x["syndromics"] is not None else "n/a"
        print(f"  {x['qubit']:6s} {x['role']:8s} {x['ibm_calibration']:9.4f} {x['reference']:9.4f} {sy:>18s}"
              f"   ({x['today_terminal']:.4f} / {x['today_midcircuit']:.4f})")
    print("\nGate errors (a repetition code sees only the bit-flip part of a CZ error, so this is rough):")
    for g in gates:
        print(f"  {g['pair']:11s} IBM {g['ibm_calibration']:.4f}   Syndromics {g['syndromics']:.4f} +/- {g['err']:.4f}")
    print("\nAgreement (Pearson r):")
    for k in ("syndromics_vs_today_reference", "syndromics_vs_ibm_calibration", "ibm_calibration_vs_today_terminal", "gate_syndromics_vs_ibm"):
        print(f"  {k:36s} {stats[k]:.3f}")
    print(f"  detection-event rate {stats['detection_rate']:.4f}; fit sparsity {stats['sparsity']}")


def _models(cfg):
    def unpack(m):
        return ({int(k): v for k, v in m["readout"].items()},
                {tuple(map(int, k.split("-"))): v for k, v in m["gate2"].items()},
                {int(k): v for k, v in m["idle"].items()})
    uni = unpack(cfg["uniform"])
    if "calibrated" in cfg:
        cal = unpack(cfg["calibrated"])
    else:  # results saved by v0.4.2 or earlier: calibrated readout and gates, median idle
        cal = ({k: v for k, v in enumerate(cfg["ibm_readout"])},
               {tuple(map(int, k.split("-"))): v for k, v in cfg["ibm_gate"].items()}, uni[2])
    return uni, cal


def report_validation(raw, cfg, sparsity, out_name):
    """Headline: held-out decoding comparison. Then: components worse than calibration. Then: per-qubit detail."""
    uni, cal = _models(cfg)
    print("\n=== 1. Does the measured noise model decode better? ===")
    cmp_ = decodecheck.compare(raw["rep"], cfg["n_data"], cfg["rounds"], uni, cal, sparsity=sparsity)
    print(decodecheck.format_comparison(cmp_))
    print("\n=== 2. Components worse than the calibration says (fit on all shots) ===")
    ch = decodecheck.changed_components(raw["rep"], cfg["n_data"], cfg["rounds"], uni, cal, cfg["line"], sparsity=sparsity)
    for r in ch:
        print(f"  {r['component']:34s} x{r['ratio']:.1f} +/- {r['err']:.1f}   (calibration {r['calibration']:.4f}, measured {r['measured']:.4f})")
    if not ch:
        print("  none: the calibration describes this line well today")
    else:
        print("  A repetition code localises a fault well (which qubit or pair) but splits it only roughly between a")
        print("  qubit's measurement and its neighbouring gate; treat large ratios as 'look here', not exact values.")
    print("\n=== 3. Shots where a measure qubit behaves randomly (leakage-like bursts) ===")
    cal_c = ibm.rep_code_stim(cfg["n_data"], cfg["rounds"], *cal)
    Dd, _ = decodecheck._detectors(cal_c, raw["rep"])
    bs = adaptive.burst_stats(cal_c, Dd)
    print(f"  shots where one measure qubit fires >= {bs['k']} detection events: real {bs['real']*100:.2f}%, "
          f"noise model predicts {bs['model']*100:.3f}% ({bs['real']/max(bs['model'],1e-6):.0f}x)")
    print("  per measure qubit: " + ", ".join(f"q{cfg['line'][a]} {r*100:.2f}%" for a, r in zip(bs["anc"], bs["per_qubit_real"])))
    print("  Leakage-aware decoding (section 1) handles these shots one by one from their own syndromes.")
    print("\n=== 4. Drift during the job (6 time windows, fresh fit per window) ===")
    W = len(Dd) // 6
    table, alerts = adaptive.track_drift(cal_c, [Dd[i*W:(i+1)*W] for i in range(6)], cfg["line"])
    tops = [r["component"] for r in ch[:4]]
    for r in ch[:4]:
        c = next(c for c in table[0] if adaptive._name(c, cfg["line"]) == r["component"])
        print(f"  {r['component']:34s} " + " ".join(f"{row[c]:5.1f}" for row in table) + "   (x calibration, per window)")
    if alerts:
        for a in alerts:
            print(f"  ALERT window {a['window']} (first seen {a['first_seen']}): {a['component']} {a['before']:.1f}x -> {a['now']:.1f}x calibration")
    else:
        print("  no confirmed drift alerts: the components were stable over this job")
    drift = {"alerts": alerts, "windows": [{adaptive._name(c, cfg["line"]): v for c, v in row.items()} for row in table]}
    print("\n=== 5. Detail: per-qubit measurement error (for reference) ===")
    print("  Syndromics measures the effective per-round error of each measurement (measure + reset + gate errors),")
    print("  which is larger than the end-of-circuit readout IBM publishes; the 'today' columns were measured in the same job.")
    rows, gates, stats = analyse_validation(raw, cfg, sparsity)
    print_validation(rows, gates, stats)
    return save(out_name, {"meta": cfg["meta"], "job_id": cfg["job_id"], "line": cfg["line"], "rounds": cfg["rounds"],
                           "shots": cfg["shots"], "decoding": cmp_, "changed_components": ch, "bursts": bs, "drift": drift,
                           "agreement": stats, "rows": rows, "gates": gates})


def step_validate(a):
    """A: from syndromes alone, recover each qubit's measurement error; check against same-day reference
    measurements taken in the same job, and against IBM's published calibration."""
    from qiskit import transpile
    backend, sampler, label = get_backend(a)
    L = 2 * a.n_data - 1
    if getattr(a, "line", None):
        line = [int(q) for q in a.line.replace(" ", "").split(",")]
        if len(line) != L:
            sys.exit(f"--line needs {L} qubits for --n-data {a.n_data}, got {len(line)}")
        edges = {tuple(sorted(e)) for e in backend.coupling_map.get_edges()}
        bad = [(x, y) for x, y in zip(line, line[1:]) if tuple(sorted((x, y))) not in edges]
        if bad:
            sys.exit(f"--line is not a connected path on {backend.name}: no coupler for {bad}")
    else:
        line = ibm.find_line(backend, L)
    print(f"Backend: {label}\nQubits (line order): {line}\nCalibration last updated: {calibration_time(backend)}")
    refs = reference_circuits(L, a.ref_rounds)
    # IBM's sampler requires the same shot count for every circuit in a job: all 5 use a.shots
    confirm(a, "Validation (1 repetition code + 4 reference circuits, one job)", 1 + len(refs), a.shots, backend)
    rep = transpile(ibm.rep_code_qiskit(a.n_data, a.rounds), backend, initial_layout=line, optimization_level=1, seed_transpiler=1)
    pubs = [rep] + [transpile(c, backend, initial_layout=line, optimization_level=1, seed_transpiler=1) for c in refs.values()]
    res, jid = run_pubs(a, sampler, pubs, a.shots)
    d = res[0].data
    raw = {"rep": np.hstack([getattr(d, f"r{k}").to_bool_array(order="little") for k in range(a.rounds)]
                            + [d.final.to_bool_array(order="little")]).astype(np.uint8)}
    raw["t0"] = res[1].data.t.to_bool_array(order="little").astype(np.uint8)
    raw["t1"] = res[2].data.t.to_bool_array(order="little").astype(np.uint8)
    for i, nm in ((3, "m0"), (4, "m1")):
        raw[nm] = np.stack([getattr(res[i].data, f"m{r}").to_bool_array(order="little") for r in range(a.ref_rounds)], axis=1).astype(np.uint8)
    uniform = ibm.nominal_model(backend, line, "uniform")
    cal = ibm.nominal_model(backend, line, "calibrated")
    cfg = {"meta": meta(a, backend, label), "job_id": jid, "line": line, "n_data": a.n_data, "rounds": a.rounds,
           "ref_rounds": a.ref_rounds, "shots": a.shots, "ref_shots": a.shots,
           "uniform": {"readout": {str(k): v for k, v in uniform[0].items()},
                       "gate2": {f"{x}-{y}": v for (x, y), v in uniform[1].items()},
                       "idle": {str(k): v for k, v in uniform[2].items()}},
           "ibm_readout": [cal[0][k] for k in range(L)],
           "ibm_gate": {f"{x}-{y}": v for (x, y), v in cal[1].items()},
           "calibrated": {"readout": {str(k): v for k, v in cal[0].items()},
                          "gate2": {f"{x}-{y}": v for (x, y), v in cal[1].items()},
                          "idle": {str(k): v for k, v in cal[2].items()}}}
    os.makedirs(RESULTS, exist_ok=True)
    np.savez_compressed(os.path.join(RESULTS, "A_raw.npz"), **raw)
    save("A_config.json", cfg)
    p = report_validation(raw, cfg, a.sparsity, "A_validate.json")
    print(f"\nSaved {p}, raw data in {RESULTS}/A_raw.npz (re-analyse any time with: syndromics-ibm-demo reanalyse)")


def step_reanalyse(a):
    """Re-run the validation analysis on saved raw data: no account, no QPU time."""
    raw = dict(np.load(os.path.join(RESULTS, "A_raw.npz")))
    cfg = json.load(open(os.path.join(RESULTS, "A_config.json")))
    print(f"Re-analysing job {cfg['job_id']} on {cfg['meta']['backend']} ({cfg['meta']['run_at_utc']})")
    report_validation(raw, cfg, a.sparsity, "A_validate.json")


def step_compare_runs(a):
    """Prospective test, no QPU: freeze the model and adaptive settings learned on an earlier run, decode this run."""
    prev = a.previous
    if os.path.isdir(os.path.join(prev, RESULTS)):
        prev = os.path.join(prev, RESULTS)
    p_raw = dict(np.load(os.path.join(prev, "A_raw.npz"))); p_cfg = json.load(open(os.path.join(prev, "A_config.json")))
    n_raw = dict(np.load(os.path.join(RESULTS, "A_raw.npz"))); n_cfg = json.load(open(os.path.join(RESULTS, "A_config.json")))
    print(f"Run 1: job {p_cfg['job_id']} ({p_cfg['meta']['run_at_utc']})\nRun 2: job {n_cfg['job_id']} ({n_cfg['meta']['run_at_utc']})")
    if any(p_cfg[k] != n_cfg[k] for k in ("line", "n_data", "rounds")):
        return _report_transfer(p_raw, p_cfg, n_raw, n_cfg, a)
    try:
        r = adaptive.compare_runs(p_raw["rep"], p_cfg, n_raw["rep"], n_cfg, _models, sparsity=a.sparsity)
    except ValueError as e:
        sys.exit(f"Cannot compare: {e}")
    n = r["shots_run2"]
    print(f"\n=== Decoding run 2 ({n:,} shots) with everything frozen from run 1 ===")
    rows = [("IBM calibration at run 2", "run2_calibration"), ("IBM calibration at run 1", "run1_calibration"),
            ("Syndromics model from run 1 (frozen)", "frozen_syndromics"),
            (f"Syndromics + adaptive from run 1 (frozen: k={r['frozen_settings']['k']}, boost={r['frozen_settings']['boost']})", "frozen_syndromics_la")]
    base = r["run2_calibration"]["rate"]
    for lab, k in rows:
        v = r[k]
        print(f"  {lab:62s} {v['errors']:4d} errors = {v['rate']*100:.3f}%   {'baseline' if k == 'run2_calibration' else format((v['rate']/base-1)*100, '+.0f') + '%'}")
    print("\n=== Components that changed between the runs ===")
    for c in r["component_changes"]:
        print(f"  {c['component']:34s} x{c['run1']:.1f} -> x{c['run2']:.1f} calibration (z = {c['z']:+.1f})")
    if not r["component_changes"]:
        print("  none: the line behaved the same in both runs")
    b = r["bursts_run2"]
    print(f"\n=== Erratic-measure-qubit shots in run 2: {b['real']*100:.2f}% (model predicts {b['model']*100:.3f}%) ===")
    print("\nFor the run-2-only held-out test, run: syndromics-ibm-demo reanalyse")
    save("B_compare_runs.json", {"run1": p_cfg["job_id"], "run2": n_cfg["job_id"], **r})


def _report_transfer(p_raw, p_cfg, n_raw, n_cfg, a):
    """Run 2 is a different circuit (more rounds, or a shorter code on part of run 1's line): transfer test."""
    try:
        r = adaptive.transfer_runs(p_raw["rep"], p_cfg, n_raw["rep"], n_cfg, _models, sparsity=a.sparsity)
    except ValueError as e:
        sys.exit(f"Cannot compare: {e}")
    r1, r2 = r["run1"], r["run2"]
    print(f"\nTransfer test: model learned on run 1 (distance {r1['n_data']}, {r1['rounds']} rounds, {len(r1['line'])} qubits),\n"
          f"used on a different circuit in run 2 (distance {r2['n_data']}, {r2['rounds']} rounds, qubits {r2['line']}).")
    print(f"\n=== Logical error on {r['test_shots']:,} held-out run-2 shots (same shots for every row) ===")
    base = r["run2_calibration"]["rate"]
    def row(lab, v):
        rel = "baseline" if v is r["run2_calibration"] else format((v['rate'] / base - 1) * 100, '+.0f') + '%'
        print(f"  {lab:66s} {v['errors']:4d} errors = {v['rate']*100:.3f}%   {rel}")
    print("  Needs no run-2 data:")
    row("IBM calibration at run 2", r["run2_calibration"])
    row("Syndromics model from run 1 (frozen, transferred)", r["frozen_syndromics"])
    fs = r["frozen_settings"]
    row(f"Syndromics + adaptive from run 1 (frozen: k={fs['k']}, boost={fs['boost']})", r["frozen_syndromics_la"])
    print("  Learned on run 2's own training half (needs data from the new circuit):")
    for v in r["reweighting_on_run2"]:
        row(f"Correlation re-weighting, {v['train_shots']:,} run-2 shots", v)
    row(f"Re-weighting + adaptive, {r['reweighting_la_on_run2']['train_shots']:,} run-2 shots", r["reweighting_la_on_run2"])
    row(f"Syndromics refitted on {r['syndromics_on_run2']['train_shots']:,} run-2 shots (reference)", r["syndromics_on_run2"])
    print("\nRead: if the transferred model is close to the models learned on run 2's own data, the physical model carries"
          "\nover to a new circuit with no new training data. Small counts: check the error numbers, not only the rates.")
    save("B_transfer.json", {"run1": p_cfg["job_id"], "run2": n_cfg["job_id"], **r})


def step_health(a):
    """B: chip-wide check of what has changed since IBM's last calibration."""
    backend, sampler, label = get_backend(a)
    lines = pl.cover_with_lines(backend, length=a.line_length, min_length=min(7, a.line_length))
    if a.max_lines:
        lines = lines[:a.max_lines]
    nq = len({q for l in lines for q in l})
    print(f"Backend: {label}\n{len(lines)} diagnostic lines covering {nq} of {backend.num_qubits} qubits")
    print(f"Calibration last updated: {calibration_time(backend)}")
    confirm(a, "Health check", len(lines), a.shots, backend)
    out, jid = run_lines(backend, sampler, lines, a.rounds, a.shots, a)
    h = pl.diagnose_chip(backend, lambda line, n, r, s: out[tuple(line)], lines=lines, rounds=a.rounds, shots=a.shots)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "chip_health.json"), "w") as fh:
        fh.write(h.to_json())
    print(f"\n{len(h.flagged)} change(s) since calibration:")
    for k, v in sorted(h.flagged.items()):
        print(f"  {k:10s} {v}")
    if not h.flagged:
        print("  none: today's chip matches its calibration on the measured qubits (that is a valid result).")
    p = save("B_health.json", {"meta": meta(a, backend, label), "job_id": jid, "lines": lines, "rounds": a.rounds,
                               "shots_per_line": a.shots, "qubits_covered": nq, "flagged": h.flagged,
                               "known_bad_ignored": len(h.known_bad)})
    print(f"Saved {p} and {RESULTS}/chip_health.json")


def step_ab(a):
    """C: same circuits, default placement vs Syndromics placement, interleaved in one job."""
    backend, sampler, label = get_backend(a)
    hp = os.path.join(RESULTS, "chip_health.json")
    if not os.path.exists(hp):
        sys.exit("Run the health step first (it writes results/chip_health.json).")
    h = pl.ChipHealth.from_json(open(hp).read())
    if h.backend != backend.name:
        print(f"Warning: chip_health.json was measured on {h.backend}, not {backend.name}.")
    prompts = [p.strip() for p in a.circuits.split(";") if p.strip()]
    pubs, info = [], []
    for prompt in prompts:
        r = copilot.generate(prompt, allow_llm=False)
        d = pl.place(r.circuit, backend, None); s = pl.place(r.circuit, backend, h)
        same = d.qubits == s.qubits
        info.append({"circuit": r.name, "default_qubits": d.qubits, "syndromics_qubits": s.qubits,
                     "same_layout": same, "avoided": s.avoided, "ideal": pl.ideal_distribution(r.circuit)})
        pubs += [d.circuit, s.circuit]
        print(f"{r.name}: default {d.qubits}\n{'':{len(r.name)}}  syndromics {s.qubits}{'  (identical: nothing to avoid)' if same else ''}")
    confirm(a, "A/B comparison", len(pubs), a.shots, backend)
    res, jid = run_pubs(a, sampler, pubs, a.shots)
    rows = []
    for i, it in enumerate(info):
        cd = res[2 * i].data.c.get_counts(); cs = res[2 * i + 1].data.c.get_counts()
        fd, fs = pl.fidelity(cd, it["ideal"]), pl.fidelity(cs, it["ideal"])
        rows.append({k: v for k, v in it.items() if k != "ideal"} | {"default_fidelity": fd, "syndromics_fidelity": fs})
        print(f"  {it['circuit']:32s} default {fd:.3f}   syndromics {fs:.3f}")
    p = save("C_ab.json", {"meta": meta(a, backend, label), "job_id": jid, "shots": a.shots, "rows": rows,
                           "health_measured_at": h.created})
    print(f"Saved {p}")


# ---------------------------------------------------------------- report
def step_report(a):
    def load(n):
        p = os.path.join(RESULTS, n)
        return json.load(open(p)) if os.path.exists(p) else None
    A, B, C, R = load("A_validate.json"), load("B_health.json"), load("C_ab.json"), load("rehearsal.json")
    e = html.escape
    parts = [f"<h1>Syndromics demo results</h1><p class=s>Generated {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} "
             f"&middot; Syndromics {e(__version__)} &middot; files in <code>{RESULTS}/</code></p>"]
    if A:
        m = A["meta"]
        ag = A["agreement"]
        dec, ch = A.get("decoding"), A.get("changed_components", [])
        if dec:
            from .decodecheck import LABELS, MODELS
            mm = dec["mean"]
            wins = sum(t["syndromics"] < t["calibration"] for t in dec["tests"])
            parts.append(f"<h2>A. Held-out decoding test</h2><p>{e(m['label'])}; IBM calibration updated {e(str(m['calibration_last_updated']))}; "
                         f"job <code>{e(str(A['job_id']))}</code>; line {A['line']}, {A['rounds']} rounds, {A['shots']:,} shots. "
                         f"Each of {len(dec['tests'])} tests fits the noise model on half the shots and decodes the other half (PyMatching).</p>"
                         "<table><tr><th>Noise model given to the decoder</th><th>Logical error (mean)</th><th>vs calibration</th></tr>" +
                         "".join(f"<tr><td>{'<b>' if k == 'syndromics' else ''}{e(LABELS[k])}{'</b>' if k == 'syndromics' else ''}</td><td>{mm[k]*100:.3f}%</td>"
                                 f"<td>{'baseline' if k == 'calibration' else format((mm[k]/mm['calibration']-1)*100, '+.0f') + '%'}</td></tr>" for k in MODELS if k in mm) +
                         f"</table><p>Syndromics beat the calibration in <b>{wins} of {len(dec['tests'])}</b> tests.</p>")
            parts.append("<h3>Components worse than calibration</h3>" + ("<table><tr><th>Component</th><th>Ratio</th><th>Calibration</th><th>Measured</th></tr>" +
                         "".join(f"<tr><td>{e(r['component'])}</td><td>&times;{r['ratio']:.1f}</td><td>{r['calibration']:.4f}</td><td>{r['measured']:.4f}</td></tr>" for r in ch) +
                         "</table><p class=s>A repetition code localises a fault well but splits it only roughly between a measurement and its neighbouring gate.</p>"
                         if ch else "<p>None: the calibration describes this line well today.</p>"))
        parts.append(f"<h3>Detail: measurement errors vs same-day references</h3><p>{e(m['label'])}; IBM calibration updated "
                     f"{e(str(m['calibration_last_updated']))}; job <code>{e(str(A['job_id']))}</code>; line {A['line']}, "
                     f"{A['rounds']} rounds, {A['shots']:,} shots.</p>"
                     f"<p><b>Syndromics vs today's reference:</b> r = {ag['syndromics_vs_today_reference']:.3f} &middot; "
                     f"Syndromics vs IBM calibration: r = {ag['syndromics_vs_ibm_calibration']:.3f} &middot; "
                     f"IBM calibration vs today's terminal readout: r = {ag['ibm_calibration_vs_today_terminal']:.3f}</p>"
                     "<table><tr><th>Qubit</th><th>Role</th><th>IBM calibration</th><th>Today's reference</th><th>Syndromics (syndromes only)</th></tr>" +
                     "".join(f"<tr><td>{e(r['qubit'])}</td><td>{e(r['role'])}</td><td>{r['ibm_calibration']:.4f}</td><td>{r['reference']:.4f}</td>"
                             f"<td>{(format(r['syndromics'], '.4f') + ' &plusmn; ' + format(r['err'], '.4f')) if r['syndromics'] is not None else 'n/a'}</td></tr>" for r in A["rows"]) +
                     "</table><p class=s>Today's reference: terminal readout for data qubits, mid-circuit measure+reset for measurement qubits, measured in the same job.</p>")
    if B:
        m = B["meta"]
        fl = B["flagged"]
        parts.append(f"<h2>B. Changes since calibration</h2><p>{e(m['label'])}; calibration updated {e(str(m['calibration_last_updated']))}; "
                     f"job <code>{e(str(B['job_id']))}</code>; {len(B['lines'])} lines, {B['qubits_covered']} qubits, "
                     f"{B['shots_per_line']:,} shots per line.</p>" +
                     ("<table><tr><th>Component</th><th>Finding</th></tr>" + "".join(f"<tr><td>{e(k)}</td><td>{e(v)}</td></tr>" for k, v in sorted(fl.items())) + "</table>"
                      if fl else "<p>No changes found: the measured qubits match their calibration.</p>"))
    if C:
        parts.append(f"<h2>C. Default vs Syndromics placement</h2><p>job <code>{e(str(C['job_id']))}</code>; {C['shots']:,} shots per circuit; "
                     "fidelity = Hellinger fidelity to the ideal output (1 = perfect).</p><table><tr><th>Circuit</th><th>Default</th><th>Syndromics</th><th>Layout</th></tr>" +
                     "".join(f"<tr><td>{e(r['circuit'])}</td><td>{r['default_fidelity']:.3f}</td><td>{r['syndromics_fidelity']:.3f}</td>"
                             f"<td>{'identical' if r['same_layout'] else 'avoided ' + e(', '.join(r['avoided']) or 'flagged items')}</td></tr>" for r in C["rows"]) + "</table>")
    if R:
        parts.append("<h2>Local rehearsal (simulated ibm_kingston, planted drift)</h2><table><tr><th>Circuit</th><th>Default</th><th>Syndromics</th></tr>" +
                     "".join(f"<tr><td>{e(r['circuit'])}</td><td>{r['default']:.3f}</td><td>{r['syndromics']:.3f}</td></tr>" for r in R["placement"]) +
                     f"</table><p>Changes found: {len(R['flagged'])}; false alarms on the drift-free control: {R['control_false_alarms']}.</p>")
    if len(parts) == 1:
        parts.append("<p>No results yet. Run a step first.</p>")
    css = ("body{font-family:system-ui,Arial,sans-serif;max-width:900px;margin:30px auto;padding:0 16px;color:#14202e}"
           "h1{margin-bottom:0}.s{color:#5a6675}h2{color:#0f5f8c;border-bottom:2px solid #0f5f8c;padding-bottom:3px;margin-top:28px}"
           "table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #d9dee5;padding:5px 8px;text-align:left}"
           "th{background:#f3f5f8}code{background:#f3f5f8;padding:0 4px}")
    os.makedirs(RESULTS, exist_ok=True)
    p = os.path.join(RESULTS, "report.html")
    with open(p, "w") as fh:
        fh.write(f"<!DOCTYPE html><html><head><meta charset=utf-8><title>Syndromics demo results</title><style>{css}</style></head><body>{''.join(parts)}</body></html>")
    print(f"Wrote {p}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="syndromics-ibm-demo", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    for name, fn, shots, extra in (("check", step_check, 0, False), ("rehearse", step_rehearse, 0, False),
                                   ("reanalyse", step_reanalyse, 0, False),
                                   ("compare-runs", step_compare_runs, 0, False),
                                   ("validate", step_validate, 30000, True), ("health", step_health, 50000, True),
                                   ("ab", step_ab, 4000, True), ("report", step_report, 0, False)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--instance", help="IBM Quantum instance (CRN), if not the saved default")
        if extra:
            p.add_argument("--backend", default="ibm_kingston")
            p.add_argument("--fake", action="store_true", help="local simulator of the backend (no account, no QPU use)")
            p.add_argument("--yes", action="store_true", help="actually submit to hardware")
            p.add_argument("--shots", type=int, default=shots)
            p.add_argument("--rounds", type=int, default=10 if name == "validate" else 6)
        if name == "validate":
            p.add_argument("--line", help="comma-separated qubits to use, e.g. 117,105,106,107,108,109,118,129,128 (default: best line)")
            p.add_argument("--n-data", type=int, default=5)
            p.add_argument("--ref-rounds", type=int, default=6)
        if name == "compare-runs":
            p.add_argument("--previous", required=True, help="folder of the earlier run (containing results/ or the results folder itself)")
        if name in ("validate", "reanalyse", "compare-runs"):
            p.add_argument("--sparsity", type=float, default=0.5,
                           help="0.5 = tuned for recovering IBM calibration; 0 = every component free")
        if name == "health":
            p.add_argument("--max-lines", type=int, default=0, help="only the first N lines (quick test)")
            p.add_argument("--line-length", type=int, default=19, help="qubits per diagnostic line (odd)")
        if name == "ab":
            p.add_argument("--circuits", default="GHZ state on 12 qubits; Grover search for 101; QFT on 5 qubits")
    a = ap.parse_args(argv)
    t = time.time()
    a.fn(a)
    print(f"({time.time()-t:.0f}s)")


if __name__ == "__main__":
    main()
