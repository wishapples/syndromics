"""Command line: `syndromics demo | diagnose | google | ibm | health | place`."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import stim

from . import __version__
from .diagnose import diagnose, score_against_truth
from .plot import plot_report


def _demo(a):
    from .sim import real_chip_circuit, nominal_circuit, sample
    real, truth = real_chip_circuit(a.d, a.d, a.p, seed=a.seed)
    nominal = nominal_circuit(a.d, a.d, a.p)
    det, _ = sample(real, a.shots, seed=3)
    print(f"Simulated distance-{a.d} chip, {a.shots:,} shots. Diagnosing without knowledge of the defects...")
    rep = diagnose(det, nominal, a.p)
    print(rep.summary())
    score = score_against_truth(rep, truth)
    print("\nScore against hidden truth:", json.dumps(score))
    if a.plot:
        plot_report(rep, nominal, a.plot, truth=truth,
                    title=f"{score['localised']}/{score['true_defects']} defects localised, "
                          f"{score['false_alarms']} false alarms")
        print("Chip map written to", a.plot)


def _diagnose(a):
    nominal = stim.Circuit.from_file(a.circuit)
    det = stim.read_shot_data_file(path=a.detections, format=a.format,
                                   num_detectors=nominal.num_detectors).astype(np.uint8)
    rep = diagnose(det, nominal, a.p, threshold=a.threshold)
    print(rep.summary())
    if a.json:
        with open(a.json, "w") as fh:
            json.dump([{"members": [list(map(str, m)) for m in f.members], "extra_rate": f.extra_rate,
                        "times_spec": f.times_spec, "flagged": f.flagged} for f in rep.findings], fh, indent=1)
    if a.plot:
        plot_report(rep, nominal, a.plot)
        print("Chip map written to", a.plot)


def _google(a):
    from .google import run
    run(a.root, out=a.out, target_rounds=a.rounds, limit=a.limit, crosstalk=not a.no_crosstalk)


def _ibm(a):
    import os
    from . import ibm
    if a.fake:
        from qiskit_ibm_runtime import fake_provider
        cls = "Fake" + a.backend.replace("ibm_", "").capitalize()
        backend = getattr(fake_provider, cls)()
        from qiskit_aer.primitives import SamplerV2 as AerSampler
        sampler = AerSampler.from_backend(backend)
        print(f"Simulating {a.backend} with its published noise model ({cls}).")
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        backend = QiskitRuntimeService().backend(a.backend)
        sampler = SamplerV2(mode=backend)
        print(f"Running on real hardware: {backend.name}. This uses QPU minutes from your plan.")
    L = 2 * a.n_data - 1
    line = [int(x) for x in a.line.split(",")] if a.line else ibm.find_line(backend, L)
    print("Qubits used (line order):", line)
    uniform = ibm.nominal_model(backend, line, "uniform")
    nominal = ibm.rep_code_stim(a.n_data, a.rounds, *uniform)
    os.makedirs(a.out, exist_ok=True)
    meas = ibm.run_sampler(backend, a.n_data, a.rounds, line, a.shots, sampler=sampler)
    np.save(os.path.join(a.out, "measurements.npy"), meas)
    nominal.to_file(os.path.join(a.out, "nominal_uniform.stim"))
    fit, det, _ = ibm.analyse(meas, nominal, sparsity=a.sparsity)
    print(f"\nDetection-event rate {det.mean():.4f}. Components worse than the uniform model:")
    print(fit.table(only_outliers=True))
    cmp = ibm.compare_with_calibration(fit, backend, line, uniform)
    print("\nSyndromics estimate vs IBM's published calibration:")
    for kind, name, cal, est, err in cmp["rows"]:
        print(f"  {kind:7s} {name:11s} IBM {cal:.4f}   Syndromics {est:.4f} +/- {err:.4f}")
    stats = {k: v for k, v in cmp.items() if k != "rows"}
    print("Agreement:", {k: round(v, 3) for k, v in stats.items()})
    with open(os.path.join(a.out, "comparison.json"), "w") as fh:
        json.dump({"backend": a.backend, "fake": a.fake, "line": line, "n_data": a.n_data,
                   "rounds": a.rounds, "shots": a.shots, "agreement": stats,
                   "rows": cmp["rows"]}, fh, indent=1)
    print(f"Saved raw measurements and results to {a.out}/")


def _backend(name, fake):
    if fake or name.startswith("fake_"):
        from qiskit_ibm_runtime import fake_provider
        cls = "Fake" + name.replace("fake_", "").replace("ibm_", "").capitalize()
        return getattr(fake_provider, cls)()
    from qiskit_ibm_runtime import QiskitRuntimeService
    return QiskitRuntimeService().backend(name)


def _drift_example(backend):
    """A fixed, documented drift for the offline demo (see examples/noise_aware_demo.py)."""
    from . import ibm
    cz = {(142, 143): 0.06}
    for p in ibm._gate_errors(backend):
        if 104 in p:
            cz[p] = 0.04
    return {"readout": {103: 0.15, 105: 0.09}, "sq": {104: 0.01}, "cz": cz}


def _health(a):
    from . import placement as pl
    backend = _backend(a.backend, a.fake)
    if a.fake:
        chip = pl.SimulatedChip(backend, _drift_example(backend) if a.drift else None)
        ex = chip.executor
        print(f"Simulated {backend.name}" + (" with example drift" if a.drift else " (no drift)"))
    else:
        ex = pl.hardware_executor(backend)
        print(f"Running diagnostic lines on {backend.name} (this uses machine time)...")
    lines = pl.cover_with_lines(backend)
    print(f"{len(lines)} diagnostic lines covering {len({q for l in lines for q in l})} qubits")
    h = pl.diagnose_chip(backend, ex, lines=lines, rounds=a.rounds, shots=a.shots)
    with open(a.out, "w") as fh:
        fh.write(h.to_json())
    print(f"\n{len(h.flagged)} change(s) since the published calibration:")
    for k, v in sorted(h.flagged.items()):
        print(f"  {k:10s} {v}")
    print(f"Chip health written to {a.out}")


def _place(a):
    from . import copilot, placement as pl
    from .plot import plot_placement
    recipe = copilot.generate(a.prompt, allow_llm=not a.no_llm)
    print(f"{recipe.name}  ({recipe.circuit.num_qubits} qubits)")
    print("  what it does:", recipe.explanation)
    print("  ideal result:", recipe.expected)
    for n in recipe.notes:
        print("  note:", n)
    backend = _backend(a.backend, a.fake)
    health = None
    if a.health:
        health = pl.ChipHealth.from_json(open(a.health).read())
    default = pl.place(recipe.circuit, backend, None)
    print(f"\nDefault placement (published calibration): qubits {default.qubits}")
    if health:
        ours = pl.place(recipe.circuit, backend, health)
        print(f"Syndromics placement (today's health):     qubits {ours.qubits}")
        if ours.avoided:
            hit = [k for k in health.flagged
                   if k in pl.place(recipe.circuit, backend, health).avoided and _touches(default, k)]
            if hit:
                print("  avoided problems the default placement would have used:", ", ".join(sorted(hit)))
        if ours.touched_flags:
            print("  could not avoid:", ", ".join(ours.touched_flags))
        est_def = pl.estimate_success(default.circuit, pl.corrected_target(backend, health))
        print(f"  estimated success with today's errors: default {est_def:.3f} -> syndromics "
              f"{ours.estimated_success:.3f}")
        chosen = ours
    else:
        print("  (no --health file: run `syndromics health` first to place around today's problems)")
        chosen = default
    fid = {}
    if a.fake and a.simulate:
        chip = pl.SimulatedChip(backend, _drift_example(backend) if a.drift else None)
        ideal = pl.ideal_distribution(recipe.circuit)
        fid["d"] = pl.fidelity(chip.run(default.circuit), ideal)
        msg = f"\nNoisy simulation of the true chip, fidelity to ideal: default {fid['d']:.3f}"
        if health:
            fid["s"] = pl.fidelity(chip.run(chosen.circuit), ideal)
            msg += f", syndromics {fid['s']:.3f}"
        print(msg)
    if a.qasm:
        from qiskit import qasm3
        with open(a.qasm, "w") as fh:
            fh.write(qasm3.dumps(chosen.circuit))
        print("Compiled circuit written to", a.qasm)
    if a.plot:
        tag = lambda k: f"\nfidelity {fid[k]:.2f}" if k in fid else ""
        panels = [("Default placement\n(published calibration)" + tag("d"), default.qubits,
                   pl.used_pairs(default.circuit), "#5b8def")]
        if health:
            panels.append(("Syndromics placement\n(today's measured health)" + tag("s"), chosen.qubits,
                           pl.used_pairs(chosen.circuit), "#1f9e6e"))
        plot_placement(backend, health, panels, a.plot, suptitle=f"{recipe.name} on {backend.name.replace('fake_', 'IBM ').title()}"
                       + (" (simulated)" if a.fake else ""))
        print("Chip map written to", a.plot)


def _touches(pl_result, key):
    if "-" in key:
        a, b = (int(x[1:]) for x in key.split("-"))
        from .placement import used_pairs
        return (a, b) in set(used_pairs(pl_result.circuit))
    return int(key[1:]) in set(pl_result.qubits)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="syndromics", description="Chip diagnostics from QEC syndrome data")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="diagnose a simulated chip with hidden defects")
    d.add_argument("--d", type=int, default=5, help="code distance")
    d.add_argument("--p", type=float, default=0.004, help="nominal physical error rate")
    d.add_argument("--shots", type=int, default=200_000)
    d.add_argument("--seed", type=int, default=7, help="which random chip")
    d.add_argument("--plot", default="chip_map.png")
    d.set_defaults(fn=_demo)

    g = sub.add_parser("diagnose", help="diagnose your own detection-event data")
    g.add_argument("--circuit", required=True, help="nominal Stim circuit (.stim)")
    g.add_argument("--detections", required=True, help="detection events file")
    g.add_argument("--format", default="01", choices=["01", "b8", "r8", "ptb64", "hits", "dets"])
    g.add_argument("--p", type=float, required=True, help="nominal physical error rate")
    g.add_argument("--threshold", type=float, default=0.5, help="flag if extra rate > threshold * p")
    g.add_argument("--json", help="write all findings to this JSON file")
    g.add_argument("--plot", help="write a chip map PNG")
    g.set_defaults(fn=_diagnose)

    w = sub.add_parser("google", help="analyse Google's public Willow surface-code dataset")
    w.add_argument("root", help="folder where the Zenodo zip was unpacked")
    w.add_argument("--out", default="willow_results")
    w.add_argument("--rounds", type=int, default=10, help="pick the experiment closest to this many rounds")
    w.add_argument("--limit", type=int, help="analyse only the first N experiments")
    w.add_argument("--no-crosstalk", action="store_true", help="skip crosstalk candidates (faster)")
    w.set_defaults(fn=_google)

    b = sub.add_parser("ibm", help="repetition-code diagnostics on IBM Quantum hardware")
    b.add_argument("--backend", default="ibm_kingston")
    b.add_argument("--fake", action="store_true", help="simulate the backend locally instead")
    b.add_argument("--n-data", type=int, default=5, help="data qubits (uses 2n-1 physical qubits)")
    b.add_argument("--rounds", type=int, default=10)
    b.add_argument("--shots", type=int, default=60_000)
    b.add_argument("--line", help="comma-separated physical qubits to use (default: best line found)")
    b.add_argument("--sparsity", type=float, default=0.5, help="0.5 recovers gate errors best on IBM")
    b.add_argument("--out", default="ibm_results")
    b.set_defaults(fn=_ibm)

    h = sub.add_parser("health", help="measure today's chip health (changes since calibration)")
    h.add_argument("--backend", default="ibm_kingston")
    h.add_argument("--fake", action="store_true", help="simulate the backend locally")
    h.add_argument("--drift", action="store_true", help="with --fake: add the documented example drift")
    h.add_argument("--rounds", type=int, default=6)
    h.add_argument("--shots", type=int, default=200_000)
    h.add_argument("--out", default="chip_health.json")
    h.set_defaults(fn=_health)

    pp = sub.add_parser("place", help="plain English -> circuit, placed on today's healthy qubits")
    pp.add_argument("prompt", help='e.g. "GHZ state on 12 qubits"')
    pp.add_argument("--backend", default="ibm_kingston")
    pp.add_argument("--fake", action="store_true")
    pp.add_argument("--health", help="chip_health.json from `syndromics health`")
    pp.add_argument("--simulate", action="store_true", help="with --fake: score by noisy simulation")
    pp.add_argument("--drift", action="store_true", help="with --fake --simulate: the true chip has the example drift")
    pp.add_argument("--qasm", help="write the compiled circuit (OpenQASM 3)")
    pp.add_argument("--plot", help="write a chip map PNG")
    pp.add_argument("--no-llm", action="store_true", help="never call a language model")
    pp.set_defaults(fn=_place)

    a = ap.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
