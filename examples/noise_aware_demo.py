"""
Does placing circuits with Syndromics' measured chip health beat the compiler's default?

Simulated IBM Kingston (156 qubits, published calibration from Qiskit's fake backend) plus
hidden drift: faults that appeared after the last calibration. Three placements per circuit:

  default    Qiskit optimization_level=3 using the published calibration (what everyone does)
  syndromics same compiler, but the Target carries today's error rates measured by Syndromics
  oracle     same compiler with the true error rates (the best any placement could do)

Each compiled circuit is run in a noisy simulation of the *true* chip and scored by Hellinger
fidelity against the ideal output (1.0 = perfect).

    python examples/noise_aware_demo.py [--trials 10] [--out docs/placement]
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from qiskit_ibm_runtime.fake_provider import FakeKingston

from syndromics import copilot, ibm, placement as pl

PROMPTS = ["GHZ state on 12 qubits", "Bernstein-Vazirani with secret 101101",
           "QFT on 5 qubits", "Grover search for 101", "teleport a qubit",
           "MaxCut on 6 nodes"]


def three_way(backend, chip, health, recipe, shots=4000):
    ideal = pl.ideal_distribution(recipe.circuit)
    truth = chip.truth_target()
    out = {}
    for name, target in (("default", backend.target),
                         ("syndromics", pl.corrected_target(backend, health) if health else None),
                         ("oracle", truth)):
        if target is None:
            continue
        tqc, est = pl._transpile_best(recipe.circuit, target, range(6), target)
        f = pl.fidelity(chip.run(tqc, shots=shots), ideal)
        out[name] = {"fidelity": f, "qubits": pl.used_qubits(tqc)}
    return out


def drift_on_default_choice(backend, rng):
    """Faults on the qubits the default compiler likes most: the case that matters."""
    from collections import Counter
    cnt, pair_cnt = Counter(), Counter()
    for p in PROMPTS:
        r = copilot.generate(p, allow_llm=False)
        tqc, _ = pl._transpile_best(r.circuit, backend.target, range(6), backend.target)
        cnt.update(pl.used_qubits(tqc))
        for inst in tqc.data:
            if len(inst.qubits) == 2:
                pair_cnt[tuple(sorted(tqc.find_bit(q).index for q in inst.qubits))] += 1
    common = [q for q, _ in cnt.most_common(12)]
    pairs = [p for p, _ in pair_cnt.most_common(12)]
    q_ro, q_tls = common[0], common[1]
    pr = next(p for p in pairs if q_ro not in p and q_tls not in p)
    cz = {pr: 0.06}
    for p in ibm._gate_errors(backend):
        if q_tls in p:
            cz[p] = 0.04
    return {"readout": {q_ro: 0.15, common[2]: 0.09}, "sq": {q_tls: 0.01}, "cz": cz}, \
        f"readout q{q_ro}=15%, q{common[2]}=9%; TLS defect on q{q_tls} (its gates 4%); bad gate q{pr[0]}-q{pr[1]} 6%"


def random_drift(backend, rng, n_faults=6):
    qs = list(range(backend.num_qubits)); pairs = list(ibm._gate_errors(backend))
    d = {"readout": {}, "sq": {}, "cz": {}}
    for _ in range(n_faults):
        kind = rng.choice(["readout", "tls", "cz"])
        if kind == "readout":
            d["readout"][int(rng.choice(qs))] = float(rng.uniform(0.06, 0.2))
        elif kind == "tls":
            q = int(rng.choice(qs)); d["sq"][q] = 0.01
            for p in pairs:
                if q in p:
                    d["cz"][p] = float(rng.uniform(0.03, 0.08))
        else:
            d["cz"][tuple(pairs[int(rng.integers(len(pairs)))])] = float(rng.uniform(0.03, 0.1))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--out", default="docs/placement")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    backend = FakeKingston()
    rng = np.random.default_rng(2026)
    lines = pl.cover_with_lines(backend)
    covered = sorted({q for l in lines for q in l})
    print(f"diagnostic lines: {len(lines)}, covering {len(covered)}/{backend.num_qubits} qubits")

    # ---- scenario A
    drift, desc = drift_on_default_choice(backend, rng)
    print("\nScenario A: drift on the qubits the default compiler prefers")
    print("  hidden faults:", desc)
    chip = pl.SimulatedChip(backend, drift)
    t = time.time()
    health = pl.diagnose_chip(backend, chip.executor, lines=lines, log=lambda s: None)
    print(f"  Syndromics diagnosis: {time.time()-t:.1f}s compute, found {len(health.flagged)} change(s) since calibration ({len(health.known_bad)} known-bad items ignored):")
    for k, v in sorted(health.flagged.items()):
        print(f"    {k:10s} {v}")
    with open(os.path.join(a.out, "chip_health_scenarioA.json"), "w") as f:
        f.write(health.to_json())
    rowsA = []
    print(f"\n  {'circuit':38s} {'default':>8s} {'syndromics':>11s} {'oracle':>8s}")
    for p in PROMPTS:
        r = copilot.generate(p, allow_llm=False)
        res = three_way(backend, chip, health, r)
        rowsA.append({"circuit": r.name, **{k: v["fidelity"] for k, v in res.items()}})
        print(f"  {r.name:38s} {res['default']['fidelity']:8.3f} {res['syndromics']['fidelity']:11.3f} "
              f"{res['oracle']['fidelity']:8.3f}")

    # ---- scenario B
    print(f"\nScenario B: {a.trials} random drift patterns (6 faults anywhere on the chip), GHZ-12")
    ghz = copilot.generate("GHZ state on 12 qubits", allow_llm=False)
    rowsB = []
    for i in range(a.trials):
        d = random_drift(backend, rng)
        chip = pl.SimulatedChip(backend, d)
        h = pl.diagnose_chip(backend, chip.executor, lines=lines, log=lambda s: None)
        res = three_way(backend, chip, h, ghz)
        rowsB.append({k: v["fidelity"] for k, v in res.items()})
        print(f"  trial {i+1:2d}: default {res['default']['fidelity']:.3f}  syndromics "
              f"{res['syndromics']['fidelity']:.3f}  oracle {res['oracle']['fidelity']:.3f}")
    m = {k: float(np.mean([r[k] for r in rowsB])) for k in rowsB[0]}
    wins = sum(r["syndromics"] > r["default"] + 0.01 for r in rowsB)
    ties = sum(abs(r["syndromics"] - r["default"]) <= 0.01 for r in rowsB)
    print(f"  mean: default {m['default']:.3f}  syndromics {m['syndromics']:.3f}  oracle {m['oracle']:.3f}"
          f"   (better in {wins}, same in {ties}, worse in {a.trials-wins-ties})")

    # ---- no-drift control: Syndromics must not make things worse when nothing changed
    chip = pl.SimulatedChip(backend, None)
    h = pl.diagnose_chip(backend, chip.executor, lines=lines, log=lambda s: None)
    ctrl = three_way(backend, chip, h, ghz)
    print(f"\nControl (no drift, GHZ-12): default {ctrl['default']['fidelity']:.3f}  "
          f"syndromics {ctrl['syndromics']['fidelity']:.3f}  oracle {ctrl['oracle']['fidelity']:.3f}"
          f"  | false alarms: {len(h.flagged)}")

    json.dump({"scenarioA": {"faults": desc, "rows": rowsA, "flagged": health.flagged},
               "scenarioB": {"rows": rowsB, "mean": m, "wins": wins, "ties": ties},
               "control": {k: v["fidelity"] for k, v in ctrl.items()}, "control_flags": h.flagged,
               "coverage": [len(covered), backend.num_qubits]},
              open(os.path.join(a.out, "results.json"), "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
