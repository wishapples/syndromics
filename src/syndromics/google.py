"""
Run Syndromics on Google Quantum AI's public surface-code datasets.

Data: "Quantum error correction below the surface code threshold", Nature 638, 920 (2025),
Zenodo doi:10.5281/zenodo.13273331. Unzip so that folders look like

    <root>/d3_at_q6_7/Z/r10/detection_events.b8
    <root>/d3_at_q6_7/Z/r10/circuit_noisy_si1000*.stim     (Google's nominal SI1000 model)
    <root>/d3_at_q6_7/Z/r10/obs_flips_actual.01            (used for the decoding check)

Validation without an answer key:
  1. Held-out decoding: fit on half the shots, decode the other half with PyMatching using
     (a) Google's nominal SI1000 model and (b) the Syndromics-calibrated model. If the fitted
     component errors are real, (b) makes fewer logical errors on data it never saw.
  2. Repeatability: the same physical qubits appear in independent experiments (X and Z basis,
     overlapping patches). Their fitted scale factors should agree.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import time
from dataclasses import dataclass

import numpy as np
import stim

from .model import fit_noise


@dataclass
class Experiment:
    path: str
    patch: str
    basis: str
    rounds: int
    circuit_file: str
    det_file: str
    obs_file: str | None


def find_experiments(root: str):
    exps = []
    for det in glob.glob(os.path.join(root, "**", "detection_events.b8"), recursive=True):
        d = os.path.dirname(det)
        stims = sorted(glob.glob(os.path.join(d, "*.stim")))
        noisy = [s for s in stims if "si1000" in os.path.basename(s)] or \
                [s for s in stims if "noisy" in os.path.basename(s)]
        if not noisy:
            continue
        parts = os.path.relpath(d, root).split(os.sep)
        m = re.search(r"r0*(\d+)", parts[-1]) if parts else None
        obs = os.path.join(d, "obs_flips_actual.01")
        exps.append(Experiment(
            path=d, patch=parts[-3] if len(parts) >= 3 else d,
            basis=parts[-2] if len(parts) >= 2 else "?",
            rounds=int(m.group(1)) if m else -1, circuit_file=noisy[0], det_file=det,
            obs_file=obs if os.path.exists(obs) else None))
    return sorted(exps, key=lambda e: (e.patch, e.basis, e.rounds))


def pick(exps, target_rounds=10, max_per_patch_basis=1):
    """One experiment per (patch, basis): the round count closest to `target_rounds`."""
    best = {}
    for e in exps:
        k = (e.patch, e.basis)
        best.setdefault(k, []).append(e)
    out = []
    for k, lst in sorted(best.items()):
        lst.sort(key=lambda e: (abs(e.rounds - target_rounds), e.rounds))
        out += lst[:max_per_patch_basis]
    return out


def _decode_ler(circuit, det, obs):
    import pymatching
    dem = circuit.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True,
                                       approximate_disjoint_errors=True)
    m = pymatching.Matching.from_detector_error_model(dem)
    pred = m.decode_batch(det)
    return float((pred[:, 0] != obs[:, 0]).mean())


def _coord_name(coords, q):
    c = coords.get(q)
    return f"q{int(c[0])}_{int(c[1])}" if c is not None else f"q{q}"


def analyse(e: Experiment, crosstalk=True):
    circ = stim.Circuit.from_file(e.circuit_file)
    D = circ.num_detectors
    det = stim.read_shot_data_file(path=e.det_file, format="b8", num_detectors=D).astype(np.uint8)
    obs = None
    if e.obs_file:
        obs = stim.read_shot_data_file(path=e.obs_file, format="01",
                                       num_observables=circ.num_observables).astype(np.uint8)
    n = len(det)
    idx = np.random.default_rng(1).permutation(n)
    fit_i, test_i = idx[: n // 2], idx[n // 2:]
    t = time.time()
    fit = fit_noise(circ, det[fit_i], crosstalk=crosstalk)
    secs = time.time() - t
    coords = circ.get_final_qubit_coordinates()
    rows = []
    for f in fit.fits:
        c = f.component
        phys = [_coord_name(coords, q) for q in c[1:3] if isinstance(q, int)]
        rows.append({"patch": e.patch, "basis": e.basis, "rounds": e.rounds,
                     "kind": c[0], "qubits": "-".join(phys), "pauli": c[3] if c[0] == "xtalk" else "",
                     "scale": f.scale, "err": f.err, "label": f.label,
                     "alternatives": " | ".join("-".join(_coord_name(coords, q) for q in m[1:3]) for m in f.members)})
    res = {"patch": e.patch, "basis": e.basis, "rounds": e.rounds, "shots": n, "fit_seconds": secs,
           "outliers": [{"label": f.label, "scale": f.scale, "err": f.err,
                         "qubits": [_coord_name(coords, q) for q in f.component[1:3] if isinstance(q, int)]}
                        for f in fit.outliers()]}
    if obs is not None:
        res["ler_nominal_heldout"] = _decode_ler(circ, det[test_i], obs[test_i])
        res["ler_calibrated_heldout"] = _decode_ler(fit.calibrated_circuit(), det[test_i], obs[test_i])
    return res, rows


def repeatability(rows, summaries):
    """Do independent experiments on the same physical qubits agree?

    * flag_agreement: of the physical qubits flagged as outliers in any experiment, the share
      flagged again in another experiment covering the same qubit.
    * pearson_r over components that deviated from nominal in at least one experiment
      (components left at exactly 1.0 by the sparse fit are excluded, as they would inflate r).
    """
    by = {}
    for r in rows:
        if r["kind"] == "xtalk":
            continue
        by.setdefault((r["kind"], r["qubits"]), []).append(r)
    xs, ys = [], []
    for lst in by.values():
        if len(lst) >= 2:
            a, b = lst[0], lst[1]
            if abs(a["scale"] - 1) > 1e-6 or abs(b["scale"] - 1) > 1e-6:
                xs.append(a["scale"]); ys.append(b["scale"])
    covered = {}
    for r in rows:
        for q in r["qubits"].split("-"):
            covered.setdefault(q, set()).add((r["patch"], r["basis"], r["rounds"]))
    flagged = {}
    for s in summaries:
        key = (s["patch"], s["basis"], s["rounds"])
        for o in s["outliers"]:
            for q in o["qubits"]:
                flagged.setdefault(q, set()).add(key)
    testable = [q for q in flagged if len(covered.get(q, ())) >= 2]
    agreed = [q for q in testable if len(flagged[q]) >= 2]
    out = {"deviating_component_pairs": len(xs),
           "flagged_qubits_testable": len(testable), "flagged_again": len(agreed),
           "flag_agreement": (len(agreed) / len(testable)) if testable else None}
    if len(xs) >= 3:
        out["pearson_r_deviating"] = float(np.corrcoef(xs, ys)[0, 1])
    return out


def run(root, out="willow_results", target_rounds=10, limit=None, crosstalk=True):
    os.makedirs(out, exist_ok=True)
    exps = pick(find_experiments(root), target_rounds)
    if limit:
        exps = exps[:limit]
    if not exps:
        raise SystemExit(f"No experiments found under {root}. Expected .../detection_events.b8 "
                         f"next to a circuit_noisy_si1000*.stim file.")
    print(f"{len(exps)} experiments selected (one per patch and basis, ~{target_rounds} rounds).")
    summaries, rows = [], []
    for k, e in enumerate(exps, 1):
        print(f"[{k}/{len(exps)}] {e.patch} {e.basis} r={e.rounds} ...", flush=True)
        s, r = analyse(e, crosstalk=crosstalk)
        summaries.append(s); rows += r
        if "ler_nominal_heldout" in s:
            print(f"    held-out logical error: nominal {s['ler_nominal_heldout']:.4f} -> "
                  f"calibrated {s['ler_calibrated_heldout']:.4f}   ({len(s['outliers'])} outliers)")
    with open(os.path.join(out, "components.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    rep = repeatability(rows, summaries)
    dec = [s for s in summaries if "ler_nominal_heldout" in s]
    summary = {"experiments": summaries, "repeatability": rep}
    if dec:
        better = sum(s["ler_calibrated_heldout"] < s["ler_nominal_heldout"] for s in dec)
        rel = [1 - s["ler_calibrated_heldout"] / s["ler_nominal_heldout"] for s in dec if s["ler_nominal_heldout"] > 0]
        summary["decoding"] = {"experiments": len(dec), "calibrated_better": better,
                               "median_relative_reduction": float(np.median(rel)) if rel else None}
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print("\nRepeatability across independent experiments:", rep)
    if dec:
        print("Decoding with calibrated model:", summary["decoding"])
    print(f"Results written to {out}/")
    return summary
