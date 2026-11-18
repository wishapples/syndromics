"""sweep.py - Robustness of diagnostics: 5 random chips x 3 defect strengths x 3 data budgets (d=5)."""
import json, numpy as np
from syndromics import diagnose, score_against_truth
from syndromics import sim as chip

STRENGTHS = {  # (bad qubit factor, crosstalk rate / p, bad readout factor)
    "strong (6x / 3p / 5x)": (6.0, 3.0, 5.0),
    "medium (3x / 1.5p / 3x)": (3.0, 1.5, 3.0),
    "weak (2x / 0.75p / 2x)": (2.0, 0.75, 2.0),
}
SHOTS = [50_000, 200_000, 1_000_000]
d, p = 5, 0.004
rows = []
for sname, (bf, xf, rf) in STRENGTHS.items():
    for shots in SHOTS:
        agg = {"true_defects": 0, "localised": 0, "exact": 0, "false_alarms": 0, }
        for seed in range(100, 105):
            real, truth = chip.real_chip_circuit(d, d, p, seed=seed, bad_factor=bf,
                                                 xtalk_p_factor=xf, readout_factor=rf)
            det, _ = chip.sample(real, shots, seed=seed + 1)
            ev = score_against_truth(diagnose(det, chip.nominal_circuit(d, d, p), p), truth)
            for k in agg: agg[k] += ev[k]
        row = {"strength": sname, "shots": shots, **agg,
               "recall": agg["localised"] / agg["true_defects"]}
        rows.append(row); print(row, flush=True)
json.dump(rows, open("results_sweep.json", "w"), indent=1)
