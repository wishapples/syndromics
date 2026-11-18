# Changelog

Syndromics is updated continuously. Every hardware finding becomes a feature and an automated test.

## 0.5.1 (18 November 2026)

- `syndromics-ibm-demo validate --line ...` runs on chosen qubits, so a second run can reuse the first run's line.
- `syndromics-ibm-demo compare-runs --previous DIR`: prospective test. Freezes the noise model and adaptive
  settings learned on an earlier run, decodes the new run, and lists components that changed between runs.

## 0.5.0 (18 November 2026): adaptive learning

- **Leakage-aware, per-shot adaptive decoding** (`syndromics.adaptive`). When one measure qubit fires at least
  *k* detection events in a shot, the decoder stops trusting that qubit's measurements for that shot only.
  Settings are chosen by cross-validation on training data. On real `ibm_kingston` data (held-out shots):
  0.198% → 0.103% logical error, better in 6 of 6 splits. Works with any decoder model.
- **Burst statistics.** Shots where one measure qubit behaves erratically (>= 6 detection events): 2.26% of
  real shots vs 0.037% predicted by the fitted model (62x). Consistent with leakage or reset failure, not yet confirmed.
- **Drift tracking** over time windows, with alerts confirmed in two windows. Simulated gate x4 and
  measurement x3 drift caught within one window, no false alarms; no drift found during the real job.
- Tested and rejected: "memory" updates that start each fit from the previous fitted model (unstable
  against simulated data with a known answer).

## 0.4.3 (18 November 2026)

- Held-out decoding comparison (uniform, calibration, correlation re-weighting, Syndromics) is the headline of
  `syndromics-ibm-demo validate` and `reanalyse`.
- Fitted crosstalk is kept out of the decoder model (it could add a logical-flipping shortcut).
- Real `ibm_kingston` data added as a regression test.

## 0.4.0 - 0.4.2 (18 November 2026)

- Self-service IBM demo kit (`syndromics-ibm-demo`): check, rehearse, validate, health, A/B placement, report.
- Same-job reference measurements; raw data saved; equal shots per job (IBM requirement).
- First run on real hardware: `ibm_kingston`, job `dbukcjvf8rhs73bmf9l0`.

## 0.3.0 (16 November 2026)

- Noise-aware qubit placement from measured chip health; plain-English circuit recipes.

## 0.1.0 - 0.2.0 (16 November 2026)

- Chip diagnostics from surface-code syndromes; circuit-agnostic noise fitting (`fit_noise`) for any Stim
  circuit; Google Willow public-data pipeline; IBM repetition-code experiment.
