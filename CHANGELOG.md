# Changelog

Syndromics is updated continuously. Every hardware finding becomes a feature and an automated test.

## 0.6.1 (19 November 2026): transfer test on hardware

- Run 3 on ibm_kingston (job `dbvdl8ups3fc73bkad70`): distance 3, 20 rounds on q108-q128. The model frozen from run 2
  (distance 5, 10 rounds) gave 320 logical errors over 5 held-out splits vs 769 for the calibration and 332 for
  re-weighting trained on 15,000 run-3 shots (411 with 3,000; 511 with 1,000). Data in `tests/data/`, new regression test.
- Adaptive settings transfer less well: boost 20 (chosen on run 2) gave 149 vs 94 for re-weighting + adaptive trained on
  run 3; boost 40-80 gives about 85 (post hoc; to be confirmed on a new run). Scaling k with the number of rounds did
  not help.

## 0.6.0 (19 November 2026): transfer test

- `compare-runs` now also handles a **different circuit** in run 2 (more rounds, or a shorter code on part of run 1's
  line). It decodes run 2 with the model frozen from run 1 and compares it with correlation re-weighting learned on
  1,000 / 3,000 / 15,000 shots of the new circuit (`adaptive.transfer_runs`).
- Simulated check (truth = model fitted on real Kingston data): d=3, 20 rounds on q108-q128, transferred model 0.29% vs
  calibration 1.06% vs re-weighting on 15,000 new shots 0.50%. Not yet run on hardware.
- New test `test_transfer_to_shorter_code_on_part_of_the_line`.
- `examples/cudaq_qec_export.py`: Syndromics noise model inside NVIDIA CUDA-Q QEC (0.9.0) decoders. Run 2, CUDA-Q QEC
  PyMatching: calibration 123 errors, frozen Syndromics 51. GPU decoders need a GPU; not yet run.

## 0.5.2 (19 November 2026): second hardware run

- **Prospective test on a second day.** Run 2 on the same `ibm_kingston` qubits (job `dbvc3lv82tmc73dupa7g`),
  after IBM recalibrated. Frozen from run 1: Syndromics 0.183%, Syndromics + adaptive 0.057%, vs 0.410% for
  run 2's fresh calibration. Frozen re-weighting gives the same (0.163%, 0.057%); stated plainly.
- **Fix: `compare-runs` change list.** It used an unregularised fit, which cannot separate a measurement error from
  its neighbouring gate, and reported a spurious "q117-q105 gate x7.9 -> x1.5" change. It now uses the same
  regularised fit as the main report. Real change found: q107 measurement x1.3 -> x2.6.
- Run 2 data added to `tests/data/`; new regression test `test_prospective_second_day_on_real_kingston`.
- Burst ratios vary with the fitted model (62x, 75x, 19x); reports should quote the measured burst rate (2.3-2.4%).
- Research: neural hybrid trained from run 1 only, 0.034% on run 2 (3 seeds); without bursts no gain.

## Research: neural decoder on real data (19 November 2026)

- `research/neural_decoder/real_kingston.py`: neural hybrid decoder trained on simulated shots from the Syndromics
  model plus erratic-qubit bursts; 0.064% logical error on held-out real `ibm_kingston` shots (adaptive decoding:
  0.10%). Ablation without bursts: no gain.

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
