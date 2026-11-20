# Syndromics demo for IBM Quantum: run it yourself

This kit lets an IBM Quantum engineer check Syndromics on an IBM backend **with their own account**,
in about 30 minutes of hands-on time and a few minutes of QPU time.

| Step | What it shows | IBM account | QPU use |
|---|---|---|---|
| 0. `check` | Environment and account are ready | optional | none |
| 1. `rehearse` | Reproduces our simulation results locally | no | none |
| 2. `reanalyse` on our data | Reproduces our **real ibm_kingston result** from the raw shots included in the kit | no | none |
| A. `validate` | On your chip: does a decoder given the **Syndromics noise model** make fewer logical errors than one given **IBM's calibration**? Tested on held-out shots | yes | 5 circuits × 30,000 shots, one job |
| B. `health` | What has **changed since the last calibration**, chip-wide | yes | ~7 circuits × 50,000 shots |
| C. `ab` | The same circuits with **default vs Syndromics placement**, interleaved in one job | yes | 6 circuits × 4,000 shots |
| `reanalyse` | Re-runs the step-A analysis on saved raw data | no | none |
| `report` | One HTML page with all results | no | none |

**Privacy.** The kit talks only to IBM Quantum, through your saved Qiskit Runtime account. Nothing is
sent to Syndromics. Results go to `./results/` (JSON + `report.html`); share them only if you wish.
They contain backend names, job IDs, qubit numbers and error rates, but no credentials.

**Safety.** Hardware steps print the circuits, shots and a rough QPU-time estimate, then stop. They
submit only when you add `--yes`.

---

## 1. Install (5 minutes)

Requires Python 3.10–3.13.

```bash
python -m venv syndromics-env
# Windows: syndromics-env\Scripts\activate      macOS/Linux: source syndromics-env/bin/activate
source syndromics-env/bin/activate
pip install -r requirements.txt
```

If you have not used Qiskit Runtime on this machine before, save your account once
(token and instance CRN from https://quantum.cloud.ibm.com):

```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(token="YOUR_API_KEY", instance="YOUR_INSTANCE_CRN", set_as_default=True)
```

## 2. Check and rehearse (no QPU use)

```bash
syndromics-ibm-demo check
syndromics-ibm-demo rehearse
```

`check` lists the backends your account can use. `rehearse` runs everything on a local simulator of
`ibm_kingston` with four planted post-calibration faults and should print, in about 20 seconds:

```
Health check: 7 lines, 115 qubits; changes found:
  q103       readout 0.1546, calibration says 0.0116 (13.3x)
  q103-q104  gate 0.0459, calibration says 0.0016 (28.1x)
  q104       readout 0.0124, calibration says 0.0048 (2.6x)
  q104-q105  gate 0.0374, calibration says 0.0016 (22.7x)
  q105       readout 0.0948, calibration says 0.0059 (16.2x)
  q142-q143  gate 0.0597, calibration says 0.0013 (47.7x)
Control (same chip, no drift): 0 false alarm(s)
  GHZ state on 12 qubits         default 0.646  syndromics 0.877
  Grover search for 101          default 0.566  syndromics 0.981
  QFT round trip on 5 qubits     default 0.762  syndromics 0.964
```

## 3. Reproduce our hardware result (no account, no QPU)

The kit includes the raw shots from our run on `ibm_kingston` (job `dbukcjvf8rhs73bmf9l0`, 18 November 2026,
qubits 117-105-106-107-108-109-118-129-128, 10 rounds, 30,000 shots). From the kit folder:

```bash
cd example_ibm_kingston_2026-11-18
syndromics-ibm-demo reanalyse
```

Expected output (about 20 seconds), abridged:

```
=== 1. Does the measured noise model decode better? ===
  Noise model given to the decoder       logical error  vs calibration
  Uniform (no device data)                      0.447%            +13%
  Device calibration                            0.396%        baseline
  Standard correlation re-weighting             0.199%            -50%
  Syndromics noise model                        0.198%            -50%
  Re-weighting + leakage-aware                  0.093%            -76%
  Syndromics + leakage-aware                    0.097%            -76%
  Syndromics beat the calibration in 6 of 6 tests.
  Leakage-aware decoding improved on Syndromics in 6 of 6 tests ...

=== 2. Components worse than the calibration says (fit on all shots) ===
  q129-q128 two-qubit gate           x17.2 +/- 0.6   (calibration 0.0012, measured 0.0201)
  q105 measurement                   x4.3 +/- 0.1   (calibration 0.0020, measured 0.0083)
  ...
=== 3. Shots where a measure qubit behaves randomly (leakage-like bursts) ===
  shots where one measure qubit fires >= 6 detection events: real 2.26%, noise model predicts about 0.03% (ratio varies by machine)
=== 4. Drift during the job (6 time windows, fresh fit per window) ===
  no confirmed drift alerts: the components were stable over this job
```

How to read it: the decoder given the Syndromics model made **half as many logical errors** as the one
given IBM's calibration, in every one of six train/test splits (five random, one time-ordered).
Leakage-aware decoding then halves it again: shots where one measure qubit behaves randomly (2.3% of shots, far more
common than a Pauli model predicts) are flagged per shot and that qubit's measurements are distrusted. Standard
model-free correlation re-weighting does equally well on this small code; we say so plainly. What Syndromics
adds is the **physical attribution** (section 2): it points at the q128-q129 end of the line, where IBM's own
data shows q128 with the lowest T2. A repetition code localises a fault well but splits it only roughly between
a measurement and its neighbouring gate, so read large ratios as "look here".

## 4. Run on your hardware

Pick any Heron backend from the `check` output (examples use `ibm_kingston`). Run each step once
without `--yes` to see what it will submit, then again with `--yes`.

### A. Validate: does the measured noise model decode better on your chip? (5 circuits, one job)

```bash
syndromics-ibm-demo validate --backend ibm_kingston --yes
```

One job contains a 5-data-qubit repetition code (10 rounds) on the best line it finds, plus four short
reference circuits on the same qubits (terminal readout and repeated mid-circuit measure + reset). The kit then:

1. **Held-out decoding test.** Splits the shots in half six times (five random splits and one time split). For
   each split it fits the noise model on one half and decodes the other half with PyMatching, using four
   noise models: uniform, IBM's calibration, standard correlation re-weighting, and Syndromics (a physical
   fit starting from IBM's calibration).
2. **Components worse than calibration.** Lists each measurement, gate or idle component that is at least
   2x worse than its calibrated value.
3. **Detail.** Each qubit's measurement error as Syndromics sees it, next to today's references and IBM's
   calibration. Syndromics measures the effective per-round error (measure + reset + neighbouring gate errors),
   which is larger than the end-of-circuit readout IBM publishes. So these numbers are not expected to match
   the published readout one-to-one.

Raw data is saved to `results/A_raw.npz`; `syndromics-ibm-demo reanalyse` repeats the analysis at no QPU cost.

*What to look for:* on our run, Syndromics roughly halved the calibration's logical error in 6 of 6 tests.
On a line that is well described by its calibration, expect a smaller gain and "none" in part 2. Both are
valid results. If the kit prints a NOTE that there are too few logical errors, add `--rounds 20` or more shots.

### B. Health: what changed since calibration (≈7 circuits)

```bash
syndromics-ibm-demo health --backend ibm_kingston --yes
```

Covers the chip with disjoint lines of qubits (115 of 156 on Kingston) and runs all of them in one
job. The fit starts from **IBM's own calibration** and reports only components clearly worse than
their own calibrated value (≥2× and ≥4σ). Writes `results/chip_health.json`.

*What to look for:* if nothing has drifted, it reports **no changes**: a valid and useful result. If
it flags components, they can be checked with a targeted single-qubit or two-qubit characterisation.

### C. A/B: default vs Syndromics placement (6 circuits)

```bash
syndromics-ibm-demo ab --backend ibm_kingston --yes
```

For each circuit (default: GHZ-12, Grover-3, QFT-5), Qiskit transpiles twice at optimisation level
3: once with IBM's published calibration, once with the Target corrected by step B. Both versions run
in the same job, so they share conditions. The output is Hellinger fidelity to the ideal result.

*What to look for:* where step B found changes on the qubits Qiskit prefers, Syndromics should score
higher. Where it found none, both layouts are identical and the scores differ only by shot noise; the
report says so.

### Report

```bash
syndromics-ibm-demo report      # writes results/report.html
```

## Options

| Option | Default | Meaning |
|---|---|---|
| `--backend NAME` | `ibm_kingston` | any backend from `check` |
| `--instance CRN` | saved default | a different IBM Quantum instance |
| `--shots N` | 30,000 / 50,000 / 4,000 | per circuit, for validate / health / ab |
| `--sparsity S` | 0.5 | validate / reanalyse: fit regularisation; 0 lets every component move |
| `--fake` | off | local simulator with IBM's published noise; no account, no QPU |
| `--max-lines N`, `--line-length L` | all, 19 | health: fewer or shorter lines (quick test) |
| `--circuits "A; B"` | GHZ-12; Grover-3; QFT-5 | ab: plain-English circuit list (see `syndromics place --help`) |

## Notes and limits

- The QPU-time estimate is rough. Actual usage appears in your IBM Quantum dashboard.
- Diagnostic lines cover qubits along the lines and the couplers between them. Other couplers keep
  IBM's published values.
- The model is Pauli noise: leakage and coherent errors are not modelled yet.
- In local `--fake` mode, health uses short lines for speed. Full noisy simulation of 19-qubit lines
  is slow; real hardware is not.

## 5. Second run: test everything learned on a fresh run (prospective test)

A held-out half of the same job is a good test; a new job hours or days later is a stronger one. Run the
validation again **on the same qubits**, in a new folder, then decode the new run with the model and adaptive
settings frozen from the first run:

```bash
mkdir run2 && cd run2
syndromics-ibm-demo validate --backend ibm_kingston --line 117,105,106,107,108,109,118,129,128 --yes
syndromics-ibm-demo compare-runs --previous ../example_ibm_kingston_2026-11-18
syndromics-ibm-demo reanalyse        # run 2's own held-out test, for comparison
```

`compare-runs` (no QPU time) prints the logical error on run 2 with: IBM's calibration at run 2, IBM's
calibration at run 1, the Syndromics model frozen from run 1, and Syndromics + adaptive frozen from run 1.
It also lists components that changed between the runs (drift) and the erratic-measure-qubit rate in run 2.
Our own second run (job `dbvc3lv82tmc73dupa7g`, 19 November 2026, 27 hours later, after IBM recalibrated) gave:

```
  IBM calibration at run 2                                        123 errors = 0.410%   baseline
  IBM calibration at run 1                                        119 errors = 0.397%   -3%
  Syndromics model from run 1 (frozen)                             55 errors = 0.183%   -55%
  Syndromics + adaptive from run 1 (frozen: k=4, boost=40)         17 errors = 0.057%   -86%
=== Components that changed between the runs ===
  q107 measurement                   x1.3 -> x2.6 calibration (z = +4.4)
  q108-q109 two-qubit gate           x3.3 -> x1.0 calibration (z = -4.0)
```

The kit includes it as `example_ibm_kingston_2026-11-19/`: from that folder,
`syndromics-ibm-demo compare-runs --previous ../example_ibm_kingston_2026-11-18` reproduces it with no account.



### Next runs (planned before they are run)

Kit 0.6.2 fixes one setting in advance: the adaptive boost is chosen from 40 or 80 (it was 5-40). The next runs test this
and the transfer on new hardware: run 4 on ibm_fez (distance 5, 10 rounds, the line it chooses), run 5 a distance-3,
20-round code on five qubits of run 4's line, and run 6 a 30-round code on the same nine Kingston qubits. Each is decoded with
`compare-runs --previous` the earlier run, with nothing tuned on the new data.

## 6. Transfer test: does the model carry over to a different circuit?

Correlation re-weighting learns numbers for one specific circuit, so a new circuit needs new training shots. A physical
model describes the qubits and gates, so it should carry over. To test that, run a **different** circuit on part of the same
line, here a distance-3 code with 20 rounds on the end of the line where the q128-q129 fault sits, and decode it with the
model learned on the earlier run:

```bash
mkdir run3 && cd run3
syndromics-ibm-demo validate --backend ibm_kingston --n-data 3 --rounds 20 --line 108,109,118,129,128 --yes
syndromics-ibm-demo compare-runs --previous ../example_ibm_kingston_2026-11-19
```

`compare-runs` sees that the circuits differ and runs the transfer test. On the same held-out half of the new run it
compares IBM's calibration and the transferred Syndromics model (no new training data) with correlation re-weighting
learned on 1,000, 3,000 and 15,000 shots of the new circuit. On data simulated from the Kingston model (where the true
answer is known), the transferred model reached 0.29% logical error against 1.06% for the calibration and 0.50% for
re-weighting trained on 15,000 new shots. Real hardware has noise the model does not contain, so expect a smaller gap.

Our own hardware run (job `dbvdl8ups3fc73bkad70`, included as `example_ibm_kingston_2026-11-19_transfer/`):
calibration 150 errors, transferred Syndromics model 69, re-weighting on 1,000 / 3,000 / 15,000 run-3 shots 105 / 102 / 77
(15,000 held-out shots). From that folder: `syndromics-ibm-demo compare-runs --previous ../example_ibm_kingston_2026-11-19`.

Questions or results: Vishal J Shah, CEO, Wish Apples Ventures Private Limited (Syndromics), ceo@wishapples.com.
