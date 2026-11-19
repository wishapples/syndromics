# Syndromics Neural Decoder (research)

A neural network that decodes quantum error correction using what Syndromics measures about the chip.

## Result on real IBM hardware (ibm_kingston, job dbukcjvf8rhs73bmf9l0)

Same 9-qubit repetition-code data and the same held-out test as the main package: each split fits on 15,000
real shots and is scored on the other 15,000, which no model ever sees. Totals over 6 splits (90,000 test shots):

| Decoder | Logical errors | Rate |
|---|---|---|
| Matching with IBM calibration | 356 | 0.40% |
| Matching with the Syndromics noise model | 178 | 0.20% |
| Syndromics + adaptive (leakage-aware) decoding | 93 | 0.10% |
| **Neural hybrid** | **58** | **0.064%** |

Better than the adaptive decoder in 5 of 6 splits, tied in 1. Per split: `results_real_kingston.json`.

**How it is trained.** Never on real test shots. The network learns from about 1.5 million *simulated* shots drawn
from the Syndromics noise model fitted on the training half, with erratic-measure-qubit bursts (a measure qubit
returning random outcomes for 3-10 rounds) added at the rate seen in the training half. Input: the detection events
plus the Syndromics matching decoder's answer; the network learns when to override it.

**Why it works (ablation).** Trained without the simulated bursts, the network reproduces the matching decoder
exactly (31, 34 and 27 errors on three splits, identical to it). The whole gain comes from the erratic-measure-qubit
behaviour that Syndromics found in the real data: diagnosis turned into training data.

**Limits.** One run, one 9-qubit line, a repetition code. The burst model is simple (random outcomes on one measure
qubit) and its rate sat at the top of the grid searched. A correlation-based noise model could also serve as the
simulator; that comparison is not done yet. About 1 minute per split on a laptop CPU for training; inference speed
is not yet optimised.

Run: `pip install -e ".[ibm,dev]" torch`, then `python real_kingston.py` (all 6 splits) or
`NOBURST=1 python real_kingston.py 1500000 5,0,2` (ablation).

## Prospective test: a second day (job dbvc3lv82tmc73dupa7g, 19 November 2026)

Everything learned on run 1 only (noise model, burst rate, network), then all 30,000 shots of run 2 decoded:
same qubits, 27 hours later, after IBM recalibrated the chip. Nothing is tuned on run 2.

| Decoder (run 2, 30,000 shots) | Logical errors | Rate |
|---|---|---|
| Matching with IBM calibration from the morning of run 2 | 123 | 0.41% |
| Matching with the Syndromics model from run 1 | 55 | 0.18% |
| Syndromics + adaptive, settings from run 1 | 17 | 0.057% |
| **Neural hybrid trained from run 1** (3 seeds) | **9, 11, 11** | **0.034%** |
| Neural hybrid without bursts (ablation, 3 seeds) | 55, 55, 54 | 0.18% |

About 12x fewer logical errors than the fresh calibration. Counts are small (about 10 vs 17 against adaptive
decoding), so read the neural-vs-adaptive gap as promising, not settled. Run: `python prospective_kingston.py`;
results in `results_prospective_kingston.json` and `results_prospective_kingston_noburst.json`.

## Earlier simulation study (`chip.py`, `train.py`)

Surface code with hidden faulty qubits, crosstalk and bad readout (Stim). At distance 5 the hybrid decoder reached
1.71% logical error vs 2.46% for matching with a uniform model and 1.69% for matching given the true hidden noise;
at distance 3 it is essentially optimal (5.96% vs a 5.97% lookup-table bound). The multilayer perceptron does not
scale past distance 7; larger codes need convolutional or transformer architectures and GPU training.
