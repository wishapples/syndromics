"""Quickstart: diagnose a simulated distance-5 chip with hidden defects.

    pip install syndromics        # or: pip install -e .  from a clone
    python examples/quickstart.py
"""
from syndromics import diagnose, score_against_truth
from syndromics.plot import plot_report
from syndromics.sim import nominal_circuit, real_chip_circuit, sample

P = 0.004                                                     # nominal physical error rate
real, truth = real_chip_circuit(5, 5, P, seed=7)              # the "real" chip; truth is hidden
nominal = nominal_circuit(5, 5, P)                            # what the vendor designed

detections, _ = sample(real, 200_000, seed=3)                 # 200k shots of syndrome data
report = diagnose(detections, nominal, P)                     # no access to `truth`

print(report.summary())
print(score_against_truth(report, truth))
plot_report(report, nominal, "chip_map.png", truth=truth)
