import numpy as np

from syndromics import diagnose, score_against_truth
from syndromics.sim import nominal_circuit, real_chip_circuit, sample


def test_clean_chip_has_no_findings():
    nominal = nominal_circuit(3, 3, 0.004)
    det, _ = sample(nominal, 200_000, seed=1)
    rep = diagnose(det, nominal, 0.004, baseline_shots=400_000)
    assert rep.flagged == []


def test_finds_hidden_defects_d5():
    real, truth = real_chip_circuit(5, 5, 0.004, seed=7)
    nominal = nominal_circuit(5, 5, 0.004)
    det, _ = sample(real, 200_000, seed=3)
    score = score_against_truth(diagnose(det, nominal, 0.004), truth)
    assert score["localised"] == score["true_defects"]
    assert score["false_alarms"] == 0


def test_shape_mismatch_raises():
    nominal = nominal_circuit(3, 3, 0.004)
    try:
        diagnose(np.zeros((10, 5), np.uint8), nominal, 0.004)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
