"""The IBM self-service demo runs end to end on a local simulator of ibm_kingston."""
import json
import os

import pytest

pytest.importorskip("qiskit_aer")
pytest.importorskip("qiskit_ibm_runtime")

from syndromics import ibmdemo


def test_demo_steps_on_fake_backend(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ibmdemo.main(["validate", "--fake", "--shots", "15000"])
    a = json.load(open("results/A_validate.json"))
    assert a["meta"]["fake"] and a["agreement"]["syndromics_vs_today_reference"] > 0.8
    assert os.path.exists("results/A_raw.npz")
    assert set(a["decoding"]["mean"]) == {"uniform", "calibration", "reweighting", "syndromics", "reweighting_la", "syndromics_la"}
    assert "bursts" in a and "drift" in a
    assert len(a["decoding"]["tests"]) == 6
    ibmdemo.main(["reanalyse", "--sparsity", "2"])
    assert json.load(open("results/A_validate.json"))["decoding"]["sparsity"] == 2.0
    ibmdemo.main(["health", "--fake", "--shots", "2000", "--max-lines", "2", "--line-length", "9"])
    b = json.load(open("results/B_health.json"))
    assert b["flagged"] == {}          # IBM's full noise model, no planted drift: no false alarms
    ibmdemo.main(["ab", "--fake", "--shots", "1000", "--circuits", "GHZ state on 4 qubits"])
    c = json.load(open("results/C_ab.json"))
    assert 0.5 < c["rows"][0]["default_fidelity"] <= 1.0
    ibmdemo.main(["report"])
    rep = open("results/report.html").read()
    assert "Held-out decoding test" in rep and "same-day references" in rep


def test_hardware_step_needs_yes(tmp_path, monkeypatch):
    """Without --yes a real-hardware step must not submit anything (no account needed to check)."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        ibmdemo.main(["validate", "--backend", "ibm_kingston"])
    assert not os.path.exists("results/A_validate.json")


def test_validate_uses_equal_shots_per_job(tmp_path, monkeypatch):
    """IBM's sampler rejects a job whose circuits have different shot counts; Aer does not, so check it."""
    monkeypatch.chdir(tmp_path)
    from qiskit_aer.primitives import SamplerV2 as AerSampler
    orig = AerSampler.run

    def strict_run(self, pubs, *, shots=None):
        assert not any(isinstance(p, tuple) and len(p) > 2 and p[2] is not None for p in pubs), "per-circuit shots"
        return orig(self, pubs, shots=shots)

    monkeypatch.setattr(AerSampler, "run", strict_run)
    ibmdemo.main(["validate", "--fake", "--shots", "2000"])
    assert os.path.exists("results/A_raw.npz")
