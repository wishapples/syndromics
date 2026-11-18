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


def test_compare_runs_on_split_real_data(tmp_path, monkeypatch):
    """Run 1 = first half of the real Kingston job, run 2 = second half: frozen run-1 model beats run-2 calibration."""
    import numpy as np
    here = os.path.join(os.path.dirname(__file__), "data")
    z = np.load(os.path.join(here, "kingston_20261118_rep.npz"))
    rep = np.unpackbits(z["rep"], axis=1)[:, : int(z["ncols"])]
    cfg = json.load(open(os.path.join(here, "kingston_20261118_config.json")))
    for d, sl in (("run1", slice(0, 15000)), ("run2", slice(15000, 30000))):
        os.makedirs(tmp_path / d / "results")
        np.savez_compressed(tmp_path / d / "results" / "A_raw.npz", rep=rep[sl])
        c = dict(cfg); c["job_id"] = d
        json.dump(c, open(tmp_path / d / "results" / "A_config.json", "w"))
    monkeypatch.chdir(tmp_path / "run2")
    ibmdemo.main(["compare-runs", "--previous", str(tmp_path / "run1")])
    r = json.load(open("results/B_compare_runs.json"))
    assert r["frozen_syndromics_la"]["rate"] < r["frozen_syndromics"]["rate"] < 0.7 * r["run2_calibration"]["rate"]
