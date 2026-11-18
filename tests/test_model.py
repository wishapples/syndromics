import os

import numpy as np
import pytest
import stim

from syndromics.model import fit_noise
from syndromics.sim import nominal_circuit, real_chip_circuit, sample


def test_general_fit_finds_bad_qubit_and_readout():
    real, truth = real_chip_circuit(3, 3, 0.004, seed=1, n_xtalk=0)
    det, _ = sample(real, 300_000, seed=2)
    fit = fit_noise(nominal_circuit(3, 3, 0.004), det, crosstalk=False)
    flagged = {f.component for f in fit.outliers()}
    for q in truth["bad_data_qubits"]:
        assert ("qubit", q) in flagged
    for q in truth["bad_readout_qubits"]:
        assert ("spam", q) in flagged


def test_clean_chip_general_fit_quiet():
    nom = nominal_circuit(3, 3, 0.004)
    det, _ = sample(nom, 300_000, seed=5)
    assert fit_noise(nom, det).outliers() == []


def test_calibrated_model_decodes_better():
    pymatching = pytest.importorskip("pymatching")
    real, _ = real_chip_circuit(5, 5, 0.004, seed=7)
    nom = nominal_circuit(5, 5, 0.004)
    det, _ = sample(real, 200_000, seed=3)
    test_det, test_obs = sample(real, 200_000, seed=9)
    fit = fit_noise(nom, det)

    def ler(c):
        m = pymatching.Matching.from_detector_error_model(
            c.detector_error_model(decompose_errors=True, ignore_decomposition_failures=True))
        return (m.decode_batch(test_det)[:, 0] != test_obs).mean()

    assert ler(fit.calibrated_circuit()) < 0.85 * ler(nom)


def test_google_loader(tmp_path):
    pytest.importorskip("pymatching")
    from syndromics.google import find_experiments, run
    real, _ = real_chip_circuit(3, 4, 0.004, seed=3)
    d = tmp_path / "d3_at_q4_4" / "Z" / "r04"
    d.mkdir(parents=True)
    nominal_circuit(3, 4, 0.004).to_file(str(d / "circuit_noisy_si1000.stim"))
    det, obs = real.compile_detector_sampler(seed=1).sample(40_000, separate_observables=True)
    stim.write_shot_data_file(data=det, path=str(d / "detection_events.b8"), format="b8",
                              num_detectors=real.num_detectors)
    stim.write_shot_data_file(data=obs, path=str(d / "obs_flips_actual.01"), format="01",
                              num_observables=1)
    exps = find_experiments(str(tmp_path))
    assert len(exps) == 1 and exps[0].patch == "d3_at_q4_4" and exps[0].rounds == 4
    summary = run(str(tmp_path), out=str(tmp_path / "out"), target_rounds=4)
    assert "decoding" in summary
    assert os.path.exists(tmp_path / "out" / "components.csv")


def test_ibm_circuits_agree():
    pytest.importorskip("qiskit_aer")
    from qiskit_aer.primitives import SamplerV2
    from syndromics import ibm
    n, r = 4, 3
    z = {k: 0.0 for k in range(2 * n - 1)}
    g = {(k, k + 1): 0.0 for k in range(2 * n - 2)}
    nom = ibm.rep_code_stim(n, r, z, g, z)
    res = SamplerV2().run([ibm.rep_code_qiskit(n, r)], shots=100).result()[0]
    cols = [getattr(res.data, f"r{t}").to_bool_array(order="little") for t in range(r)]
    cols.append(res.data.final.to_bool_array(order="little"))
    det = nom.compile_m2d_converter().convert(measurements=np.hstack(cols), separate_observables=False)
    assert det.sum() == 0
