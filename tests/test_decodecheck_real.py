"""Regression test on real ibm_kingston data (job dbukcjvf8rhs73bmf9l0, 18 Nov 2026, 9-qubit line, 30,000 shots).

Guards two things found on that data: (1) the Syndromics model must beat the device calibration on held-out
shots, and (2) fitted crosstalk terms must not reach the decoder (they once tripled the logical error)."""
import json
import os

import numpy as np

from syndromics import decodecheck, ibmdemo

HERE = os.path.join(os.path.dirname(__file__), "data")


def load():
    z = np.load(os.path.join(HERE, "kingston_20261118_rep.npz"))
    rep = np.unpackbits(z["rep"], axis=1)[:, : int(z["ncols"])]
    cfg = json.load(open(os.path.join(HERE, "kingston_20261118_config.json")))
    return rep, cfg


def test_syndromics_beats_calibration_on_real_kingston():
    rep, cfg = load()
    uni, cal = ibmdemo._models(cfg)
    res = decodecheck.compare(rep, cfg["n_data"], cfg["rounds"], uni, cal, leakage_aware=False)
    m = res["mean"]
    assert m["syndromics"] < 0.65 * m["calibration"]
    assert all(t["syndromics"] < t["calibration"] for t in res["tests"])
    assert abs(m["syndromics"] - m["reweighting"]) < 0.0008     # on par with the literature baseline


def test_fault_localised_at_q128_q129():
    rep, cfg = load()
    uni, cal = ibmdemo._models(cfg)
    ch = decodecheck.changed_components(rep, cfg["n_data"], cfg["rounds"], uni, cal, cfg["line"])
    assert ch[0]["component"].startswith("q129-q128")


def test_leakage_aware_halves_errors_again_on_real_kingston():
    """Real shots where one measure qubit behaves randomly cause about half the failures; flagging them per shot fixes most."""
    rep, cfg = load()
    uni, cal = ibmdemo._models(cfg)
    res = decodecheck.compare(rep, cfg["n_data"], cfg["rounds"], uni, cal, splits=2)
    m = res["mean"]
    assert m["syndromics_la"] < 0.7 * m["syndromics"]
    assert all(t["syndromics_la"] < t["syndromics"] for t in res["tests"])


def test_bursts_far_above_pauli_model_on_real_kingston():
    from syndromics import adaptive, ibm
    rep, cfg = load()
    _, cal = ibmdemo._models(cfg)
    c = ibm.rep_code_stim(cfg["n_data"], cfg["rounds"], *cal)
    D, _ = decodecheck._detectors(c, rep)
    bs = adaptive.burst_stats(c, D)
    assert bs["real"] > 20 * bs["model"]


def load_run2():
    z = np.load(os.path.join(HERE, "kingston_20261119_rep.npz"))
    rep = np.unpackbits(z["rep"], axis=1)[:, : int(z["ncols"])]
    cfg = json.load(open(os.path.join(HERE, "kingston_20261119_config.json")))
    return rep, cfg


def test_prospective_second_day_on_real_kingston():
    """Run 2 (job dbvc3lv82tmc73dupa7g, 19 Nov 2026, same qubits, after IBM recalibrated): everything frozen from
    run 1 must beat run 2's fresh calibration; the q128-q129 fault persists; no spurious q117-q105 'change'."""
    from syndromics import adaptive
    rep1, cfg1 = load(); rep2, cfg2 = load_run2()
    r = adaptive.compare_runs(rep1, cfg1, rep2, cfg2, ibmdemo._models)
    cal = r["run2_calibration"]["rate"]
    assert r["frozen_syndromics"]["rate"] < 0.6 * cal
    assert r["frozen_syndromics_la"]["rate"] < 0.25 * cal
    names = [c["component"] for c in r["component_changes"]]
    assert "q117-q105 two-qubit gate" not in names          # was an artefact of the unregularised fit (0.5.1)
    assert "q129-q128 two-qubit gate" not in names          # the fault is stable across days
    uni, cal2 = ibmdemo._models(cfg2)
    ch = decodecheck.changed_components(rep2, cfg2["n_data"], cfg2["rounds"], uni, cal2, cfg2["line"])
    assert ch[0]["component"] == "q129-q128 two-qubit gate" and ch[0]["ratio"] > 8


def test_transfer_to_a_different_circuit_on_real_kingston():
    """Run 3 (job dbvdl8ups3fc73bkad70): distance-3, 20-round code on q108-q128, decoded with the model frozen from
    run 2 (distance 5, 10 rounds). With no run-3 training data it must beat the calibration and match correlation
    re-weighting trained on 15,000 run-3 shots."""
    from syndromics import adaptive
    rep2, cfg2 = load_run2()
    z = np.load(os.path.join(HERE, "kingston_20261119b_rep.npz"))
    rep3 = np.unpackbits(z["rep"], axis=1)[:, : int(z["ncols"])]
    cfg3 = json.load(open(os.path.join(HERE, "kingston_20261119b_config.json")))
    r = adaptive.transfer_runs(rep2, cfg2, rep3, cfg3, ibmdemo._models, train_sizes=(3000, None))
    cal = r["run2_calibration"]["errors"]; syn = r["frozen_syndromics"]["errors"]
    rw3k, rw15k = (x["errors"] for x in r["reweighting_on_run2"])
    assert syn < 0.6 * cal
    assert syn <= 1.1 * rw15k and syn < rw3k
