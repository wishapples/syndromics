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
