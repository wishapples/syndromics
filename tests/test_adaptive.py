"""Drift tracking on simulated drift with a known answer, starting from the real Kingston fit."""
import json
import os

import numpy as np

from syndromics import adaptive, decodecheck, ibm, ibmdemo
from syndromics.model import fit_noise, rebuild

HERE = os.path.join(os.path.dirname(__file__), "data")


def _truth():
    z = np.load(os.path.join(HERE, "kingston_20261118_rep.npz"))
    rep = np.unpackbits(z["rep"], axis=1)[:, : int(z["ncols"])]
    cfg = json.load(open(os.path.join(HERE, "kingston_20261118_config.json")))
    _, cal = ibmdemo._models(cfg)
    c = ibm.rep_code_stim(cfg["n_data"], cfg["rounds"], *cal)
    D, _ = decodecheck._detectors(c, rep)
    f = fit_noise(c, D.astype(np.uint8), sparsity=0.5, crosstalk=False)
    sc = {x.component: max(0.0, x.scale) for x in f.fits if x.component[0] != "xtalk"}
    return c, f, sc, cfg["line"]


def _windows(f, sc, mult, seed, W=5000):
    def circ(m):
        s = dict(sc)
        for k, v in m.items():
            s[k] = s.get(k, 1.0) * v
        return rebuild(f.circuit, lambda x: s.get(x, 1.0))
    a = circ({}).compile_detector_sampler(seed=seed).sample(3 * W)
    b = circ(mult).compile_detector_sampler(seed=seed + 100).sample(3 * W)
    d = np.concatenate([a, b])
    return [d[i * W:(i + 1) * W] for i in range(6)]


def test_gate_drift_detected_and_no_false_alarm():
    c, f, sc, line = _truth()
    _, alerts = adaptive.track_drift(c, _windows(f, sc, {("gate", 4, 5): 4.0}, 11), line)
    assert [a["component"] for a in alerts] == ["q108-q109 two-qubit gate"]
    assert alerts[0]["first_seen"] == 4
    _, none = adaptive.track_drift(c, _windows(f, sc, {}, 11), line)
    assert none == []


def test_transfer_to_shorter_code_on_part_of_the_line():
    """A model learned on the d=5 line must carry over to a d=3 code with more rounds on the bad end of the line
    (simulated from the model fitted on real Kingston data, so the true answer is known)."""
    import json, os
    from syndromics import ibm, ibmdemo
    from syndromics.model import rebuild
    here = os.path.join(os.path.dirname(__file__), "data")
    cfg = json.load(open(os.path.join(here, "kingston_20261118_config.json")))
    z = np.load(os.path.join(here, "kingston_20261118_rep.npz")); rep = np.unpackbits(z["rep"], axis=1)[:, : int(z["ncols"])]
    _, cal = ibmdemo._models(cfg); c1 = ibm.rep_code_stim(5, 10, *cal)
    D, _ = decodecheck._detectors(c1, rep)
    f = fit_noise(c1, D.astype(np.uint8), sparsity=0.5, crosstalk=False)
    sc = {x.component: max(0, x.scale) for x in f.fits if x.component[0] != "xtalk"}

    def truth(nd, R, off):
        cb = ibm.rep_code_stim(nd, R, *adaptive._slice_model(cal, off, 2 * nd - 1))
        return rebuild(cb, lambda c: sc.get((c[0],) + tuple(v + off for v in c[1:]), 1.0))
    rep_a = truth(5, 10, 0).compile_sampler(seed=1).sample(20000).astype(np.uint8)
    rep_b = truth(3, 20, 4).compile_sampler(seed=2).sample(20000).astype(np.uint8)
    cfg_b = dict(cfg, line=cfg["line"][4:9], n_data=3, rounds=20)
    for key, m in (("calibrated", cal), ("uniform", ibmdemo._models(cfg)[0])):
        ro, g2, idle = adaptive._slice_model(m, 4, 5)
        cfg_b[key] = {"readout": {str(k): v for k, v in ro.items()}, "gate2": {f"{x}-{y}": v for (x, y), v in g2.items()},
                      "idle": {str(k): v for k, v in idle.items()}}
    r = adaptive.transfer_runs(rep_a, cfg, rep_b, cfg_b, ibmdemo._models, train_sizes=(1000, None))
    assert r["offset"] == 4
    assert r["frozen_syndromics"]["rate"] < 0.4 * r["run2_calibration"]["rate"]
    assert r["frozen_syndromics"]["rate"] <= 1.5 * r["syndromics_on_run2"]["rate"] + 0.0005
