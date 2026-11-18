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
