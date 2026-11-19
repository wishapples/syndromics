"""Use a Syndromics noise model inside NVIDIA CUDA-Q QEC decoders.

The Syndromics model is a Stim circuit, so its detector error model (DEM) goes straight into CUDA-Q QEC
(`cudaq_qec.get_decoder(name, dem_text, ...)`). This script repeats the prospective test on real ibm_kingston data:
learn on run 1 (18 Nov 2026), decode all of run 2 (19 Nov 2026) with CUDA-Q QEC decoders given either IBM's
calibration or the frozen Syndromics model. Needs: pip install cudaq-qec-cu12 (or -cu13) and this package.
Runs on CPU; on an NVIDIA GPU the GPU decoders (e.g. nv-qldpc-decoder) can be used the same way."""
import json, os, sys, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
import cudaq_qec as qec
from syndromics import decodecheck as dc, ibm, ibmdemo
from syndromics.model import fit_noise

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "data")


def load(tag):
    cfg = json.load(open(os.path.join(HERE, f"kingston_{tag}_config.json")))
    z = np.load(os.path.join(HERE, f"kingston_{tag}_rep.npz"))
    return np.unpackbits(z["rep"], axis=1)[:, : int(z["ncols"])], cfg


def dem_text(circ):
    return str(circ.detector_error_model(decompose_errors=True))


def cudaq_errors(name, circ, det, obs, **kw):
    dec = qec.get_decoder(name, dem_text(circ), output="observables", **kw)
    t = time.time()
    res = dec.decode_batch(det.astype(np.uint8).tolist())
    pred = np.array([int(round(r.result[0])) for r in res], dtype=np.uint8)
    return int((pred != obs).sum()), time.time() - t


if __name__ == "__main__":
    rep1, cfg1 = load("20261118"); rep2, cfg2 = load("20261119")
    nd, R = cfg1["n_data"], cfg1["rounds"]
    c1 = ibm.rep_code_stim(nd, R, *ibmdemo._models(cfg1)[1])
    c2 = ibm.rep_code_stim(nd, R, *ibmdemo._models(cfg2)[1])
    D1, _ = dc._detectors(c1, rep1); D2, O2 = dc._detectors(c2, rep2)
    frozen = fit_noise(c1, D1.astype(np.uint8), sparsity=0.5, crosstalk=False).calibrated_circuit(crosstalk=False)
    names = sys.argv[1:] or ["pymatching"]
    print(f"Run 2: {len(D2):,} shots, decoded with CUDA-Q QEC {', '.join(names)}")
    out = {}
    for name in names:
        for lab, circ in (("IBM calibration (run 2)", c2), ("Syndromics model frozen from run 1", frozen)):
            e, s = cudaq_errors(name, circ, D2, O2)
            out[f"{name} | {lab}"] = e
            print(f"  {name:12s} {lab:38s} {e:4d} errors = {e/len(D2)*100:.3f}%   ({s:.1f} s)")
    json.dump(out, open("results_cudaq_qec.json", "w"), indent=1)
