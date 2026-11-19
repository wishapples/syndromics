"""Neural hybrid decoder on REAL ibm_kingston data (job dbukcjvf8rhs73bmf9l0).

Trains only on simulated shots: the Syndromics noise model fitted on the training half, plus erratic-measure-qubit
bursts at the rate seen in the training half. Tested on the held-out real half. Input = detectors + the Syndromics
matching decoder's prediction. Usage: python real_kingston.py [n_sim_shots] [splits e.g. 0,1,2,3,4,5]
Set NOBURST=1 for the ablation without bursts. Needs: pip install torch (CPU is fine, about 1 minute per split).
"""
import json, sys, time, numpy as np, torch, torch.nn as nn, warnings
warnings.filterwarnings("ignore")
from syndromics import decodecheck as dc, ibm, ibmdemo, adaptive
from syndromics.model import fit_noise
torch.set_num_threads(2); torch.manual_seed(0)
import os
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "data")
cfg = json.load(open(os.path.join(HERE, "kingston_20261118_config.json")))
_z = np.load(os.path.join(HERE, "kingston_20261118_rep.npz")); rep = np.unpackbits(_z["rep"], axis=1)[:, : int(_z["ncols"])]
nd, R = cfg["n_data"], cfg["rounds"]; NA = nd - 1
uni, cal = ibmdemo._models(cfg)
cal_c = ibm.rep_code_stim(nd, R, *cal)
conv = cal_c.compile_m2d_converter()
def m2d(meas):
    d, o = conv.convert(measurements=meas.astype(bool), separate_observables=True)
    return d.astype(np.uint8), o[:, 0].astype(np.uint8)
D, O = m2d(rep)
pos, anc = adaptive._measure_detectors(cal_c)
def maxburst(Dx): return np.stack([Dx[:, pos == a].sum(1) for a in anc], 1).max(1)

def simulate(fitc, n, f, seed):
    """measurements from the fitted model, plus erratic-measure-qubit bursts in a fraction f of shots"""
    rng = np.random.default_rng(seed)
    meas = fitc.compile_sampler(seed=seed).sample(n).astype(np.uint8)
    k = rng.random(n) < f
    idx = np.where(k)[0]
    a = rng.integers(0, NA, len(idx)); r0 = rng.integers(0, R, len(idx)); L = rng.integers(3, R + 1, len(idx))
    for s, aa, rr, ll in zip(idx, a, r0, L):
        for r in range(rr, min(R, rr + ll)):
            meas[s, r * NA + aa] = rng.integers(0, 2)
    return m2d(meas)

class Net(nn.Module):
    def __init__(s, n_in, w=256, d=3):
        super().__init__(); L = []; k = n_in
        for _ in range(d): L += [nn.Linear(k, w), nn.LayerNorm(w), nn.GELU()]; k = w
        L += [nn.Linear(k, 1)]; s.f = nn.Sequential(*L)
    def forward(s, x): return s.f(x).squeeze(-1)

def train(X, y, Xv, yv, epochs=6, bs=4096, lr=2e-3):
    m = Net(X.shape[1]); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * (len(X) // bs); sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    Xt, yt = torch.from_numpy(X).float(), torch.from_numpy(y).float(); Xvt = torch.from_numpy(Xv).float()
    lf = nn.BCEWithLogitsLoss(); best = (9, None)
    for ep in range(epochs):
        m.train(); p = torch.randperm(len(Xt))
        for i in range(0, len(Xt) - bs + 1, bs):
            j = p[i:i + bs]; loss = lf(m(Xt[j]), yt[j]); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        with torch.no_grad():
            m.eval(); e = ((m(Xvt) > 0).numpy().astype(np.uint8) != yv).mean()
        if e < best[0]: best = (e, {k: v.clone() for k, v in m.state_dict().items()})
    m.load_state_dict(best[1]); return m

def nnpred(m, X):
    with torch.no_grad():
        m.eval(); return (m(torch.from_numpy(X).float()) > 0).numpy().astype(np.uint8)

import os
NOBURST = int(os.environ.get("NOBURST", "0"))
NSIM = int(sys.argv[1]) if len(sys.argv) > 1 else 1500000
rng = np.random.default_rng(100)
splits = [(lambda p: (p[:15000], p[15000:]))(rng.permutation(30000)) for _ in range(5)] + [(np.arange(15000), np.arange(15000, 30000))]
which = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else range(6)
rows = []
for si in which:
    tr, te = splits[si]; t0 = time.time()
    fit = fit_noise(cal_c, D[tr].astype(np.uint8), sparsity=0.5, crosstalk=False)
    fitc = fit.calibrated_circuit(crosstalk=False)
    m_syn = dc._matching(fitc)
    la = adaptive.fit_leakage_aware(cal_c, D[tr], O[tr])
    # burst rate f chosen on the TRAINING half: match P(maxburst >= 6)
    target = (maxburst(D[tr]) >= 6).mean(); best = None
    for f in (0.02, 0.04, 0.06, 0.08, 0.10, 0.12):
        Ds, _ = simulate(fitc, 60000, f, 7)
        gap = abs((maxburst(Ds) >= 6).mean() - target)
        if best is None or gap < best[0]: best = (gap, f)
    f = best[1] if NOBURST == 0 else 0.0
    Xs, ys = simulate(fitc, NSIM, f, 11 + si)
    # hybrid input: detectors + Syndromics-MWPM prediction
    ps = m_syn.decode_batch(Xs)[:, 0].astype(np.uint8)
    Xh = np.concatenate([Xs, ps[:, None]], 1)
    # validation = real TRAINING half (model never sees the test half)
    pv = m_syn.decode_batch(D[tr])[:, 0].astype(np.uint8)
    Xv = np.concatenate([D[tr], pv[:, None]], 1)
    net = train(Xh, ys, Xv, O[tr])
    pt = m_syn.decode_batch(D[te])[:, 0].astype(np.uint8)
    Xt = np.concatenate([D[te], pt[:, None]], 1)
    r = {"split": si, "f": f,
         "calibration": int((dc._matching(cal_c).decode_batch(D[te])[:, 0] != O[te]).sum()),
         "syndromics": int((pt != O[te]).sum()),
         "adaptive": int((la.decode_batch(D[te])[:, 0] != O[te]).sum()),
         "neural_hybrid": int((nnpred(net, Xt) != O[te]).sum())}
    rows.append(r); print(r, f"{time.time()-t0:.0f}s", flush=True)
json.dump(rows, open(f"nn_rows_nb{NOBURST}_{'_'.join(map(str, which))}.json", "w"))
tot = {k: sum(r[k] for r in rows) for k in ("calibration", "syndromics", "adaptive", "neural_hybrid")}
n = 15000 * len(rows)
print({k: f"{v/n*100:.3f}%" for k, v in tot.items()})
