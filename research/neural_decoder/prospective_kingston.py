"""Prospective neural test on REAL ibm_kingston data: learn everything on run 1 (18 Nov 2026,
job dbukcjvf8rhs73bmf9l0), decode all 30,000 shots of run 2 (19 Nov 2026, job dbvc3lv82tmc73dupa7g,
same qubits, after IBM's recalibration). Nothing is tuned on run 2.

The network trains only on simulated shots from the Syndromics model fitted on run 1, plus erratic-measure-qubit
bursts at run 1's rate; model selection uses run 1's real shots. Usage: python prospective_kingston.py [n_sim] [seeds]
Set NOBURST=1 for the ablation. Needs torch (CPU is fine)."""
import json, os, sys, time, numpy as np, torch, warnings
warnings.filterwarnings("ignore")
from syndromics import decodecheck as dc, ibm, ibmdemo, adaptive
from syndromics.model import fit_noise
torch.set_num_threads(4)
HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "data")
def load(tag):
    cfg = json.load(open(os.path.join(HERE, f"kingston_{tag}_config.json")))
    z = np.load(os.path.join(HERE, f"kingston_{tag}_rep.npz"))
    return np.unpackbits(z["rep"], axis=1)[:, : int(z["ncols"])], cfg
rep1, cfg1 = load("20261118"); rep2, cfg2 = load("20261119")
nd, R = cfg1["n_data"], cfg1["rounds"]; NA = nd - 1
c1 = ibm.rep_code_stim(nd, R, *ibmdemo._models(cfg1)[1]); c2 = ibm.rep_code_stim(nd, R, *ibmdemo._models(cfg2)[1])
def m2d(circ, meas):
    d, o = circ.compile_m2d_converter().convert(measurements=meas.astype(bool), separate_observables=True)
    return d.astype(np.uint8), o[:, 0].astype(np.uint8)
D1, O1 = m2d(c1, rep1); D2, O2 = m2d(c2, rep2)
pos, anc = adaptive._measure_detectors(c1)
def maxburst(Dx): return np.stack([Dx[:, pos == a].sum(1) for a in anc], 1).max(1)

NSIM = int(sys.argv[1]) if len(sys.argv) > 1 else 1500000
SEEDS = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0, 1, 2]
NOBURST = int(os.environ.get("NOBURST", "0"))
import torch.nn as nn
# network and training: identical to real_kingston.py
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


fit = fit_noise(c1, D1, sparsity=0.5, crosstalk=False); fitc = fit.calibrated_circuit(crosstalk=False)
m_syn = dc._matching(fitc); la = adaptive.fit_leakage_aware(c1, D1, O1)
def simulate(n, f, seed):
    rng = np.random.default_rng(seed); meas = fitc.compile_sampler(seed=seed).sample(n).astype(np.uint8)
    idx = np.where(rng.random(n) < f)[0]
    a = rng.integers(0, NA, len(idx)); r0 = rng.integers(0, R, len(idx)); L = rng.integers(3, R + 1, len(idx))
    for s, aa, rr, ll in zip(idx, a, r0, L):
        for r in range(rr, min(R, rr + ll)): meas[s, r * NA + aa] = rng.integers(0, 2)
    return m2d(c1, meas)
target = (maxburst(D1) >= 6).mean(); best = None
for f in (0.02, 0.04, 0.06, 0.08, 0.10, 0.12):
    gap = abs((maxburst(simulate(60000, f, 7)[0]) >= 6).mean() - target)
    if best is None or gap < best[0]: best = (gap, f)
f = 0.0 if NOBURST else best[1]
def hyb(Dx): return np.concatenate([Dx, m_syn.decode_batch(Dx)[:, 0].astype(np.uint8)[:, None]], 1)
Xv = hyb(D1); X2 = hyb(D2)
res = {"burst_fraction": f, "shots_run2": len(D2),
       "calibration_run2": int((dc._matching(c2).decode_batch(D2)[:, 0] != O2).sum()),
       "syndromics_frozen": int((m_syn.decode_batch(D2)[:, 0] != O2).sum()),
       "adaptive_frozen": int((la.decode_batch(D2)[:, 0] != O2).sum()), "neural_frozen": []}
for sd in SEEDS:
    t0 = time.time(); torch.manual_seed(sd)
    Xs, ys = simulate(NSIM, f, 11 + sd)
    net = train(hyb(Xs), ys, Xv, O1)
    e = int((nnpred(net, X2) != O2).sum()); res["neural_frozen"].append(e)
    print(f"seed {sd}: neural {e} errors ({e/len(D2)*100:.3f}%)  {time.time()-t0:.0f}s", flush=True)
print(json.dumps(res))
json.dump(res, open(f"results_prospective_kingston{'_noburst' if NOBURST else ''}.json", "w"), indent=1)
