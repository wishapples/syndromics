"""
train.py - Train a chip-adaptive neural decoder and benchmark it.

Decoders compared on the same held-out shots from the simulated chip:
  1. MWPM-uniform : PyMatching with the nominal (uniform) noise model   <- industry default
  2. MWPM-oracle  : PyMatching with the TRUE hidden noise model          <- upper reference (needs perfect characterisation)
  3. NN           : neural network trained only on raw syndrome data from the chip
  4. NN+MWPM      : hybrid - NN sees syndrome + MWPM-uniform's answer and learns when to override it

Usage: python train.py --d 5 --p 0.004 --train 3000000
"""
import argparse, json, time
import numpy as np, torch, torch.nn as nn, pymatching
import chip

torch.set_num_threads(max(1, torch.get_num_threads()))


class Decoder(nn.Module):
    def __init__(self, n_in, width=512, depth=4):
        super().__init__()
        layers, w = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(w, width), nn.LayerNorm(width), nn.GELU()]
            w = width
        layers += [nn.Linear(w, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit(X, y, Xv, yv, epochs, lr=2e-3, bs=4096, width=512, depth=4, log=""):
    model = Decoder(X.shape[1], width, depth)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * (len(X) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps)
    Xt, yt = torch.from_numpy(X).float(), torch.from_numpy(y).float()
    Xvt = torch.from_numpy(Xv).float()
    lossf = nn.BCEWithLogitsLoss()
    best, best_state = 1.0, None
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xt)); t0 = time.time()
        for i in range(0, len(Xt) - bs + 1, bs):
            idx = perm[i:i + bs]
            loss = lossf(model(Xt[idx]), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        ler = (predict(model, Xvt) != yv).mean()
        if ler < best:
            best, best_state = ler, {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  {log} epoch {ep+1}/{epochs} loss {loss.item():.4f} val LER {ler:.5f} ({time.time()-t0:.0f}s)", flush=True)
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model, Xt):
    model.eval()
    out = [(model(Xt[i:i + 65536]) > 0).numpy() for i in range(0, len(Xt), 65536)]
    return np.concatenate(out).astype(np.uint8)


def wilson(k, n):
    p = k / n; z = 1.96
    c = (p + z*z/(2*n)) / (1 + z*z/n); h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--p", type=float, default=0.004)
    ap.add_argument("--train", type=int, default=3_000_000)
    ap.add_argument("--test", type=int, default=1_000_000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--out", default="results.json")
    a = ap.parse_args()
    r = a.rounds or a.d
    torch.manual_seed(0)

    real, desc = chip.real_chip_circuit(a.d, r, a.p)
    nominal = chip.nominal_circuit(a.d, r, a.p)
    print("Hidden chip noise:", desc)

    mU = pymatching.Matching.from_detector_error_model(nominal.detector_error_model(decompose_errors=True))
    mO = pymatching.Matching.from_detector_error_model(real.detector_error_model(decompose_errors=True))

    X, y = chip.sample(real, a.train, seed=1)
    Xte, yte = chip.sample(real, a.test, seed=2)
    nv = 200_000
    Xv, yv = X[-nv:], y[-nv:]; X, y = X[:-nv], y[:-nv]

    t = time.time(); pU = mU.decode_batch(Xte)[:, 0]; tU = (time.time() - t) / a.test
    pO = mO.decode_batch(Xte)[:, 0]
    pU_tr, pU_v = mU.decode_batch(X)[:, 0], mU.decode_batch(Xv)[:, 0]

    print("Training NN (syndrome only)...")
    nn1 = fit(X, y, Xv, yv, a.epochs, log="NN")
    print("Training NN+MWPM hybrid...")
    H, Hv, Hte = (np.hstack([A, p[:, None]]) for A, p in ((X, pU_tr), (Xv, pU_v), (Xte, pU)))
    nn2 = fit(H, y, Hv, yv, a.epochs, log="Hybrid")

    Xte_t, Hte_t = torch.from_numpy(Xte).float(), torch.from_numpy(Hte).float()
    t = time.time(); p1 = predict(nn1, Xte_t); tN = (time.time() - t) / a.test
    p2 = predict(nn2, Hte_t)

    # single-shot latency (batch of 1), CPU, unoptimised PyTorch
    x1 = Xte_t[:1]; nn1.eval()
    with torch.no_grad():
        for _ in range(50): nn1(x1)
        t = time.time()
        for _ in range(500): nn1(x1)
    lat = (time.time() - t) / 500

    res = {"chip": desc, "train_shots": len(X), "test_shots": a.test, "decoders": {}}
    for name, pr in [("MWPM-uniform", pU), ("MWPM-oracle", pO), ("NN", p1), ("NN+MWPM", p2)]:
        k = int((pr != yte).sum()); lo, hi = wilson(k, a.test)
        res["decoders"][name] = {"LER": k / a.test, "ci95": [lo, hi]}
    base = res["decoders"]["MWPM-uniform"]["LER"]
    for v in res["decoders"].values():
        v["vs_uniform"] = f"{(1 - v['LER'] / base) * 100:+.1f}% fewer errors"
    res["throughput_us_per_shot"] = {"MWPM-uniform": tU * 1e6, "NN_batched": tN * 1e6}
    res["nn_single_shot_latency_us_cpu"] = lat * 1e6
    res["nn_params"] = sum(p.numel() for p in nn1.parameters())
    print(json.dumps(res, indent=2))
    json.dump(res, open(a.out, "w"), indent=2)
    torch.save(nn2.state_dict(), a.out.replace(".json", "_hybrid.pt"))


if __name__ == "__main__":
    main()
