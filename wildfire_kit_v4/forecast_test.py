"""
Fire forecast test (v2): predict where the fire will be K steps ahead from the current
fire map, burnt area, flammability, the (known) wind field and the terrain height.
Trains on 64 x 64 windows cut around the fire from the big map, compared against a
'fire stays where it is' baseline.

v4: --fire-model picks the engine that generates the TRAINING fires; the trained network is then
tested on fires from every engine in --test-fire-models (generalisation to other simulators).

Usage:
    python forecast_test.py --k 10 --train-eps 120 --epochs 15
    python forecast_test.py --k 10 --fire-model rothermel --test-fire-models native,pyrorl,haksar,rothermel
"""
import argparse, time
import numpy as np
import torch
import torch.nn as nn
from world import World, MAP_TYPES

WIN = 64


def make_data(n_eps, k, seed, per_ep=12, size=128, fire_model="native"):
    rng = np.random.default_rng(seed)
    X, Y, Y0 = [], [], []
    for _ in range(n_eps):
        w = World(size=size, rng=rng, map_type=MAP_TYPES[rng.integers(len(MAP_TYPES))], difficulty="medium",
                  fire_model=fire_model)
        w.reset()
        frames = []
        for _ in range(260 + k):
            w.step()
            frames.append((w.fire_map().copy(), w.burnt.copy(), w.wx.copy(), w.wy.copy()))
        for t in rng.choice(np.arange(20, 260), size=per_ep, replace=False):
            fire, burnt, wx, wy = frames[t]
            ys, xs = np.nonzero(fire > 0)
            if len(ys) == 0:
                continue
            j = rng.integers(len(ys))                      # window centred near a burning cell
            cy = int(np.clip(ys[j] + rng.integers(-16, 17), WIN // 2, size - WIN // 2))
            cx = int(np.clip(xs[j] + rng.integers(-16, 17), WIN // 2, size - WIN // 2))
            sl = (slice(cy - WIN // 2, cy + WIN // 2), slice(cx - WIN // 2, cx + WIN // 2))
            x = np.stack([fire[sl], burnt[sl].astype(np.float32), w._spread_mult[sl] / 2.0, wx[sl], wy[sl],
                          (w.elev[sl] - w.elev[sl].mean()) / 50.0])
            X.append(x.astype(np.float32))
            Y.append((frames[t + k][0][sl] > 0).astype(np.float32))
            Y0.append((fire[sl] > 0).astype(np.float32))
    return (torch.tensor(np.array(X)), torch.tensor(np.array(Y))[:, None], torch.tensor(np.array(Y0))[:, None])


class ForecastNet(nn.Module):
    """Small fully convolutional net; dilated layers see about 30 cells around each cell."""

    def __init__(self, c_in=6, h=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, h, 3, padding=1), nn.ReLU(),
            nn.Conv2d(h, h, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(h, h, 3, padding=4, dilation=4), nn.ReLU(),
            nn.Conv2d(h, h, 3, padding=8, dilation=8), nn.ReLU(),
            nn.Conv2d(h, h, 3, padding=1), nn.ReLU(),
            nn.Conv2d(h, 1, 1),
        )

    def forward(self, x):
        return self.net(x)


def scores(pred, target, prev):
    inter = (pred * target).sum().item()
    union = ((pred + target) > 0).sum().item()
    new = target * (1 - prev)
    return inter / max(1.0, union), (pred * new).sum().item() / max(1.0, new.sum().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--train-eps", type=int, default=120)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fire-model", default="native", help="engine that generates the training fires")
    ap.add_argument("--test-fire-models", default="", help="comma list of engines to test on afterwards")
    args = ap.parse_args()
    torch.manual_seed(0)
    t0 = time.time()
    Xtr, Ytr, _ = make_data(args.train_eps, args.k, seed=1, fire_model=args.fire_model)
    Xte, Yte, Y0te = make_data(25, args.k, seed=99, fire_model=args.fire_model)
    print(f"data: {len(Xtr)} train / {len(Xte)} test windows ({time.time() - t0:.0f}s), device {args.device}")
    dev = torch.device(args.device)
    model = ForecastNet().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    pos_w = torch.tensor(((1 - Ytr.mean()) / Ytr.mean()).item() ** 0.5, device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), 32):
            idx = perm[i:i + 32]
            loss = lossf(model(Xtr[idx].to(dev)), Ytr[idx].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = torch.cat([(torch.sigmoid(model(Xte[i:i + 64].to(dev))) > 0.5).float().cpu()
                              for i in range(0, len(Xte), 64)])
        iou, rec = scores(pred, Yte, Y0te)
        print(f"epoch {ep + 1:2d}  loss {loss.item():.3f}  test IoU {iou:.3f}  new-fire recall {rec:.3f}", flush=True)
    iou_p, rec_p = scores(Y0te, Yte, Y0te)
    print(f"\n'Fire stays where it is' baseline: IoU {iou_p:.3f}, new-fire recall {rec_p:.3f}")
    print(f"Forecast network:                  IoU {iou:.3f}, new-fire recall {rec:.3f}")
    if args.test_fire_models:
        print(f"\nTrained on '{args.fire_model}' fires; tested on fires from each engine (IoU / new-fire recall):")
        for fm in args.test_fire_models.split(","):
            X2, Y2, Y02 = make_data(25, args.k, seed=99, fire_model=fm)
            with torch.no_grad():
                p2 = torch.cat([(torch.sigmoid(model(X2[i:i + 64].to(dev))) > 0.5).float().cpu()
                                for i in range(0, len(X2), 64)])
            a, b = scores(p2, Y2, Y02)
            a0, b0 = scores(Y02, Y2, Y02)
            print(f"  {fm:10s} network IoU {a:.3f} recall {b:.3f}   | 'stays where it is' IoU {a0:.3f}")
    print(f"total time {time.time() - t0:.0f}s")
    torch.save(model.state_dict(), "forecast_net.pt")


if __name__ == "__main__":
    main()
