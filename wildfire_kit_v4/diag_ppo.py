import argparse
import csv
import io
import numpy as np
import torch
from stable_baselines3 import PPO
from env import SingleDroneEnv
import train_ppo  # noqa: F401  (needed so the custom SmallCNN can be loaded with the model)

# torch can't read straight from a zip stream (which stable-baselines3 passes it);
# read the stream into memory first.
_orig_load = torch.load


def _load(f, *a, **k):
    if hasattr(f, "read") and not isinstance(f, io.BytesIO):
        f = io.BytesIO(f.read())
    return _orig_load(f, *a, **k)


torch.load = _load

# usage: python diag_ppo.py runs\1M_B2_test\model.zip --episodes 2 --maps mountain_valley   (hiter test)
#        python diag_ppo.py runs\1M_B2_test\model.zip                                        (polna analiza)
ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--maps", default="rolling_hills,mountain_valley,dense_forest,farmland,lake_shore,urban_interface")
ap.add_argument("--episodes", type=int, default=20)
ap.add_argument("--seed0", type=int, default=5000)
ap.add_argument("--difficulty", default="medium")
ap.add_argument("--csv", default="diag_episodes_B4.csv")
args = ap.parse_args()

# SmallCNN is saved by value from the cluster's Python (3.13); use the local class instead.
model = PPO.load(args.model, device="cpu", custom_objects=dict(
    policy_kwargs=dict(features_extractor_class=train_ppo.SmallCNN,
                       features_extractor_kwargs=dict(features_dim=128),
                       normalize_images=False)))


def episode(map_name, seed, deterministic):
    env = SingleDroneEnv(size=128, max_steps=300, map_types=map_name,
                         difficulty=args.difficulty, fire_models="rothermel")
    obs, _ = env.reset(seed=seed)
    core = env.core
    torch.manual_seed(seed)
    ret_pre, ret_post = 0.0, 0.0
    te, td, hover, ent, steps, done = None, None, 0, 0.0, 0, False
    while not done:
        obs_t, _ = model.policy.obs_to_tensor(obs)
        with torch.no_grad():
            dist = model.policy.get_distribution(obs_t)
            a = int(dist.get_actions(deterministic=deterministic).reshape(-1)[0])
            ent += float(dist.entropy().mean())
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        # nagrada tega koraka se steje v "pred", ce zaznave pred korakom se ni bilo
        if td is None:
            ret_pre += r
        else:
            ret_post += r
        steps += 1
        hover += int(a == 6)
        if te is None and core.world.fire_map().sum() > 0:
            te = core.t
        if td is None and core.detected_ever.any():
            td = core.t
    t_end = core.t
    return dict(map=map_name, seed=seed, det=int(deterministic),
                ret=ret_pre + ret_post, ret_pre=ret_pre, ret_post=ret_post,
                t_fire=te, t_detect=td, t_end=t_end,
                latency=(td - te) if (td is not None and te is not None) else None,
                post_per_step=(ret_post / (t_end - td)) if (td is not None and t_end > td) else None,
                hover=hover / max(steps, 1), entropy=ent / max(steps, 1),
                lost=core.lost_reason[0] or "no")


def rank(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in np.unique(x):
        m = x == v
        r[m] = r[m].mean()
    return r


def spearman(x, y):
    if len(x) < 5:
        return float("nan")
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def mse(v):
    v = np.asarray([u for u in v if u is not None], float)
    if len(v) == 0:
        return "n/a"
    se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
    return f"{v.mean():.2f}+-{se:.2f}"


rows = []
maps = args.maps.split(",")
for det in (True, False):
    for m in maps:
        for i in range(args.episodes):
            rows.append(episode(m, args.seed0 + i, det))
        print(f"koncano: det={det} map={m}", flush=True)

with open(args.csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

for det in (1, 0):
    print(f"\n===== {'DETERMINISTICNA' if det else 'STOHASTICNA'} politika (n={args.episodes} na mapo, SE = std/sqrt(n)) =====")
    print(f"{'mapa':<16}{'return':>13}{'zaznan':>8}{'latenca(med)':>14}{'ret_pre':>13}{'ret_post':>13}{'post/korak':>13}{'HOVER':>7}{'entr.':>7}")
    for m in maps + ["VSE"]:
        sel = [r for r in rows if r["det"] == det and (m == "VSE" or r["map"] == m)]
        lat = [r["latency"] for r in sel if r["latency"] is not None]
        n_det = sum(r["t_detect"] is not None for r in sel)
        n_fire = sum(r["t_fire"] is not None for r in sel)
        print(f"{m:<16}{mse([r['ret'] for r in sel]):>13}{f'{n_det}/{n_fire}':>8}"
              f"{(f'{np.median(lat):.0f}' if lat else 'n/a'):>14}"
              f"{mse([r['ret_pre'] for r in sel]):>13}{mse([r['ret_post'] for r in sel]):>13}"
              f"{mse([r['post_per_step'] for r in sel]):>13}"
              f"{100 * np.mean([r['hover'] for r in sel]):>6.1f}%{np.mean([r['entropy'] for r in sel]):>7.3f}")
    sel = [r for r in rows if r["det"] == det and r["latency"] is not None]
    print(f"Spearman(return, latenca zaznave), n={len(sel)}: {spearman([r['ret'] for r in sel], [r['latency'] for r in sel]):.2f}")
    s2 = [r for r in sel if r["post_per_step"] is not None]
    print(f"Spearman(return po zaznavi na korak, latenca), n={len(s2)}: "
          f"{spearman([r['post_per_step'] for r in s2], [r['latency'] for r in s2]):.2f}")
    print("izgubljeni droni:", sum(r["lost"] != "no" for r in rows if r["det"] == det), "epizod")

print(f"\nPo-epizodni podatki: {args.csv}")