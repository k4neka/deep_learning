import argparse
import csv
import numpy as np
from env import SingleDroneEnv

try:
    from baselines import RandomPolicy, Lawnmower, Greedy
except ImportError:
    from baseline import RandomPolicy, Lawnmower, Greedy

# usage: python diag_baselines.py --episodes 2 --maps mountain_valley                 (hiter test)
#        python diag_baselines.py --ppo-csv diag_B2_full.csv                           (polna primerjava)
ap = argparse.ArgumentParser()
ap.add_argument("--maps", default="rolling_hills,mountain_valley,dense_forest,farmland,lake_shore,urban_interface")
ap.add_argument("--episodes", type=int, default=20)
ap.add_argument("--seed0", type=int, default=5000)
ap.add_argument("--difficulty", default="medium")
ap.add_argument("--csv", default="diag_baselines_B3.csv")
ap.add_argument("--ppo-csv", default=None, help="diag_B2_full.csv, za skupno primerjalno tabelo")
args = ap.parse_args()

HOVER = 6  # N, S, E, W, UP, DOWN, HOVER (enak vrstni red kot v diag_ppo.py)


class Hover:
    def reset(self, core):
        pass

    def act(self, core, i):
        return HOVER


def episode(policy, name, map_name, seed):
    env = SingleDroneEnv(size=128, max_steps=300, map_types=map_name,
                         difficulty=args.difficulty, fire_models="rothermel")
    env.reset(seed=seed)
    core = env.core
    policy.reset(core)
    ret_pre, ret_post = 0.0, 0.0
    te, td, hover, steps, done = None, None, 0, 0, False
    while not done:
        a = int(policy.act(core, 0))
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
        if td is None:
            ret_pre += r
        else:
            ret_post += r
        steps += 1
        hover += int(a == HOVER)
        if te is None and core.world.fire_map().sum() > 0:
            te = core.t
        if td is None and core.detected_ever.any():
            td = core.t
    t_end = core.t
    return dict(policy=name, map=map_name, seed=seed,
                ret=ret_pre + ret_post, ret_pre=ret_pre, ret_post=ret_post,
                t_fire=te, t_detect=td, t_end=t_end,
                latency=(td - te) if (td is not None and te is not None) else None,
                hover=hover / max(steps, 1), lost=core.lost_reason[0] or "no")


maps = args.maps.split(",")
rows = []
for name, make in [("random", lambda: RandomPolicy(seed=0)), ("hover", Hover),
                   ("lawnmower", Lawnmower), ("greedy", Greedy)]:
    pol = make()
    for m in maps:
        for i in range(args.episodes):
            rows.append(episode(pol, name, m, args.seed0 + i))
        print(f"koncano: {name} map={m}", flush=True)

with open(args.csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)


def num(x):
    return None if x in ("", None) else float(x)


if args.ppo_csv:
    with open(args.ppo_csv, newline="") as f:
        for d in csv.DictReader(f):
            rows.append(dict(policy="ppo_det" if d["det"] == "1" else "ppo_stoch", map=d["map"],
                             seed=int(d["seed"]), ret=float(d["ret"]), t_detect=num(d["t_detect"]),
                             latency=num(d["latency"]), lost=d["lost"]))


def mse(v):
    v = np.asarray(v, float)
    se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
    return f"{v.mean():.2f}+-{se:.2f}"


names = []
for r in rows:
    if r["policy"] not in names:
        names.append(r["policy"])

print("\n===== SKUPNO (vse mape; SE = std/sqrt(n); latenca zaznane epizode, nezaznane stejejo kot 300) =====")
print(f"{'politika':<11}{'n':>5}{'return':>14}{'zaznan':>9}{'latenca(med)':>14}{'cenz.latenca':>15}{'izgub.':>8}")
for n in names:
    sel = [r for r in rows if r["policy"] == n]
    lat = [r["latency"] for r in sel if r["latency"] is not None]
    cens = [r["latency"] if r["latency"] is not None else 300 for r in sel]
    print(f"{n:<11}{len(sel):>5}{mse([r['ret'] for r in sel]):>14}"
          f"{sum(r['latency'] is not None for r in sel):>9}"
          f"{(f'{np.median(lat):.0f}' if lat else 'n/a'):>14}{mse(cens):>15}"
          f"{sum(r['lost'] != 'no' for r in sel):>8}")

print("\n===== DELEZ ZAZNANIH EPIZOD PO MAPAH =====")
print(f"{'mapa':<16}" + "".join(f"{n:>11}" for n in names))
for m in maps:
    cells = []
    for n in names:
        sel = [r for r in rows if r["policy"] == n and r["map"] == m]
        cells.append(f"{sum(r['latency'] is not None for r in sel)}/{len(sel)}")
    print(f"{m:<16}" + "".join(f"{c:>11}" for c in cells))

print(f"\nPo-epizodni podatki baselinov: {args.csv}")