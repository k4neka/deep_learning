"""
Compare the fire engines: same maps, same wind, same ignition points, different fire-spread models.

Produces (in --out, default runs/compare):
  side_by_side.png     the same mission under each engine (one row per map type)
  growth.png           burnt + burning area over time per engine (median and 25-75 % band)
  results.json         growth curves + baseline scores per engine
and prints two tables: fire growth (the calibration check) and baseline strategy scores per engine.

Usage:
    python compare_simulators.py                          # ~10-15 min on a laptop
    python compare_simulators.py --episodes 4 --quick     # ~3 min smoke test
"""
import argparse
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from world import World, MAP_TYPES, TERRAIN_NAMES
from fire_models import FIRE_MODELS
from maps_gallery import draw_world, TERRAIN_COLORS
from baselines import evaluate, RandomPolicy, Lawnmower, Greedy

LABEL = {"native": "native (kit v3)", "pyrorl": "Julian & Kochenderfer / PyroRL",
         "haksar": "Haksar & Schwager lattice", "rothermel": "Rothermel / MITRE SimFire"}
COLOR = {"native": "#555555", "pyrorl": "#1f77b4", "haksar": "#2ca02c", "rothermel": "#d62728"}


def growth(fm, seeds, difficulty, steps):
    curves = []
    for s in seeds:
        w = World(rng=np.random.default_rng(s), map_type=MAP_TYPES[s % len(MAP_TYPES)], difficulty=difficulty,
                  fire_model=fm)
        w.reset()
        c = []
        for _ in range(steps):
            w.step()
            c.append((w.burning | w.burnt).sum() * w.cell_m ** 2 / 1e4)          # hectares
        curves.append(c)
    return np.array(curves)


def side_by_side(path, seed, difficulty, steps, maps):
    fig, axes = plt.subplots(len(maps), len(FIRE_MODELS), figsize=(3.1 * len(FIRE_MODELS), 3.2 * len(maps)),
                             squeeze=False)
    for r, mt in enumerate(maps):
        for c, fm in enumerate(FIRE_MODELS):
            w = World(rng=np.random.default_rng(seed + r), map_type=mt, difficulty=difficulty, fire_model=fm)
            w.reset()
            for _ in range(steps):
                w.step()
            ha = (w.burning | w.burnt).sum() * w.cell_m ** 2 / 1e4
            name, spd = w.wind_text()
            draw_world(axes[r, c], w, f"{LABEL[fm]}\n{mt.replace('_', ' ')}: {ha:.1f} ha, wind to {name}")
    handles = [Patch(fc=TERRAIN_COLORS[k], ec="grey", label=TERRAIN_NAMES[k]) for k in range(7)]
    handles += [Patch(fc="#2f2f2f", label="burnt"), Patch(fc="#e63946", label="fire")]
    fig.legend(handles=handles, loc="lower center", ncol=9, frameon=False, fontsize=9)
    fig.suptitle(f"Same map, wind and ignition points; four fire engines ({difficulty}, step {steps})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=(0, 0.04, 1, 0.96))
    plt.savefig(path, dpi=85)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=12, help="missions per engine for growth curves and scores")
    ap.add_argument("--difficulty", default="medium")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--drones", type=int, default=3)
    ap.add_argument("--quick", action="store_true", help="skip the baseline-score table")
    ap.add_argument("--out", default="runs/compare")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    side_by_side(os.path.join(args.out, "side_by_side.png"), 40, args.difficulty, 150,
                 ["rolling_hills", "mountain_valley", "urban_interface"])
    print(f"saved {args.out}/side_by_side.png ({time.time() - t0:.0f}s)", flush=True)

    seeds = range(100, 100 + args.episodes)
    res = {"args": vars(args), "growth_ha": {}, "scores": {}}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    t = np.arange(1, args.steps + 1)
    marks = [50, 100, 200, 300]
    print(f"\nBurnt + burning area (ha), median over {args.episodes} missions, all map types, {args.difficulty}")
    print(f"{'fire model':32s} " + " ".join(f"{'t=' + str(m):>8s}" for m in marks))
    for fm in FIRE_MODELS:
        g = growth(fm, seeds, args.difficulty, args.steps)
        med = np.median(g, 0)
        res["growth_ha"][fm] = {"median": med.round(3).tolist(), "p25": np.percentile(g, 25, 0).round(3).tolist(),
                                "p75": np.percentile(g, 75, 0).round(3).tolist()}
        print(f"{LABEL[fm]:32s} " + " ".join(f"{med[m - 1]:8.1f}" for m in marks if m <= args.steps), flush=True)
        ax.plot(t, med, color=COLOR[fm], lw=2, label=LABEL[fm])
        ax.fill_between(t, np.percentile(g, 25, 0), np.percentile(g, 75, 0), color=COLOR[fm], alpha=0.12)
    ax.set_xlabel("step")
    ax.set_ylabel("area reached by fire (ha)")
    ax.set_title(f"Fire growth per engine ({args.difficulty}, all map types; band = 25-75 %)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "growth.png"), dpi=110)
    plt.close(fig)

    if not args.quick:
        print(f"\nBaseline scores ({args.drones} drones, {args.episodes} missions, all map types, {args.difficulty}); "
              f"higher is better")
        pols = [("random", RandomPolicy()), ("lawnmower", Lawnmower()), ("greedy", Greedy())]
        print(f"{'fire model':32s} " + " ".join(f"{n:>10s}" for n, _ in pols))
        for fm in FIRE_MODELS:
            out = {n: evaluate(p, n_drones=args.drones, seeds=range(1000, 1000 + args.episodes), map_types="all",
                               difficulty=args.difficulty, fire_models=fm) for n, p in pols}
            res["scores"][fm] = out
            print(f"{LABEL[fm]:32s} " + " ".join(f"{out[n]['return']:10.2f}" for n, _ in pols), flush=True)

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(f"\nsaved {args.out}/growth.png and results.json  (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
