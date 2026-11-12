"""
Re-evaluate trained PPO models against the baselines on the SAME missions, with confidence intervals.

For every mission (seed) each policy gets one criticality-weighted staleness value (lower is better).
Because all policies fly the same missions, they are compared mission by mission (paired):
"gain vs X" = mean over missions of (staleness of X - staleness of the policy), with a 95% interval.
A positive gain whose interval does not include 0 means the policy is better than X.
"gap closed" = (random - policy) / (random - greedy): 0% = as good as random, 100% = as good as greedy.

Run all models that share the same settings (map size, difficulty, fire engine) in one call:

    python evaluate_models.py --runs runs/size64 runs/size64_seed1 --size 64 --difficulty easy \
        --fire-model native --episodes 30 --workers 16 --out runs/eval_size64

Macro-action models are detected automatically (action space larger than 7); pass --macro-horizon if it
was not 8. Results: <out>/summary.md (tables for the report) and <out>/evaluation.json (all numbers).
"""
import argparse, io, json, os, time
from multiprocessing import Pool

import numpy as np

from baselines import evaluate, Lawnmower, Greedy, RandomPolicy

# torch >= 2.15 cannot read straight from the zip stream that stable-baselines3 passes it; read it into memory
# first (harmless on older versions).
import torch
_orig_load = torch.load


def _load(f, *a, **k):
    if hasattr(f, "read") and not isinstance(f, io.BytesIO):
        f = io.BytesIO(f.read())
    return _orig_load(f, *a, **k)


torch.load = _load


def load_policy(path, deterministic, horizon):
    from stable_baselines3 import PPO
    import train_ppo
    from macro import MacroPolicyWrapper
    torch.set_num_threads(1)
    model = PPO.load(path, device="cpu", custom_objects=dict(
        policy_kwargs=dict(features_extractor_class=train_ppo.SmallCNN,
                           features_extractor_kwargs=dict(features_dim=128),
                           normalize_images=False)))
    n_actions = int(model.action_space.n)
    if n_actions == 7:
        return train_ppo.PPOPolicyWrapper(model, deterministic=deterministic)
    grid = int(round((n_actions / 3) ** 0.5))
    return MacroPolicyWrapper(model, deterministic=deterministic, grid=grid, horizon=horizon)


def run_task(task):
    name, kind, path, deterministic, split, maps, kw = task
    if kind == "random":
        pol = RandomPolicy()
    elif kind == "lawnmower":
        pol = Lawnmower()
    elif kind == "greedy":
        pol = Greedy()
    else:
        pol = load_policy(path, deterministic, kw.pop("horizon"))
    kw.pop("horizon", None)
    m = evaluate(pol, map_types=maps, **kw)
    return name, split, m


def paired(a, b):
    """Mean and 95% interval of (a - b) over missions."""
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    return float(d.mean()), float(1.96 * d.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run folders that contain model.zip")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=5000)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--difficulty", default="medium")
    ap.add_argument("--fire-model", default="native")
    ap.add_argument("--maps", default="rolling_hills,mountain_valley,dense_forest,farmland")
    ap.add_argument("--test-maps", default="lake_shore,urban_interface")
    ap.add_argument("--drones", type=int, default=1)
    ap.add_argument("--macro-horizon", type=int, default=8)
    ap.add_argument("--workers", type=int, default=max(1, len(os.sched_getaffinity(0)) - 1))
    ap.add_argument("--out", default="runs/eval")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    seeds = list(range(args.seed0, args.seed0 + args.episodes))
    kw = dict(n_drones=args.drones, seeds=seeds, size=args.size, difficulty=args.difficulty,
              fire_models=args.fire_model)
    splits = [("train_maps", args.maps), ("unseen_maps", args.test_maps)]
    policies = [("random", "random", None, True), ("lawnmower", "lawnmower", None, True),
                ("greedy", "greedy", None, True)]
    for r in args.runs:
        tag = os.path.basename(os.path.normpath(r))
        path = os.path.join(r, "model.zip")
        policies += [(f"{tag} (det.)", "ppo", path, True), (f"{tag} (stoch.)", "ppo", path, False)]
    tasks = [(name, kind, path, det, split, maps, dict(kw, horizon=args.macro_horizon))
             for split, maps in splits for name, kind, path, det in policies]
    print(f"{len(tasks)} evaluations x {args.episodes} missions, {args.workers} workers", flush=True)
    t0 = time.time()
    with Pool(args.workers) as pool:
        out = pool.map(run_task, tasks, chunksize=1)
    res = {split: {} for split, _ in splits}
    for name, split, m in out:
        res[split][name] = m
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)

    lines = [f"Criticality-weighted staleness (lower is better), size {args.size}, difficulty {args.difficulty}, "
             f"fire engine {args.fire_model}, {args.drones} drone(s), {args.episodes} missions per cell "
             f"(seeds {seeds[0]}-{seeds[-1]}), mean +- 95% interval. Gains are paired over the same missions "
             f"(positive = better than that baseline).", ""]
    for split, _ in splits:
        r = res[split]
        rnd, lawn, gr = (r[k]["staleness_eps"] for k in ("random", "lawnmower", "greedy"))
        span = np.mean(rnd) - np.mean(gr)
        lines += [f"### {split}", "",
                  "| Policy | Staleness | Gain vs random | Gain vs lawnmower | Gain vs greedy | Gap closed |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for name, m in r.items():
            e = m["staleness_eps"]
            cells = []
            for base in (rnd, lawn, gr):
                g, c = paired(base, e)
                cells.append("-" if e is base else f"{g:+.3f} +- {c:.3f}")
            closed = (np.mean(rnd) - m["staleness"]) / span if span > 0 else float("nan")
            lines.append(f"| {name} | {m['staleness']:.3f} +- {m['staleness_ci95']:.3f} | "
                         + " | ".join(cells) + f" | {100 * closed:.0f}% |")
        lines.append("")
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(args.out, "evaluation.json"), "w") as f:
        json.dump({"args": vars(args), "results": res}, f, indent=2, default=float)
    print(f"\nwritten: {args.out}/summary.md and {args.out}/evaluation.json")


if __name__ == "__main__":
    main()
