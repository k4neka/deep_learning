"""
Proof of concept training: one drone keeps an up-to-date picture of the fire (see poc.py).

Curriculum (each stage starts from the previous model with --init-from):
    stage 1: python train_poc.py --fire-every 4 --steps 2000000 --out runs/poc_fe4
    stage 2: python train_poc.py --fire-every 2 --steps 2000000 --init-from runs/poc_fe4/model.zip --out runs/poc_fe2
    stage 3: python train_poc.py --fire-every 1 --steps 2000000 --init-from runs/poc_fe2/model.zip --out runs/poc_fe1
Evaluate a saved model only:
    python train_poc.py --steps 0 --init-from runs/poc_fe4/model.zip --fire-every 4 --out runs/poc_fe4_eval

Outputs in --out: model.zip, results.json, summary.md (tables for the report: mean +- 95% interval over the
same missions for every policy, and the paired difference of IoU against each baseline).
"""
import argparse, io, json, os, time
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from poc import PoCEnv, RandomPolicy, Lawnmower, Greedy, evaluate, MAP_TYPES
from train_ppo import SmallCNN

# torch >= 2.15 cannot read straight from the zip stream that SB3 passes; read it into memory first (harmless)
_orig_load = torch.load
torch.load = lambda f, *a, **k: _orig_load(io.BytesIO(f.read()) if hasattr(f, "read") and not isinstance(f, io.BytesIO) else f, *a, **k)

METRICS = ["iou", "foci_found", "foci_latency", "t_first", "hover_frac", "seen_frac"]


class PPOPolicy:
    def __init__(self, model, deterministic=True):
        self.model, self.deterministic = model, deterministic

    def reset(self, core):
        pass

    def act(self, core):
        a, _ = self.model.predict(core.observe(), deterministic=self.deterministic)
        return int(a)


def ci(v):
    v = np.asarray(v, dtype=float)
    return float(1.96 * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--gamma", type=float, default=0.98)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--fire-every", type=int, default=4, help="the fire advances once every N drone steps")
    ap.add_argument("--new-rate", type=float, default=0.01, help="chance of a new fire per fire step")
    ap.add_argument("--maps", default="rolling_hills,mountain_valley,dense_forest,farmland")
    ap.add_argument("--test-maps", default="lake_shore,urban_interface")
    ap.add_argument("--w-acc", type=float, default=1.0)
    ap.add_argument("--w-fix", type=float, default=1.0)
    ap.add_argument("--w-exp", type=float, default=0.5)
    ap.add_argument("--beta", type=float, default=3.0, help="extra weight of a new, separate focus")
    ap.add_argument("--eval-episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-from", default=None, help="model.zip to start from (curriculum / evaluation only)")
    ap.add_argument("--vecnorm", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--subproc", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="runs/poc")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cores = len(os.sched_getaffinity(0))
    torch.set_num_threads(max(1, min(4, cores)))
    weights = dict(acc=args.w_acc, fix=args.w_fix, exp=args.w_exp, beta=args.beta)
    env_kw = dict(size=args.size, max_steps=args.max_steps, fire_every=args.fire_every, new_rate=args.new_rate,
                  weights=weights)
    print(f"PoC | size {args.size} | fire every {args.fire_every} steps | new fire rate {args.new_rate} | "
          f"weights {weights} | envs {args.n_envs} x {args.n_steps} | cores {cores}", flush=True)

    vec_cls = SubprocVecEnv if args.subproc else DummyVecEnv
    env = make_vec_env(PoCEnv, n_envs=args.n_envs, seed=args.seed, vec_env_cls=vec_cls,
                       env_kwargs=dict(env_kw, map_types=args.maps))
    if args.vecnorm:
        env = VecNormalize(env, norm_obs=False, norm_reward=True, gamma=args.gamma)
    policy_kwargs = dict(features_extractor_class=SmallCNN, features_extractor_kwargs=dict(features_dim=128),
                         normalize_images=False)
    if args.init_from:
        model = PPO.load(args.init_from, env=env, device=args.device,
                         custom_objects=dict(policy_kwargs=policy_kwargs, learning_rate=args.lr,
                                             ent_coef=args.ent_coef, gamma=args.gamma, n_steps=args.n_steps,
                                             batch_size=args.batch_size))
        print(f"starting from {args.init_from}", flush=True)
    else:
        model = PPO("CnnPolicy", env, seed=args.seed, verbose=0, device=args.device, n_steps=args.n_steps,
                    batch_size=args.batch_size, n_epochs=4, gamma=args.gamma, learning_rate=args.lr,
                    ent_coef=args.ent_coef, policy_kwargs=policy_kwargs)

    eval_kw = dict(env_kw, map_types=args.maps)
    t0, log, done = time.time(), [], 0
    chunk = max(args.steps // 8, args.n_envs * args.n_steps)
    while done < args.steps:
        model.learn(chunk, reset_num_timesteps=False)
        done += chunk
        model.save(os.path.join(args.out, "model"))
        m = evaluate(PPOPolicy(model, True), range(2000, 2010), **eval_kw)
        ms = evaluate(PPOPolicy(model, False), range(2000, 2010), **eval_kw)
        row = dict(steps=done, minutes=round((time.time() - t0) / 60, 1),
                   **{k: float(np.mean(v)) for k, v in m.items()},
                   iou_stoch=float(np.mean(ms["iou"])), foci_found_stoch=float(np.mean(ms["foci_found"])))
        log.append(row)
        print(f"[{row['minutes']:5.1f} min] {done:8d} steps  IoU={row['iou']:.3f} (stoch {row['iou_stoch']:.3f})  "
              f"foci found={row['foci_found']:.2f} (stoch {row['foci_found_stoch']:.2f})  "
              f"hover={row['hover_frac']:.2f}  seen={row['seen_frac']:.2f}", flush=True)
    if args.steps > 0:
        model.save(os.path.join(args.out, "model"))
    train_min = round((time.time() - t0) / 60, 1)

    # final comparison on the same fresh missions for every policy
    seeds = range(5000, 5000 + args.eval_episodes)
    final = {}
    for split, maps in [("train_maps", args.maps), ("unseen_maps", args.test_maps)]:
        kw = dict(env_kw, map_types=maps)
        final[split] = {name: evaluate(pol, seeds, **kw) for name, pol in
                        [("random", RandomPolicy()), ("lawnmower", Lawnmower()), ("greedy", Greedy()),
                         ("ppo", PPOPolicy(model, True)), ("ppo_stoch", PPOPolicy(model, False))]}

    lines = [f"Proof of concept, one drone, size {args.size}, fire advances every {args.fire_every} steps, "
             f"new fire rate {args.new_rate}, {args.eval_episodes} missions per cell (seeds {seeds[0]}-{seeds[-1]}), "
             f"mean +- 95% interval. IoU = overlap of the drone's fire map with the real fire (higher is better).", ""]
    for split, res in final.items():
        lines += [f"### {split}", "",
                  "| Policy | IoU | Foci found | Focus latency (steps) | First fire (steps) | Hover | Map seen |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for name, r in res.items():
            lines.append(f"| {name} | " + " | ".join(f"{np.mean(r[k]):.3f} +- {ci(r[k]):.3f}" if k in
                         ("iou", "foci_found", "hover_frac", "seen_frac") else f"{np.mean(r[k]):.0f} +- {ci(r[k]):.0f}"
                         for k in METRICS) + " |")
        lines += ["", "Paired IoU difference (PPO - baseline, same missions; positive = PPO better):", "",
                  "| Policy | vs random | vs lawnmower | vs greedy |", "| --- | --- | --- | --- |"]
        for name in ("ppo", "ppo_stoch"):
            cells = []
            for base in ("random", "lawnmower", "greedy"):
                d = np.asarray(res[name]["iou"]) - np.asarray(res[base]["iou"])
                cells.append(f"{d.mean():+.3f} +- {ci(d):.3f}")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    text = "\n".join(lines)
    print("\n" + text, flush=True)
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"args": vars(args), "train_minutes": train_min, "curve": log, "final": final}, f, indent=1)


if __name__ == "__main__":
    main()
