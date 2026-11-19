"""
Proof of concept training: one drone keeps an up-to-date picture of the fire (see poc.py).

Curriculum (each stage starts from the previous model with --init-from):
    stage 1: python train_poc.py --fire-every 4 --steps 2000000 --out runs/poc_fe4
    stage 2: python train_poc.py --fire-every 2 --steps 2000000 --init-from runs/poc_fe4/model.zip --out runs/poc_fe2
    stage 3: python train_poc.py --fire-every 1 --steps 2000000 --init-from runs/poc_fe2/model.zip --out runs/poc_fe1
Version 3 (reward for the age of the information, see poc.py), one curriculum per variant:
    python train_poc.py --preset v3 --fire-every 4 --out runs/v3_age_fe4                     # age reward
    python train_poc.py --preset v3 --w-rev 0.1 --fire-every 4 --out runs/v3_rev_fe4         # + no back-and-forth
    python train_poc.py --preset v3 --action-mode goto --steps 300000 --fire-every 4 --out runs/v3_goto_fe4
                                                                                             # + 'fly to a sector'
    (with goto, --steps counts decisions; one decision is up to --horizon drone steps)
Version 4 (v3 + reversal penalty + a per-step cost for known fire that is being forgotten, see poc.py):
    python train_poc.py --preset v4 --fire-every 4 --out runs/v4_mean_fe4                    # mean forgotten fire
    python train_poc.py --preset v4 --forget-mode max --fire-every 4 --out runs/v4_max_fe4   # most forgotten fire
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

from poc import PoCEnv, RandomPolicy, Lawnmower, Greedy, Goto, evaluate, MAP_TYPES, DEFAULT_W, V3_W, V4_W
from train_ppo import SmallCNN

# torch >= 2.15 cannot read straight from the zip stream that SB3 passes; read it into memory first (harmless)
_orig_load = torch.load
torch.load = lambda f, *a, **k: _orig_load(io.BytesIO(f.read()) if hasattr(f, "read") and not isinstance(f, io.BytesIO) else f, *a, **k)

METRICS = ["iou", "foci_found", "foci_latency", "t_first", "fire_age", "oldest_fire", "forgotten", "revisit_gap",
           "dwell_frac", "rev_frac", "hover_frac", "seen_frac"]
FRACTION = ("iou", "foci_found", "forgotten", "dwell_frac", "rev_frac", "hover_frac", "seen_frac")
HEADERS = dict(iou="IoU", foci_found="Foci found", foci_latency="Focus latency", t_first="First fire",
               fire_age="Fire age", oldest_fire="Oldest fire", forgotten="Forgotten", revisit_gap="Revisit gap", dwell_frac="Dwell", rev_frac="Reversals",
               hover_frac="Hover", seen_frac="Map seen")


class PPOPolicy:
    def __init__(self, model, deterministic=True):
        self.model, self.deterministic = model, deterministic

    def reset(self, core):
        pass

    def act(self, core):
        a, _ = self.model.predict(core.observe(), deterministic=self.deterministic)
        return int(a)


class GotoPPOPolicy:
    """Runs a 'goto' model step by step exactly as PoCEnv does: a new sector is chosen when the trip is over."""

    def __init__(self, model, size, grid, horizon, deterministic=True):
        self.model, self.deterministic = model, deterministic
        self.goto = Goto(size, grid, horizon)

    def reset(self, core):
        self.goto.left = 0

    def act(self, core):
        a = self.goto.low_action(core.pos)
        while a is None:
            k, _ = self.model.predict(core.observe(), deterministic=self.deterministic)
            self.goto.start(int(k))
            a = self.goto.low_action(core.pos)
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
    ap.add_argument("--fire-every", type=float, default=4, help="the fire advances once every N drone steps (may be fractional)")
    ap.add_argument("--new-rate", type=float, default=0.01, help="chance of a new fire per fire step")
    ap.add_argument("--maps", default="rolling_hills,mountain_valley,dense_forest,farmland")
    ap.add_argument("--test-maps", default="lake_shore,urban_interface")
    ap.add_argument("--preset", choices=["v1", "v3", "v4"], default="v1",
                    help="v1: IoU + corrections + exploration (v2 = v1 with --stale --exp-floor 0.3); "
                         "v3: IoU (0.3) + corrections + age of the refreshed information; "
                         "v4: v3 + reversal penalty 0.1 + cost of forgotten known fire, age reward up to 60 steps")
    ap.add_argument("--w-acc", type=float, default=None)
    ap.add_argument("--w-fix", type=float, default=None)
    ap.add_argument("--w-exp", type=float, default=None)
    ap.add_argument("--w-age", type=float, default=None, help="v3: weight of the age reward")
    ap.add_argument("--fire-mult", type=float, default=None, help="v3: weight of keeping the whole fire fresh (paid per fire cell, divided by the fire size)")
    ap.add_argument("--t-fire", type=float, default=None, help="v3: steps after which a fire cell is fully worth checking again")
    ap.add_argument("--t-empty", type=float, default=None, help="v3: same for a cell without fire")
    ap.add_argument("--w-rev", type=float, default=None, help="penalty for reversing the previous move")
    ap.add_argument("--w-forget", type=float, default=None, help="v4: per-step cost of forgotten known fire")
    ap.add_argument("--forget-mode", choices=["mean", "max"], default=None,
                    help="v4: cost of the whole fire map (mean) or of the most forgotten separate fire (max)")
    ap.add_argument("--t-forget", type=float, default=None, help="v4: steps after which known fire counts as forgotten")
    ap.add_argument("--action-mode", choices=["primitive", "goto"], default="primitive")
    ap.add_argument("--grid", type=int, default=8, help="goto: the map is split into grid x grid sectors")
    ap.add_argument("--horizon", type=int, default=8, help="goto: maximum drone steps per decision")
    ap.add_argument("--beta", type=float, default=None, help="extra weight of a new, separate focus")
    ap.add_argument("--exp-floor", type=float, default=0.0,
                    help="v2: part of the exploration reward kept even when the fire picture is good (v1: 0)")
    ap.add_argument("--stale", action="store_true",
                    help="v2: exploration pays for refreshing areas not seen for a long time, not only never-seen ones")
    ap.add_argument("--eval-new-rate", type=float, default=None,
                    help="new fire rate used in the FINAL evaluation (default: same as --new-rate); "
                         "0.01 makes results comparable with the v1 runs")
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
    weights = dict(DEFAULT_W, **{"v1": {}, "v3": V3_W, "v4": V4_W}[args.preset])
    weights.update(exp_floor=args.exp_floor)
    if args.stale:
        weights["stale"] = 1.0
    for key, val in [("acc", args.w_acc), ("fix", args.w_fix), ("exp", args.w_exp), ("age", args.w_age),
                     ("fire_mult", args.fire_mult), ("t_fire", args.t_fire), ("t_empty", args.t_empty),
                     ("rev", args.w_rev), ("beta", args.beta), ("forget", args.w_forget),
                     ("forget_mode", args.forget_mode), ("t_forget", args.t_forget)]:
        if val is not None:
            weights[key] = val
    env_kw = dict(size=args.size, max_steps=args.max_steps, fire_every=args.fire_every, new_rate=args.new_rate,
                  weights=weights)
    act_kw = dict(action_mode=args.action_mode, grid=args.grid, horizon=args.horizon)
    print(f"PoC | size {args.size} | fire every {args.fire_every} steps | new fire rate {args.new_rate} | "
          f"actions {args.action_mode} | weights {weights} | envs {args.n_envs} x {args.n_steps} | cores {cores}",
          flush=True)

    vec_cls = SubprocVecEnv if args.subproc else DummyVecEnv
    env = make_vec_env(PoCEnv, n_envs=args.n_envs, seed=args.seed, vec_env_cls=vec_cls,
                       env_kwargs=dict(env_kw, map_types=args.maps, **act_kw))
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

    def ppo_policy(det):
        if args.action_mode == "goto":
            return GotoPPOPolicy(model, args.size, args.grid, args.horizon, det)
        return PPOPolicy(model, det)
    t0, log, done = time.time(), [], 0
    chunk = max(args.steps // 8, args.n_envs * args.n_steps)
    while done < args.steps:
        model.learn(chunk, reset_num_timesteps=False)
        done += chunk
        model.save(os.path.join(args.out, "model"))
        m = evaluate(ppo_policy(True), range(2000, 2010), **eval_kw)
        ms = evaluate(ppo_policy(False), range(2000, 2010), **eval_kw)
        row = dict(steps=done, minutes=round((time.time() - t0) / 60, 1),
                   **{k: float(np.mean(v)) for k, v in m.items()},
                   iou_stoch=float(np.mean(ms["iou"])), foci_found_stoch=float(np.mean(ms["foci_found"])))
        log.append(row)
        print(f"[{row['minutes']:5.1f} min] {done:8d} steps  IoU={row['iou']:.3f} (stoch {row['iou_stoch']:.3f})  "
              f"foci found={row['foci_found']:.2f} (stoch {row['foci_found_stoch']:.2f})  "
              f"fire age={row['fire_age']:.0f}  oldest fire={row['oldest_fire']:.0f}  "
              f"forgotten={row['forgotten']:.2f}  dwell={row['dwell_frac']:.2f}  reversals={row['rev_frac']:.2f}  "
              f"hover={row['hover_frac']:.2f}  seen={row['seen_frac']:.2f}", flush=True)
    if args.steps > 0:
        model.save(os.path.join(args.out, "model"))
    train_min = round((time.time() - t0) / 60, 1)

    # final comparison on the same fresh missions for every policy
    seeds = range(5000, 5000 + args.eval_episodes)
    final_kw = dict(env_kw, new_rate=args.eval_new_rate if args.eval_new_rate is not None else args.new_rate)
    final = {}
    for split, maps in [("train_maps", args.maps), ("unseen_maps", args.test_maps)]:
        kw = dict(final_kw, map_types=maps)
        final[split] = {name: evaluate(pol, seeds, **kw) for name, pol in
                        [("random", RandomPolicy()), ("lawnmower", Lawnmower()), ("greedy", Greedy()),
                         ("ppo", ppo_policy(True)), ("ppo_stoch", ppo_policy(False))]}

    lines = [f"Proof of concept, one drone, size {args.size}, fire advances every {args.fire_every} steps, "
             f"new fire rate {final_kw['new_rate']}, {args.eval_episodes} missions per cell (seeds {seeds[0]}-{seeds[-1]}), "
             f"mean +- 95% interval. IoU = overlap of the drone's fire map with the real fire (higher is better). "
             f"Fire age = mean steps since the burning cells were last seen (lower is better). Revisit gap = steps "
             f"between two looks at the same burning cell. Oldest fire = age of the most forgotten separate fire on "
             f"the drone's map. Forgotten = share of the drone's fire map not seen for more than 30 steps. Dwell = share of steps re-watching fire seen <= 5 steps "
             f"ago. Reversals = share of moves that undo the previous move.", ""]
    for split, res in final.items():
        lines += [f"### {split}", "",
                  "| Policy | " + " | ".join(HEADERS[k] for k in METRICS) + " |",
                  "| --- " * (len(METRICS) + 1) + "|"]
        for name, r in res.items():
            lines.append(f"| {name} | " + " | ".join(f"{np.mean(r[k]):.3f} +- {ci(r[k]):.3f}" if k in FRACTION
                         else f"{np.mean(r[k]):.0f} +- {ci(r[k]):.0f}" for k in METRICS) + " |")
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
