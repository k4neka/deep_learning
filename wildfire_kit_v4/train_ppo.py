"""
Train one drone with PPO and a small CNN on the big map, then compare against the baselines.
The observation is drone-centred (local 32x32 crop + coarse 32x32 map), so the
network size does not depend on the map size.

v4: --fire-models chooses the fire engine(s) used in TRAINING (one, or a comma list = a different
engine per episode); after training, every policy is also tested on each engine in --test-fire-models
("cross_simulator" in results.json), i.e. on fire dynamics from published simulators it never saw.

v5 (changes vs v4):
  * VecNormalize on the REWARD only (observations are left untouched). On by default,
    switch off with --no-vecnorm. Statistics are saved to <out>/vecnormalize.pkl.
  * Evaluation is done twice: deterministic (argmax action, "ppo") and stochastic (action sampled
    from the policy, "ppo_stoch"), both in the periodic evaluation and in the final comparison.
  * --n-steps (default 512): rollout batch = n_envs * n_steps (aim for about 4096-8192).
  * Warning if --n-envs is larger than the number of CPU cores available to the job.
  * The model is saved after every evaluation chunk, so a job that hits its time limit
    still leaves the latest model behind.

v6 (changes vs v5):
  * --ent-coef (default 0.02 = the value used in B1 and B2): PPO entropy bonus weight.
  * PPO verbose=1: SB3 prints its training table (train/entropy_loss, explained_variance, ...) in the log.
    This only changes the printout, not the training.

v7 (changes vs v6, B4 behaviour unchanged when the new flags are not given):
  * --action-mode macro: the policy picks a sector of a --macro-grid x --macro-grid grid and an
    altitude; a controller flies there for at most --macro-horizon low-level steps (see macro.py).
  * --steps always counts LOW-LEVEL simulator steps, in both modes, so runs are comparable.
    The curve in results.json logs "low_steps".
  * --reward-preset coverage: reward = coverage of the team map (the original v4 reward without the
    loss penalty). Default "v6" = B2-B4 reward (with --re-weight for R_E).
  * --loss-weight (default 0): penalty per lost drone. It is now OFF by default; all runs before
    13/11/2026 used 5. Pass --loss-weight 5 to reproduce them.
  * --gamma and --batch-size flags (defaults: primitive 0.98 / 1024, macro 0.95 / 512);
    in macro mode --n-steps defaults to 128 (more PPO updates for the same budget).
  * Final evaluation stores one staleness value per mission ("staleness_eps"), for paired comparisons
    with evaluate_models.py. results.json also stores the git commit.

Usage:
    python train_ppo.py --steps 50000 --n-envs 8 --subproc --out runs/smoke      # quick test
    python train_ppo.py --steps 3000000 --n-envs 32 --subproc --out runs/ppo     # real run (VSC)
    python train_ppo.py ... --fire-models native,pyrorl,haksar,rothermel         # train on all engines
    python train_ppo.py ... --ent-coef 0.005                                     # lower entropy bonus
    python train_ppo.py ... --action-mode macro --reward-preset coverage         # macro-actions
"""
import argparse, json, os, subprocess, time
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from env import SingleDroneEnv
from macro import MacroDroneEnv, MacroPolicyWrapper
from baselines import evaluate, Lawnmower, Greedy, RandomPolicy

REWARD_PRESETS = {
    "v6": {},                                                     # REWARD_W in env.py (B2-B4)
    "coverage": dict(E=0.0, D=0.0, C=1.0, FP=0.0, FN=0.0),        # coverage only (loss: see --loss-weight)
}


class SmallCNN(BaseFeaturesExtractor):
    """CNN for small grid maps (the SB3 default CNN is built for 84x84 Atari frames)."""

    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        c = observation_space.shape[0]
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),   # 32 -> 16
            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(),   # 16 -> 8
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.net(torch.zeros(1, *observation_space.shape)).shape[1]
        self.head = nn.Sequential(nn.Linear(n, features_dim), nn.ReLU())

    def forward(self, x):
        return self.head(self.net(x))


class PPOPolicyWrapper:
    """Lets a trained model control any number of drones (shared weights).

    deterministic=True  -> always the most likely action (argmax)
    deterministic=False -> action sampled from the policy distribution (as during training)
    """

    def __init__(self, model, deterministic=True):
        self.model = model
        self.deterministic = deterministic

    def reset(self, core):
        pass

    def act(self, core, i):
        a, _ = self.model.predict(core.observe(i), deterministic=self.deterministic)
        return int(a)


class LowStepCounter(BaseCallback):
    """Counts low-level simulator steps (macro-actions report info["low_steps"]; primitive = 1)."""

    def __init__(self):
        super().__init__()
        self.low_steps = 0

    def _on_step(self):
        for info in self.locals["infos"]:
            self.low_steps += int(info.get("low_steps", 1))
        return True


def git_info():
    here = os.path.dirname(os.path.abspath(__file__))
    out = {}
    for key, cmd in [("commit", ["git", "rev-parse", "HEAD"]), ("branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"])]:
        try:
            out[key] = subprocess.check_output(cmd, cwd=here, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            out[key] = "unknown"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300_000, help="training budget in LOW-LEVEL simulator steps")
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=None,
                    help="PPO rollout length per env (default 512 primitive, 128 macro); "
                         "rollout batch = n_envs * n_steps")
    ap.add_argument("--batch-size", type=int, default=None, help="PPO minibatch (default 1024 primitive, 512 macro)")
    ap.add_argument("--gamma", type=float, default=None, help="discount (default 0.98 primitive, 0.95 macro)")
    ap.add_argument("--ent-coef", type=float, default=0.02,
                    help="PPO entropy bonus weight (0.02 = value used in B1 and B2)")
    ap.add_argument("--reward-preset", choices=list(REWARD_PRESETS), default="v6",
                    help="v6 = B2-B4 reward; coverage = coverage of the team map only")
    ap.add_argument("--re-weight", type=float, default=2.0,
                    help="reward weight of R_E (exploration) in TRAINING only; 2.0 = v6 default (B1-B3). "
                         "Only used with --reward-preset v6")
    ap.add_argument("--loss-weight", type=float, default=0.0,
                    help="penalty per lost drone (heat / battery). 0 = removed (default since v7); "
                         "runs before 13/11 used 5. With one drone a loss still ends the episode.")
    ap.add_argument("--action-mode", choices=["primitive", "macro"], default="primitive")
    ap.add_argument("--macro-grid", type=int, default=8, help="macro mode: G x G sectors")
    ap.add_argument("--macro-horizon", type=int, default=8, help="macro mode: max low-level steps per macro-action")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/ppo")
    ap.add_argument("--subproc", action="store_true")
    ap.add_argument("--vecnorm", action=argparse.BooleanOptionalAction, default=True,
                    help="normalise the reward (NOT the observations) with VecNormalize; --no-vecnorm to disable")
    ap.add_argument("--start-alt", type=int, default=1, help="0 low, 1 medium, 2 high")
    ap.add_argument("--size", type=int, default=128, help="map size in cells (multiple of 32)")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--maps", default="rolling_hills,mountain_valley,dense_forest,farmland",
                    help="map types used for TRAINING (comma list or 'all')")
    ap.add_argument("--test-maps", default="lake_shore,urban_interface",
                    help="map types NEVER seen in training, used to test generalisation")
    ap.add_argument("--difficulty", default="easy,medium", help="training difficulty levels (comma list or 'all')")
    ap.add_argument("--test-difficulty", default="medium")
    ap.add_argument("--eval-episodes", type=int, default=30)
    ap.add_argument("--fire-models", default="native",
                    help="fire engine(s) for TRAINING: native, pyrorl, haksar, rothermel (comma list or 'all')")
    ap.add_argument("--test-fire-models", default="native,pyrorl,haksar,rothermel",
                    help="fire engines for the final cross-simulator test ('' = skip it)")
    args = ap.parse_args()
    macro = args.action_mode == "macro"
    if args.n_steps is None:
        args.n_steps = 128 if macro else 512
    if args.batch_size is None:
        args.batch_size = 512 if macro else 1024
    if args.gamma is None:
        args.gamma = 0.95 if macro else 0.98
    reward_w = dict(REWARD_PRESETS[args.reward_preset])
    if args.reward_preset == "v6":
        reward_w["E"] = args.re_weight
    reward_w["loss"] = args.loss_weight
    os.makedirs(args.out, exist_ok=True)
    # Use only the cores Slurm gave this job (os.cpu_count() would report the whole node)
    cores = len(os.sched_getaffinity(0))
    torch.set_num_threads(max(1, min(4, cores)))
    print(f"cores available: {cores}, device: {args.device}", flush=True)
    print(f"training maps: {args.maps} | held-out maps: {args.test_maps} | difficulty: {args.difficulty} | "
          f"fire models: {args.fire_models} | size: {args.size}", flush=True)
    print(f"action mode: {args.action_mode}" + (f" (grid {args.macro_grid}, horizon {args.macro_horizon})" if macro else "") +
          f" | reward: {args.reward_preset} {reward_w} | gamma: {args.gamma} | ent_coef: {args.ent_coef}", flush=True)
    print(f"envs: {args.n_envs} | rollout batch: {args.n_envs} x {args.n_steps} = {args.n_envs * args.n_steps} | "
          f"minibatch: {args.batch_size} | reward normalisation (VecNormalize): {args.vecnorm}", flush=True)
    if args.n_envs > cores:
        print(f"WARNING: --n-envs {args.n_envs} is larger than the {cores} cores available; "
              f"the simulators will compete for CPU time", flush=True)

    # --subproc runs each environment copy in its own process (use it when you have several CPU cores)
    # make_vec_env gives every environment its own seed (seed + index)
    vec_cls = SubprocVecEnv if args.subproc else DummyVecEnv
    env_kwargs = dict(start_alt=args.start_alt, size=args.size, map_types=args.maps, difficulty=args.difficulty,
                      fire_models=args.fire_models, reward_w=reward_w)
    if macro:
        env_cls = MacroDroneEnv
        env_kwargs.update(grid=args.macro_grid, horizon=args.macro_horizon)
    else:
        env_cls = SingleDroneEnv
    env = make_vec_env(env_cls, n_envs=args.n_envs, seed=args.seed, vec_env_cls=vec_cls, env_kwargs=env_kwargs)
    if args.vecnorm:
        # reward only: observations are images for the CNN and must stay as they are,
        # so evaluate() (which has its own, un-normalised environments) stays comparable.
        env = VecNormalize(env, norm_obs=False, norm_reward=True, gamma=args.gamma)
    model = PPO(
        "CnnPolicy", env, seed=args.seed, verbose=1, device=args.device,
        n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=4, gamma=args.gamma, learning_rate=3e-4,
        ent_coef=args.ent_coef,
        policy_kwargs=dict(features_extractor_class=SmallCNN,
                           features_extractor_kwargs=dict(features_dim=128),
                           normalize_images=False),
    )

    def make_policy(deterministic=True):
        if macro:
            return MacroPolicyWrapper(model, deterministic=deterministic,
                                      grid=args.macro_grid, horizon=args.macro_horizon)
        return PPOPolicyWrapper(model, deterministic=deterministic)

    common = dict(start_alt=args.start_alt, size=args.size)
    model_path = os.path.join(args.out, "model")
    counter = LowStepCounter()
    rollout = args.n_envs * args.n_steps
    t0 = time.time()
    log = []
    chunk = max(args.steps // 6, 2048)          # low-level steps between evaluations
    next_eval = chunk
    while counter.low_steps < args.steps:
        if macro:
            model.learn(rollout, reset_num_timesteps=False, callback=counter)   # one rollout + update
        else:
            model.learn(chunk, reset_num_timesteps=False, callback=counter)
        if counter.low_steps < next_eval and counter.low_steps < args.steps:
            continue
        next_eval = counter.low_steps + chunk
        model.save(model_path)          # checkpoint: survives a job that runs out of time
        eval_kw = dict(n_drones=1, seeds=range(2000, 2005), map_types=args.maps,
                       difficulty=args.test_difficulty, fire_models=args.fire_models, **common)
        m = evaluate(make_policy(True), **eval_kw)
        ms = evaluate(make_policy(False), **eval_kw)
        m.update(steps=model.num_timesteps, low_steps=counter.low_steps, minutes=round((time.time() - t0) / 60, 1),
                 return_stoch=ms["return"], staleness_stoch=ms["staleness"])
        log.append(m)
        print(f"[{m['minutes']:5.1f} min] {counter.low_steps:8d} low-level steps  score={m['return']:6.2f}  "
              f"score_stoch={ms['return']:6.2f}  staleness={m['staleness']:.3f}  staleness_stoch={ms['staleness']:.3f}  "
              f"lost heat/battery={m['lost_heat_per_ep']:.2f}/{m['lost_battery_per_ep']:.2f}  alt={m['alt_share']}",
              flush=True)
    model.save(model_path)
    if args.vecnorm:
        env.save(os.path.join(args.out, "vecnormalize.pkl"))
    fps = counter.low_steps / (time.time() - t0)

    # PPO policies evaluated both ways: deterministic (argmax) and stochastic (sampled)
    def ppo_policies():
        return [("ppo", make_policy(True)), ("ppo_stoch", make_policy(False))]

    # Final comparison on fresh seeds: training map types AND map types never seen in training
    seeds = range(5000, 5000 + args.eval_episodes)
    results = {}
    for split, maps in [("train_maps", args.maps), ("unseen_maps", args.test_maps)]:
        results[split] = {}
        for name, pol in [("random", RandomPolicy()), ("lawnmower", Lawnmower()),
                          ("greedy", Greedy())] + ppo_policies():
            results[split][name] = evaluate(pol, n_drones=1, seeds=seeds, map_types=maps,
                                            difficulty=args.test_difficulty, fire_models=args.fire_models,
                                            **common)
        # the same single-drone policy flying 3 drones (not trained for teams; pipeline check)
        results[split]["ppo_3drones"] = evaluate(make_policy(True), n_drones=3, seeds=seeds,
                                                 map_types=maps, difficulty=args.test_difficulty,
                                                 fire_models=args.fire_models, **common)
        results[split]["greedy_3drones"] = evaluate(Greedy(), n_drones=3, seeds=seeds, map_types=maps,
                                                    difficulty=args.test_difficulty,
                                                    fire_models=args.fire_models, **common)
    # Cross-simulator test: same seeds, maps and wind, fire spread by each published engine
    cross = {}
    for fm in [f for f in args.test_fire_models.split(",") if f]:
        cross[fm] = {}
        for name, pol in [("random", RandomPolicy()), ("lawnmower", Lawnmower()),
                          ("greedy", Greedy())] + ppo_policies():
            cross[fm][name] = evaluate(pol, n_drones=1, seeds=seeds, map_types="all",
                                       difficulty=args.test_difficulty, fire_models=fm, **common)
    out = {"train_fps": round(fps), "train_minutes": round((time.time() - t0) / 60, 1),
           "low_steps": counter.low_steps, "git": git_info(),
           "args": vars(args), "curve": log, "final": results, "cross_simulator": cross}
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    for split, res in results.items():
        print(f"\n{split} ({args.test_difficulty}):")
        for name, m in res.items():
            print(f"  {name:15s} score {m['return']:7.2f}  staleness {m['staleness']:.3f} +- {m['staleness_ci95']:.3f}  "
                  f"lost heat/battery {m['lost_heat_per_ep']:.2f}/{m['lost_battery_per_ep']:.2f}")
    if cross:
        cols = ["random", "lawnmower", "greedy", "ppo", "ppo_stoch"]
        print(f"\ncross-simulator test, staleness (all map types, {args.test_difficulty}; trained on fire: {args.fire_models}):")
        print(f"  {'fire model':12s} " + " ".join(f"{n:>10s}" for n in cols))
        for fm, res in cross.items():
            print(f"  {fm:12s} " + " ".join(f"{res[n]['staleness']:10.3f}" for n in cols))
    print("train low-level steps per second:", round(fps))


if __name__ == "__main__":
    main()
