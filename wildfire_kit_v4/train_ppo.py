"""
Train one drone with PPO and a small CNN on the big map, then compare against the baselines.
The observation is drone-centred (local 32x32 crop + coarse 32x32 map), so the
network size does not depend on the map size.

v4: --fire-models chooses the fire engine(s) used in TRAINING (one, or a comma list = a different
engine per episode); after training, every policy is also tested on each engine in --test-fire-models
("cross_simulator" in results.json), i.e. on fire dynamics from published simulators it never saw.

Usage:
    python train_ppo.py --steps 50000 --n-envs 8 --subproc --out runs/smoke      # quick test
    python train_ppo.py --steps 3000000 --n-envs 32 --subproc --out runs/ppo     # real run (VSC)
    python train_ppo.py ... --fire-models native,pyrorl,haksar,rothermel         # train on all engines
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from env import SingleDroneEnv
from baselines import evaluate, Lawnmower, Greedy, RandomPolicy


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
    """Lets a trained model control any number of drones (shared weights)."""

    def __init__(self, model):
        self.model = model

    def reset(self, core):
        pass

    def act(self, core, i):
        a, _ = self.model.predict(core.observe(i), deterministic=True)
        return int(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/ppo")
    ap.add_argument("--subproc", action="store_true")
    ap.add_argument("--start-alt", type=int, default=1, help="0 low, 1 medium, 2 high")
    ap.add_argument("--size", type=int, default=128, help="map size in cells (multiple of 32)")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--maps", default="rolling_hills,mountain_valley,dense_forest,farmland",
                    help="map types used for TRAINING (comma list or 'all')")
    ap.add_argument("--test-maps", default="lake_shore,urban_interface",
                    help="map types NEVER seen in training, used to test generalisation")
    ap.add_argument("--difficulty", default="easy,medium", help="training difficulty levels (comma list or 'all')")
    ap.add_argument("--test-difficulty", default="medium")
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--fire-models", default="native",
                    help="fire engine(s) for TRAINING: native, pyrorl, haksar, rothermel (comma list or 'all')")
    ap.add_argument("--test-fire-models", default="native,pyrorl,haksar,rothermel",
                    help="fire engines for the final cross-simulator test")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    # Use only the cores Slurm gave this job (os.cpu_count() would report the whole node)
    cores = len(os.sched_getaffinity(0))
    torch.set_num_threads(max(1, min(4, cores)))
    print(f"cores available: {cores}, device: {args.device}", flush=True)
    print(f"training maps: {args.maps} | held-out maps: {args.test_maps} | difficulty: {args.difficulty} | "
          f"fire models: {args.fire_models}", flush=True)

    # --subproc runs each environment copy in its own process (use it when you have several CPU cores)
    vec_cls = SubprocVecEnv if args.subproc else DummyVecEnv
    env = make_vec_env(SingleDroneEnv, n_envs=args.n_envs, seed=args.seed, vec_env_cls=vec_cls,
                       env_kwargs=dict(start_alt=args.start_alt, size=args.size,
                                       map_types=args.maps, difficulty=args.difficulty,
                                       fire_models=args.fire_models))
    model = PPO(
        "CnnPolicy", env, seed=args.seed, verbose=0, device=args.device,
        n_steps=512, batch_size=1024, n_epochs=4, gamma=0.98, learning_rate=3e-4, ent_coef=0.02,
        policy_kwargs=dict(features_extractor_class=SmallCNN,
                           features_extractor_kwargs=dict(features_dim=128),
                           normalize_images=False),
    )
    common = dict(start_alt=args.start_alt, size=args.size)
    t0 = time.time()
    log = []
    chunk = max(args.steps // 6, 2048)
    done_steps = 0
    while done_steps < args.steps:
        model.learn(chunk, reset_num_timesteps=False)
        done_steps = model.num_timesteps
        m = evaluate(PPOPolicyWrapper(model), n_drones=1, seeds=range(2000, 2005),
                     map_types=args.maps, difficulty=args.test_difficulty, fire_models=args.fire_models, **common)
        m.update(steps=done_steps, minutes=round((time.time() - t0) / 60, 1))
        log.append(m)
        print(f"[{m['minutes']:5.1f} min] {done_steps:8d} steps  score={m['return']:6.2f}  "
              f"staleness={m['staleness']:.3f}  lost heat/battery={m['lost_heat_per_ep']:.2f}/"
              f"{m['lost_battery_per_ep']:.2f}  alt={m['alt_share']}", flush=True)
    model.save(os.path.join(args.out, "model"))
    fps = model.num_timesteps / (time.time() - t0)

    # Final comparison on fresh seeds: training map types AND map types never seen in training
    seeds = range(5000, 5000 + args.eval_episodes)
    results = {}
    for split, maps in [("train_maps", args.maps), ("unseen_maps", args.test_maps)]:
        results[split] = {}
        for name, pol in [("random", RandomPolicy()), ("lawnmower", Lawnmower()),
                          ("greedy", Greedy()), ("ppo", PPOPolicyWrapper(model))]:
            results[split][name] = evaluate(pol, n_drones=1, seeds=seeds, map_types=maps,
                                            difficulty=args.test_difficulty, fire_models=args.fire_models,
                                            **common)
        # the same single-drone policy flying 3 drones (not trained for teams; pipeline check)
        results[split]["ppo_3drones"] = evaluate(PPOPolicyWrapper(model), n_drones=3, seeds=seeds,
                                                 map_types=maps, difficulty=args.test_difficulty,
                                                 fire_models=args.fire_models, **common)
        results[split]["greedy_3drones"] = evaluate(Greedy(), n_drones=3, seeds=seeds, map_types=maps,
                                                    difficulty=args.test_difficulty,
                                                    fire_models=args.fire_models, **common)
    # Cross-simulator test: same seeds, maps and wind, fire spread by each published engine
    cross = {}
    for fm in args.test_fire_models.split(","):
        cross[fm] = {}
        for name, pol in [("random", RandomPolicy()), ("lawnmower", Lawnmower()),
                          ("greedy", Greedy()), ("ppo", PPOPolicyWrapper(model))]:
            cross[fm][name] = evaluate(pol, n_drones=1, seeds=seeds, map_types="all",
                                       difficulty=args.test_difficulty, fire_models=fm, **common)
    out = {"train_fps": round(fps), "train_minutes": round((time.time() - t0) / 60, 1),
           "args": vars(args), "curve": log, "final": results, "cross_simulator": cross}
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    for split, res in results.items():
        print(f"\n{split} ({args.test_difficulty}):")
        for name, m in res.items():
            print(f"  {name:15s} score {m['return']:7.2f}  staleness {m['staleness']:.3f}  "
                  f"lost heat/battery {m['lost_heat_per_ep']:.2f}/{m['lost_battery_per_ep']:.2f}")
    print(f"\ncross-simulator test (all map types, {args.test_difficulty}; trained on fire: {args.fire_models}):")
    print(f"  {'fire model':12s} " + " ".join(f"{n:>10s}" for n in ["random", "lawnmower", "greedy", "ppo"]))
    for fm, res in cross.items():
        print(f"  {fm:12s} " + " ".join(f"{res[n]['return']:10.2f}" for n in ["random", "lawnmower", "greedy", "ppo"]))
    print("train fps:", round(fps))


if __name__ == "__main__":
    main()
