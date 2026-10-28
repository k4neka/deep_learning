"""Baseline strategies and a shared evaluation routine (v4: --fire-model picks the fire-spread engine)."""
import numpy as np
from env import SwarmCore, UP, DOWN, VIEW_RADIUS
from world import MAP_TYPES, DIFFICULTY


class RandomPolicy:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self, core):
        pass

    def act(self, core, i):
        return int(self.rng.integers(7))


class Lawnmower:
    """Fixed sweep at high altitude; the drones split the sweep between them."""

    def reset(self, core):
        n, r = core.size, VIEW_RADIUS[2]
        rows = list(range(r, n, 2 * r + 1))
        path = []
        for k, y in enumerate(rows):
            xs = (r, n - 1 - r) if k % 2 == 0 else (n - 1 - r, r)
            path += [(xs[0], y), (xs[1], y)]
        self.paths, self.idx = [], []
        for i in range(core.n_drones):
            shift = (i * len(path)) // core.n_drones
            self.paths.append(path[shift:] + path[:shift])
            self.idx.append(0)

    def act(self, core, i):
        if core.alt[i] < 2:
            return UP
        p = self.paths[i]
        tgt = p[self.idx[i]]
        x, y = core.pos[i]
        # next waypoint when reached (or when it lies inside a no-fly zone and we are close)
        if (abs(x - tgt[0]) <= 2 and abs(y - tgt[1]) <= 2) or \
           (core.world.nofly[tgt[1], tgt[0]] and abs(x - tgt[0]) + abs(y - tgt[1]) < 12):
            self.idx[i] = (self.idx[i] + 1) % len(p)
            tgt = p[self.idx[i]]
        return core.step_towards(i, tgt)


class Greedy:
    """Search at high altitude until fire is known, then inspect the most critical,
    most outdated fire at medium altitude."""

    def reset(self, core):
        self.targets = {}

    def act(self, core, i):
        d = core._cache()
        stale, est = d["stale"], d["est"]
        # Real fire comes in clusters: ignore isolated detections (sensor false alarms)
        hot = (est > 0).astype(np.int8)
        nb = sum(np.roll(np.roll(hot, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
        est = est * (nb >= 3)
        want_alt = 1 if (est > 0).sum() >= 6 else 2
        if core.alt[i] > want_alt:
            return DOWN
        if core.alt[i] < want_alt:
            return UP
        never = core.last_seen < -1000
        # known fire first; otherwise go where information is missing or outdated
        score = est * stale * 4 + 0.15 * stale * never + 0.05 * stale
        x, y = core.pos[i]
        score = score - 0.002 * (np.abs(core._xx - x) + np.abs(core._yy - y))
        score[core.world.nofly] = -1e9
        for j, (tx, ty) in self.targets.items():          # spread drones over different targets
            if j != i:
                score[max(0, ty - 12):ty + 12, max(0, tx - 12):tx + 12] -= 10
        ty, tx = np.unravel_index(np.argmax(score), score.shape)
        self.targets[i] = (tx, ty)
        return core.step_towards(i, (tx, ty))


def evaluate(policy, n_drones=1, seeds=range(1000, 1030), size=128, max_steps=300, start_alt=1,
             map_types="random", difficulty="medium", safety_return=True, fire_models="native"):
    """Each seed gets a map type, difficulty and fire engine drawn from the given lists. Returns a metric dict."""
    core = SwarmCore(size=size, n_drones=n_drones, max_steps=max_steps, start_alt=start_alt,
                     map_types=map_types, difficulty=difficulty, safety_return=safety_return,
                     fire_models=fire_models)
    rets, pens, lost, alt = [], [], {"heat": 0, "battery": 0}, np.zeros(3)
    for s in seeds:
        core.reset(seed=s)
        policy.reset(core)
        ret, p_sum, done = 0.0, 0.0, False
        while not done:
            r, done, info = core.step([policy.act(core, i) for i in range(n_drones)])
            ret += r
            p_sum += info["penalty"]
        rets.append(ret)
        pens.append(p_sum / core.t)
        for why in core.lost_reason:
            if why:
                lost[why] += 1
        alt += core.alt_counts
    return {
        "return": float(np.mean(rets)),
        "return_std": float(np.std(rets)),
        "staleness": float(np.mean(pens)),   # criticality-weighted staleness, lower is better
        "lost_heat_per_ep": lost["heat"] / len(seeds),
        "lost_battery_per_ep": lost["battery"] / len(seeds),
        "alt_share": (alt / max(alt.sum(), 1)).round(2).tolist(),
    }


if __name__ == "__main__":
    import argparse, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--drones", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=6)
    ap.add_argument("--fire-model", default="native", help="native, pyrorl, haksar, rothermel (or a comma list)")
    args = ap.parse_args()
    print(f"{args.drones} drone(s), {args.episodes} missions per cell, fire model: {args.fire_model}; "
          f"score = higher is better\n")
    pols = [("random", RandomPolicy()), ("lawnmower", Lawnmower()), ("greedy", Greedy())]
    print(f"{'difficulty':10s} " + " ".join(f"{n:>10s}" for n, _ in pols) + "   (all map types mixed)")
    for diff in DIFFICULTY:
        t0 = time.time()
        res = [evaluate(p, n_drones=args.drones, seeds=range(1000, 1000 + args.episodes),
                        map_types="all", difficulty=diff, fire_models=args.fire_model) for _, p in pols]
        print(f"{diff:10s} " + " ".join(f"{m['return']:10.2f}" for m in res) +
              f"   lost(heat/battery) greedy {res[2]['lost_heat_per_ep']:.2f}/{res[2]['lost_battery_per_ep']:.2f}"
              f"  ({time.time() - t0:.0f}s)")
