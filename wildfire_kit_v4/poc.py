"""
Proof of concept: ONE drone keeps an up-to-date picture of the fire.

Deliberately simple (see EXPERIMENTS / report, "proof of concept"):
  * one drone, fixed medium altitude: perfect sensor that sees a (2R+1) x (2R+1) window (R = 7 cells),
    no missed fire, no false alarms, no smoke, no heat, no battery, no no-fly zones
  * actions: N, S, E, W, hover (2 cells per move)
  * every cell is equally important (houses / forest weighting comes later)
  * the fire can be slowed down: it advances once every `fire_every` drone steps
  * besides the initial fire, new fires can start during the episode (`new_rate` per fire step), so
    there is a reason to keep searching

The drone's picture of the fire is a map B (cell believed to be burning, as last seen). The truth is F.
Quality of the picture = overlap (IoU) of B and F:   acc = |B and F| / |B or F|   (1 if both empty).
acc drops when a fire is missed, when the front advances unseen, or when B still shows fire that went out.

Reward per step (all terms roughly in [0, 1]):
    r = W_ACC * acc
      + W_FIX * sum over corrected cells of m(c) / max(|F|, Z_MIN)
      + W_EXP * (1 - acc) * (fraction of the view that was never seen before)
  corrected cell = a cell whose state in B changes when the drone looks at it (fire found, front advanced,
  fire went out). m(c) = 1 + BETA * min(1, d(c) / D) for a cell that becomes "burning" in B, where d(c) is
  its distance to the nearest cell already believed to be burning (a new, separate focus is worth up to
  1 + BETA times a cell on a known front); m(c) = 1 for other corrections.

Evaluation (metrics(), the same for every policy):
    iou           mean acc over the episode (main metric, higher is better)
    t_first       steps until the first fire is in the drone's map (censored at max_steps)
    foci_found    fraction of fire foci (initial ignitions + new fires far from existing fire) that were seen
    foci_latency  mean steps from the start of a focus until it is seen (unseen foci count as censored)
    hover_frac    fraction of steps the drone did not move
    seen_frac     fraction of the map seen at least once
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.ndimage import distance_transform_edt

from world import World, DIFFICULTY, MAP_TYPES

OBS = 32
N_CH = 8
R = 7                                     # view radius (medium altitude in the full environment)
MOVE = 2
N_, S_, E_, W_, HOVER = range(5)
MOVES = {N_: (0, -1), S_: (0, 1), E_: (1, 0), W_: (-1, 0), HOVER: (0, 0)}
DEFAULT_W = dict(acc=1.0, fix=1.0, exp=0.5, beta=3.0, d=10.0, z_min=10.0)
FOCUS_D = 10                              # cells: a new fire farther than this from existing fire is a new focus
AGE_CAP = 60


def _as_list(x, allowed):
    if isinstance(x, str):
        x = allowed if x == "all" else x.split(",")
    return list(x)


class PoCCore:
    def __init__(self, size=64, max_steps=300, fire_every=4, new_rate=0.01, map_types="all", weights=None,
                 fire_model="native"):
        assert size % OBS == 0
        self.size, self.max_steps, self.fire_every, self.new_rate = size, max_steps, fire_every, new_rate
        self.map_types = _as_list(map_types, MAP_TYPES)
        self.w = {**DEFAULT_W, **(weights or {})}
        self.world = World(size=size, rng=np.random.default_rng(0), fire_model=fire_model)

    # ------------------------------------------------------------------ episode
    def reset(self, seed=None):
        n, w = self.size, self.world
        w.rng = np.random.default_rng(seed)
        w.map_type_req = self.map_types[w.rng.integers(len(self.map_types))]
        w.difficulty = "easy"
        w.cfg = dict(DIFFICULTY["easy"], fires=(1, 1), new=self.new_rate, smoke=0.0, nofly=0, battery=None)
        w.reset()
        self.t = 0
        self.pos = np.array([n // 2, n - 2])                 # x, y: bottom middle
        self.B = np.zeros((n, n), dtype=bool)                # believed burning
        self.burnt_known = np.zeros((n, n), dtype=bool)
        self.seen = np.zeros((n, n), dtype=bool)
        self.last_seen = np.full((n, n), -10 ** 6)
        self.F_ever = w.burning.copy()
        self.foci = [[int(y), int(x), 0, -1] for y, x in np.argwhere(w.burning)]   # y, x, t_start, t_seen
        self.acc_sum, self.hover = 0.0, 0
        self.t_first = None
        self.last_reward = {}
        self._sense()
        return self

    def acc(self):
        F = self.world.burning
        union = (self.B | F).sum()
        return 1.0 if union == 0 else float((self.B & F).sum() / union)

    def _window(self):
        x, y = self.pos
        n = self.size
        return max(0, y - R), min(n, y + R + 1), max(0, x - R), min(n, x + R + 1)

    def _sense(self):
        """The drone looks: returns (corrected-cell mass, fraction of the view never seen before)."""
        w, wt = self.world, self.w
        y0, y1, x0, x1 = self._window()
        F = w.burning[y0:y1, x0:x1]
        Bw = self.B[y0:y1, x0:x1]
        changed = Bw != F
        fix = 0.0
        if changed.any():
            new_fire = changed & F
            fix = float((changed & ~F).sum())                      # fire that went out
            if new_fire.any():
                if self.B.any():
                    d = distance_transform_edt(~self.B)[y0:y1, x0:x1][new_fire]
                else:
                    d = np.full(int(new_fire.sum()), wt["d"])
                fix += float((1.0 + wt["beta"] * np.minimum(1.0, d / wt["d"])).sum())
        new_seen = float((~self.seen[y0:y1, x0:x1]).mean())
        self.B[y0:y1, x0:x1] = F
        self.burnt_known[y0:y1, x0:x1] = w.burnt[y0:y1, x0:x1]
        self.seen[y0:y1, x0:x1] = True
        self.last_seen[y0:y1, x0:x1] = self.t
        if self.t_first is None and self.B.any():
            self.t_first = self.t
        for f in self.foci:                                        # focus point inside the view
            if f[3] < 0 and y0 <= f[0] < y1 and x0 <= f[1] < x1:
                f[3] = self.t
        return fix, new_seen

    def _track_foci(self, before):
        w = self.world
        new = w.burning & ~before
        if not new.any():
            return
        d = distance_transform_edt(~before) if before.any() else np.full(new.shape, np.inf)
        for y, x in np.argwhere(new & (d > FOCUS_D)):
            if all(abs(y - f[0]) > FOCUS_D or abs(x - f[1]) > FOCUS_D for f in self.foci):
                self.foci.append([int(y), int(x), self.t, -1])

    def step(self, action):
        w, wt, n = self.world, self.w, self.size
        dx, dy = MOVES[int(action)]
        old = self.pos.copy()
        self.pos = np.clip(self.pos + MOVE * np.array([dx, dy]), 0, n - 1)
        self.hover += int((self.pos == old).all())
        self.t += 1
        if self.t % self.fire_every == 0:
            before = self.F_ever.copy()
            w.step()
            self.F_ever |= w.burning
            self._track_foci(before)
        fix, new_seen = self._sense()
        acc = self.acc()
        z = max(float(w.burning.sum()), wt["z_min"])
        parts = dict(acc=acc, fix=fix / z, exp=(1.0 - acc) * new_seen)
        r = wt["acc"] * parts["acc"] + wt["fix"] * parts["fix"] + wt["exp"] * parts["exp"]
        self.acc_sum += acc
        self.last_reward = parts
        done = self.t >= self.max_steps
        return r, done, parts

    # ------------------------------------------------------------------ observation
    def _crop(self, a, fill=0.0):
        h = OBS // 2
        x, y = self.pos
        p = np.pad(a, h, constant_values=fill)
        return p[y:y + OBS, x:x + OBS]

    def observe(self):
        n = self.size
        f = n // OBS
        age = np.minimum(1.0, (self.t - self.last_seen) / AGE_CAP).astype(np.float32)
        o = np.empty((N_CH, OBS, OBS), dtype=np.float32)
        o[0] = self._crop(self.B.astype(np.float32))
        o[1] = self._crop(age, fill=1.0)
        o[2] = self._crop(self.burnt_known.astype(np.float32))
        o[3] = self._crop(np.ones((n, n), dtype=np.float32))
        pool = lambda a, how: (a.reshape(OBS, f, OBS, f).max(axis=(1, 3)) if how == "max"
                               else a.reshape(OBS, f, OBS, f).mean(axis=(1, 3)))
        o[4] = pool(self.B.astype(np.float32), "max")
        o[5] = pool(age, "mean")
        o[6] = pool((~self.seen).astype(np.float32), "mean")
        o[7] = 0.0
        o[7, min(OBS - 1, self.pos[1] // f), min(OBS - 1, self.pos[0] // f)] = 1.0
        return o

    def metrics(self):
        T = max(self.t, 1)
        lat = [(f[3] if f[3] >= 0 else self.max_steps) - f[2] for f in self.foci]
        return dict(iou=self.acc_sum / T,
                    t_first=self.t_first if self.t_first is not None else self.max_steps,
                    foci=len(self.foci),
                    foci_found=float(np.mean([f[3] >= 0 for f in self.foci])) if self.foci else 1.0,
                    foci_latency=float(np.mean(lat)) if lat else 0.0,
                    hover_frac=self.hover / T,
                    seen_frac=float(self.seen.mean()))


class PoCEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, size=64, max_steps=300, fire_every=4, new_rate=0.01, map_types="all", weights=None):
        self.core = PoCCore(size=size, max_steps=max_steps, fire_every=fire_every, new_rate=new_rate,
                            map_types=map_types, weights=weights)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(N_CH, OBS, OBS), dtype=np.float32)
        self.action_space = spaces.Discrete(5)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(2 ** 31))
        self.core.reset(seed=seed)
        return self.core.observe(), {}

    def step(self, action):
        r, done, parts = self.core.step(action)
        info = dict(parts)
        if done:
            info["episode_metrics"] = self.core.metrics()
        return self.core.observe(), r, False, done, info


# ---------------------------------------------------------------------- baselines
def step_towards(pos, target):
    dx, dy = target[0] - pos[0], target[1] - pos[1]
    if dx == 0 and dy == 0:
        return HOVER
    if abs(dx) >= abs(dy):
        return E_ if dx > 0 else W_
    return S_ if dy > 0 else N_


class RandomPolicy:
    def reset(self, core):
        self.rng = np.random.default_rng(12345)

    def act(self, core):
        return int(self.rng.integers(5))


class Lawnmower:
    """Sweeps the map in lanes 2R+1 cells apart, forever."""

    def reset(self, core):
        n, lane = core.size, 2 * R + 1
        xs = list(range(R, n, lane))
        self.wps = []
        for i, x in enumerate(xs):
            ys = (n - 1 - R, R) if i % 2 == 0 else (R, n - 1 - R)
            self.wps += [(x, ys[0]), (x, ys[1])]
        self.k = 0

    def act(self, core):
        for _ in range(len(self.wps)):
            tx, ty = self.wps[self.k]
            if abs(core.pos[0] - tx) <= 1 and abs(core.pos[1] - ty) <= 1:
                self.k = (self.k + 1) % len(self.wps)
            else:
                return step_towards(core.pos, (tx, ty))
        return HOVER


class Greedy:
    """Monitors the known fire: goes to the believed-burning cell seen longest ago; searches (lawnmower)
    only while no fire is known. It tends to stay with the fire it found."""

    def reset(self, core):
        self.sweep = Lawnmower()
        self.sweep.reset(core)

    def act(self, core):
        if not core.B.any():
            return self.sweep.act(core)
        ys, xs = np.nonzero(core.B)
        age = core.t - core.last_seen[ys, xs]
        dist = np.abs(xs - core.pos[0]) + np.abs(ys - core.pos[1])
        k = np.lexsort((dist, -age))[0]
        return step_towards(core.pos, (xs[k], ys[k]))


def evaluate(policy, seeds, **core_kw):
    """Runs `policy` (object with reset(core) / act(core)) on each seed; returns per-episode metric lists."""
    core = PoCCore(**core_kw)
    out = {}
    for s in seeds:
        core.reset(seed=s)
        policy.reset(core)
        done = False
        while not done:
            _, done, _ = core.step(policy.act(core))
        for k, v in core.metrics().items():
            out.setdefault(k, []).append(float(v))
    return out
