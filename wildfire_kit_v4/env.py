"""
Drone swarm environment on top of world.World (v3).

New in v3:
  - every episode can use a different map type and difficulty level
  - smoke: thick smoke downwind of intense fire makes drones miss fire (worse higher up)
  - no-fly zones: drones cannot enter them (hard / extreme)
  - battery: drones must return to base to recharge (hard / extreme); an optional
    safety layer flies them home automatically when the battery runs low
  - heat limit depends on difficulty
  - observation: 22 x 32 x 32, drone-centred (local crop + coarse whole-area map)
New in v4:
  - fire_models: which fire-spread engine(s) to use ("native", "pyrorl", "haksar", "rothermel", a
    comma list or "all"); with several, each episode draws one (domain randomisation across simulators)
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from world import World, FOREST, MAP_TYPES, DIFFICULTY
from fire_models import FIRE_MODELS

N, S, E, W, UP, DOWN, HOVER = range(7)
MOVES = {N: (0, -1), S: (0, 1), E: (1, 0), W: (-1, 0)}

# Altitude settings: index 0 = low (10-20 m), 1 = medium (~50 m), 2 = high (~120 m)
VIEW_RADIUS = [3, 7, 15]           # 7x7, 15x15, 31x31 cells  (35 m, 75 m, 155 m at 5 m/cell)
BLOCK = [1, 2, 4]                  # camera detail: 1 cell, 2x2 blocks, 4x4 blocks
MISS_PROB = [0.0, 0.05, 0.15]      # chance a burning cell is missed (clear air)
SMOKE_EFFECT = [0.1, 0.4, 0.7]     # extra miss chance in thick smoke, per altitude
FALSE_ALARM = [0.0, 0.002, 0.004]  # chance an unburnt cell is reported as burning
QUALITY = [1.0, 0.6, 0.3]          # how much a look refreshes the team's information
OBS = 32
N_CHANNELS = 22
BASE_RADIUS = 2                    # cells around the base where drones recharge


def block_view(a, b):
    if b == 1:
        return a
    n = a.shape[0]
    m = n // b
    small = a[:m * b, :m * b].reshape(m, b, m, b).mean(axis=(1, 3))
    full = np.zeros_like(a)
    full[:m * b, :m * b] = np.kron(small, np.ones((b, b), dtype=a.dtype))
    return full


def pool(a, f, how="mean"):
    n = a.shape[0] // f
    r = a[:n * f, :n * f].reshape(n, f, n, f)
    return r.max(axis=(1, 3)) if how == "max" else r.mean(axis=(1, 3))


def _as_list(x, allowed):
    if isinstance(x, str):
        x = allowed if x == "all" else x.split(",")
    return list(x)


class SwarmCore:
    def __init__(self, size=128, n_drones=1, max_steps=300, age_cap=60, seed=None, start_alt=1,
                 move_cells=2, record=False, map_types="random", difficulty="medium", safety_return=True,
                 fire_models="native"):
        assert size % OBS == 0, "size must be a multiple of 32"
        self.size, self.n_drones, self.max_steps = size, n_drones, max_steps
        self.age_cap, self.start_alt, self.move = age_cap, start_alt, move_cells
        self.record, self.safety_return = record, safety_return
        self.map_types = _as_list(map_types, MAP_TYPES)
        self.difficulties = _as_list(difficulty, list(DIFFICULTY))
        self.fire_models = _as_list(fire_models, FIRE_MODELS)
        world_ss, sensor_ss = np.random.SeedSequence(seed).spawn(2)
        self.rng = np.random.default_rng(sensor_ss)
        self.world = World(size=size, rng=np.random.default_rng(world_ss))
        self._yy, self._xx = np.mgrid[0:size, 0:size]

    # ================================================================ setup
    def reset(self, seed=None, map_type=None, difficulty=None, fire_model=None):
        # Separate random streams: the world (map + fire) must evolve identically whatever the
        # drones do, so different strategies are compared on exactly the same fires.
        if seed is not None:
            world_ss, sensor_ss = np.random.SeedSequence(seed).spawn(2)
            self.world.rng = np.random.default_rng(world_ss)
            self.rng = np.random.default_rng(sensor_ss)
        w, n = self.world, self.size
        w.map_type_req = map_type or self.map_types[w.rng.integers(len(self.map_types))]
        w.difficulty = difficulty or self.difficulties[w.rng.integers(len(self.difficulties))]
        w.cfg = DIFFICULTY[w.difficulty]
        if fire_model is not None:
            w.fire_model = fire_model
        elif len(self.fire_models) == 1:                    # (no extra random draw: v3 seeds unchanged)
            w.fire_model = self.fire_models[0]
        else:
            w.fire_model = self.fire_models[w.rng.integers(len(self.fire_models))]
        w.reset()
        self.t = 0
        self.belief = np.zeros((n, n), dtype=np.float32)
        self.fresh = np.zeros((n, n), dtype=np.float32)
        self.last_seen = np.full((n, n), -10 ** 6)
        # Base: a road cell near the bottom edge (or bottom middle if there is no road)
        rc = np.argwhere(w.road)
        by, bx = rc[np.argmax(rc[:, 0])] if len(rc) else (n - 2, n // 2)
        self.base = (int(bx), int(min(by, n - 2)))
        bxx, byy = self.base
        w.nofly[max(0, byy - 8):byy + 9, max(0, bxx - 8):bxx + 9] = False     # keep the base reachable
        self.pos = np.clip(np.array([[bxx + 2 * i - (self.n_drones - 1), byy] for i in range(self.n_drones)]), 0, n - 1)
        self.alt = np.full(self.n_drones, self.start_alt)
        self.alive = np.ones(self.n_drones, dtype=bool)
        self.lost_reason = [""] * self.n_drones
        self.battery_max = w.cfg["battery"]
        self.battery = np.full(self.n_drones, float(self.battery_max or 1))
        self.returning = np.zeros(self.n_drones, dtype=bool)
        self.alt_counts = np.zeros(3)
        self.trails = [[tuple(p)] for p in self.pos]
        self.history = []
        self._views, self._smoke = {}, None
        for i in range(self.n_drones):
            self._sense(i)
        self._record()

    # ================================================================ helpers
    def criticality(self):
        w = self.world
        return (w.fire_map() * (1 + 2 * w.home_near + (w.terrain == FOREST))).astype(np.float32)

    def staleness(self):
        return 1.0 - self.fresh

    def smoke(self):
        if self._smoke is None:
            self._smoke = self.world.smoke()
        return self._smoke

    def _view(self, b):
        if b not in self._views:
            self._views[b] = block_view(self.world.fire_map(), b)
        return self._views[b]

    def dist_to_base(self, i):
        return abs(self.pos[i, 0] - self.base[0]) + abs(self.pos[i, 1] - self.base[1])

    def at_base(self, i):
        return self.dist_to_base(i) <= BASE_RADIUS

    def step_towards(self, i, target):
        """Move towards target, going around no-fly zones when the direct step is blocked."""
        x, y = self.pos[i]
        dx, dy = target[0] - x, target[1] - y
        if abs(dx) <= 1 and abs(dy) <= 1:
            return HOVER
        first = (E if dx > 0 else W) if abs(dx) >= abs(dy) else (S if dy > 0 else N)
        second = (S if dy > 0 else N) if abs(dx) >= abs(dy) else (E if dx > 0 else W)
        order = [first]
        if (second in (N, S) and dy != 0) or (second in (E, W) and dx != 0):
            order.append(second)
        order += [m for m in (N, S, E, W) if m not in order]          # detours around no-fly zones
        for a in order:
            if self._can_move(i, a):
                return a
        return HOVER

    def _can_move(self, i, a):
        mx, my = MOVES[a]
        nx = int(np.clip(self.pos[i, 0] + self.move * mx, 0, self.size - 1))
        ny = int(np.clip(self.pos[i, 1] + self.move * my, 0, self.size - 1))
        return not self.world.nofly[ny, nx]

    # ================================================================ dynamics
    def _sense(self, i):
        x, y = self.pos[i]
        a = self.alt[i]
        r = VIEW_RADIUS[a]
        y0, y1 = max(0, y - r), min(self.size, y + r + 1)
        x0, x1 = max(0, x - r), min(self.size, x + r + 1)
        v = self._view(BLOCK[a])[y0:y1, x0:x1]
        if a == 0:
            obs = v.copy()
        elif a == 1:
            obs = np.round(v * 4) / 4                       # coarse intensity levels
        else:
            obs = np.where(v > 0.25, 0.6, 0.0)               # burning-or-not; small fires vanish
        miss = MISS_PROB[a] + SMOKE_EFFECT[a] * self.smoke()[y0:y1, x0:x1]
        u = self.rng.random(obs.shape)
        obs = np.where((obs > 0) & (u < miss), 0.0, obs)
        obs = np.where((obs == 0) & (u < FALSE_ALARM[a]), 0.5, obs).astype(np.float32)
        q = QUALITY[a]
        reg = self.fresh[y0:y1, x0:x1]
        better = q >= reg
        self.belief[y0:y1, x0:x1] = np.where(better, obs, self.belief[y0:y1, x0:x1])
        self.fresh[y0:y1, x0:x1] = np.maximum(reg, q)
        self.last_seen[y0:y1, x0:x1] = self.t

    def _safety(self, i, act):
        """Battery safety layer: fly home when needed, stay until recharged."""
        if not self.battery_max or not self.safety_return:
            return act
        need = self.dist_to_base(i) / self.move + 12
        if self.returning[i] and self.at_base(i):
            if self.battery[i] >= 0.95 * self.battery_max:
                self.returning[i] = False
                return act
            return HOVER
        if self.battery[i] <= need:
            self.returning[i] = True
        if self.returning[i]:
            if self.alt[i] < 2:
                return UP                                      # climb (safe) while heading home
            return self.step_towards(i, self.base)
        return act

    def step(self, actions):
        lost_now = 0
        w = self.world
        for i, act in enumerate(actions):
            if not self.alive[i]:
                continue
            act = self._safety(i, int(act))
            if act in MOVES and self._can_move(i, act):        # no-fly zones block the move
                dx, dy = MOVES[act]
                self.pos[i] = np.clip(self.pos[i] + self.move * np.array([dx, dy]), 0, self.size - 1)
            elif act == UP:
                self.alt[i] = min(2, self.alt[i] + 1)
            elif act == DOWN:
                self.alt[i] = max(0, self.alt[i] - 1)
            self.trails[i].append(tuple(self.pos[i]))
        w.step()
        self.t += 1
        self._views, self._smoke = {}, None
        self.fresh = np.maximum(0.0, self.fresh - 1.0 / self.age_cap)
        for i in range(self.n_drones):
            if not self.alive[i]:
                continue
            x, y = self.pos[i]
            if self.battery_max:
                if self.at_base(i):
                    self.battery[i] = min(self.battery_max, self.battery[i] + self.battery_max / 8)
                else:
                    self.battery[i] -= 1
                if self.battery[i] <= 0:
                    self.alive[i], self.lost_reason[i] = False, "battery"
                    lost_now += 1
                    continue
            if self.alt[i] == 0 and w.burning[y, x] and w.intensity[y, x] > w.cfg["heat"]:
                self.alive[i], self.lost_reason[i] = False, "heat"
                lost_now += 1
                continue
            self.alt_counts[self.alt[i]] += 1
            self._sense(i)
        crit = self.criticality()
        penalty = float((crit * self.staleness()).sum() / (crit.sum() + 1.0))
        coverage = float((crit * self.fresh).sum() / (crit.sum() + 1.0))
        reward = coverage - 5.0 * lost_now
        self._record()
        done = self.t >= self.max_steps or not self.alive.any()
        return reward, done, {"penalty": penalty, "coverage": coverage, "lost": lost_now}

    def _record(self):
        if not self.record:
            return
        w = self.world
        crit = self.criticality()
        near = w.burning.copy()
        for _ in range(4):
            near = near | np.roll(near, 1, 0) | np.roll(near, -1, 0) | np.roll(near, 1, 1) | np.roll(near, -1, 1)
        self.history.append(dict(t=self.t, known_critical=float((crit * self.fresh).sum() / max(crit.sum(), 1e-6)),
                                 burning_ha=w.burning.sum() * w.cell_m ** 2 / 1e4,
                                 burnt_ha=(w.burnt | w.burning).sum() * w.cell_m ** 2 / 1e4,
                                 homes_threatened=int((w.homes & near).sum() // 4)))

    # ================================================================ observation (22 x 32 x 32)
    def _cache(self):
        w, f = self.world, self.size // OBS
        if getattr(self, "_static_ep", None) is not w.terrain:
            self._static_ep = w.terrain
            self._crit_w = ((1 + 2 * w.home_near + (w.terrain == FOREST)) / 4.0).astype(np.float32)
            self._flam = (w._spread_mult / 2.0).astype(np.float32)          # how easily each cell burns
            self._homes = w.homes.astype(np.float32)
            self._nofly = w.nofly.astype(np.float32)
            self._ones = np.ones((self.size, self.size), dtype=np.float32)
            base = np.zeros((OBS, OBS), dtype=np.float32)
            base[min(OBS - 1, self.base[1] // f), min(OBS - 1, self.base[0] // f)] = 1.0
            self._static_pools = dict(flam=pool(self._flam, f), homes=pool(self._homes, f, "max"),
                                      nofly=pool(self._nofly, f, "max"), base=base)
        if getattr(self, "_cache_t", None) != self.t:
            self._cache_t = self.t
            stale = self.staleness()
            est = self.belief * self._crit_w
            self._dyn = dict(stale=stale, est=est,
                             belief_pool=pool(self.belief, f, "max"), stale_pool=pool(stale, f),
                             est_pool=pool(est, f, "max"), wx_pool=pool(w.wx, f), wy_pool=pool(w.wy, f))
        return self._dyn

    @staticmethod
    def _crop(L, x, y, fill=0.0):
        n, h = L.shape[0], OBS // 2
        out = np.full((OBS, OBS), fill, dtype=np.float32)
        y0, y1, x0, x1 = y - h, y + h, x - h, x + h
        sy0, sx0, sy1, sx1 = max(0, y0), max(0, x0), min(n, y1), min(n, x1)
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = L[sy0:sy1, sx0:sx1]
        return out

    def observe(self, i):
        n, f, w = self.size, self.size // OBS, self.world
        d = self._cache()
        sp = self._static_pools
        x, y = int(self.pos[i, 0]), int(self.pos[i, 1])
        others = np.zeros((n, n), dtype=np.float32)
        for j in range(self.n_drones):
            if j != i and self.alive[j]:
                others[self.pos[j, 1], self.pos[j, 0]] = 1.0
        obs = np.empty((N_CHANNELS, OBS, OBS), dtype=np.float32)
        # detailed local crop around the drone
        for c, L in enumerate([self.belief, d["stale"], self._flam, self._homes, others, d["est"]]):
            obs[c] = self._crop(L, x, y)
        obs[6] = self._crop(self._ones, x, y)                                  # inside the map
        obs[7] = self._crop(self._nofly, x, y)
        obs[8] = np.clip(self._crop(w.elev - w.elev[y, x], x, y) / 50.0, -1, 1)  # terrain height around me
        # coarse map of the whole area
        obs[9], obs[10], obs[11] = d["belief_pool"], d["stale_pool"], sp["flam"]
        obs[12], obs[13] = sp["homes"], d["est_pool"]
        obs[14] = 0.0
        obs[14, min(OBS - 1, y // f), min(OBS - 1, x // f)] = 1.0              # where I am
        obs[15] = pool(others, f, "max")
        obs[16], obs[17] = d["wx_pool"], d["wy_pool"]                          # known wind field
        obs[18], obs[19] = sp["nofly"], sp["base"]
        obs[20] = self.alt[i] / 2.0
        obs[21] = self.battery[i] / self.battery_max if self.battery_max else 1.0
        return obs


class SingleDroneEnv(gym.Env):
    """Gymnasium wrapper for one drone. Observation: 22 x 32 x 32."""

    metadata = {"render_modes": []}

    def __init__(self, size=128, max_steps=300, seed=None, start_alt=1, map_types="random",
                 difficulty="medium", safety_return=True, fire_models="native"):
        self.core = SwarmCore(size=size, n_drones=1, max_steps=max_steps, seed=seed, start_alt=start_alt,
                              map_types=map_types, difficulty=difficulty, safety_return=safety_return,
                              fire_models=fire_models)
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(N_CHANNELS, OBS, OBS), dtype=np.float32)
        self.action_space = spaces.Discrete(7)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.core.reset(seed=seed)
        return self.core.observe(0), {}

    def step(self, action):
        r, done, info = self.core.step([int(action)])
        truncated = self.core.t >= self.core.max_steps
        return self.core.observe(0), r, done and not truncated, truncated, info
