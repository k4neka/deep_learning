"""
World generation and fire spread for the wildfire swarm project (v3).

  * 6 map types (plus "random"), each generated fresh from a seed:
        rolling_hills, mountain_valley, lake_shore, dense_forest, urban_interface, farmland
  * Elevation (metres) with two fire effects:
        - slope: fire spreads faster uphill and slower downhill (rule of thumb: the spread rate
          roughly doubles per 10 degrees of upslope)
        - high ground: cells in the top 30% of elevation burn faster by `high_ground_boost`
          (simplified: drier, windier ridges; 20% at "medium")
  * Difficulty levels: easy, medium, hard, extreme (see DIFFICULTY)
  * Wind is part of the environment (known to the drones): prevailing wind with local
    variation, stronger on high ground, gusts and weather shifts depending on difficulty.
  * v4: the fire-spread rule is pluggable (`fire_model=`): "native" (this file, unchanged from v3)
    or an engine from a published open-source wildfire simulator, see fire_models/__init__.py:
    "pyrorl" (Julian & Kochenderfer 2019 / PyroRL), "haksar" (Haksar & Schwager 2018),
    "rothermel" (Rothermel 1972 via MITRE SimFire). Maps, wind, spot fires, smoke and the drones
    are shared by all engines; the same seed gives the same map, wind and ignition points.

NumPy only (the external engines also need torch for "pyrorl").
"""
import numpy as np
from fire_models import make_fire_model

# ------------------------------------------------------------------ terrain types
WATER, ROAD, ROCK, GRASS, FOREST, HOUSE, FIELD = range(7)
TERRAIN_NAMES = ["water", "road", "rock", "grassland", "forest", "houses", "farmland"]
#                       water road rock grass forest house field
SPREAD = np.array([0.0, 0.0, 0.0, 0.8, 1.3, 0.6, 1.6], dtype=np.float32)   # how easily a cell catches fire
INTENSITY = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 0.8, 0.45], dtype=np.float32)
BURN_TIME = np.array([0, 0, 0, 14, 30, 40, 8], dtype=np.int16)              # steps a cell keeps burning

MAP_TYPES = ["rolling_hills", "mountain_valley", "lake_shore", "dense_forest", "urban_interface", "farmland"]

# ------------------------------------------------------------------ difficulty levels
DIFFICULTY = {
    #             fires  wind speed   shifts gusts relief  high-ground spread  spot   new-fire smoke  no-fly battery heat
    #                                               x       boost      x       rate   rate              zones  (steps) limit
    "easy":    dict(fires=(1, 2), wind=(0.2, 0.4), shifts=0, gusts=0.00, relief=0.5, high_boost=0.10, spread=0.85,
                    spot=0.03, new=0.000, smoke=0.0, nofly=0, battery=None, heat=0.85),
    "medium":  dict(fires=(2, 3), wind=(0.35, 0.7), shifts=1, gusts=0.05, relief=1.0, high_boost=0.20, spread=1.00,
                    spot=0.08, new=0.003, smoke=0.5, nofly=0, battery=None, heat=0.70),
    "hard":    dict(fires=(3, 4), wind=(0.5, 0.9), shifts=1, gusts=0.10, relief=1.3, high_boost=0.20, spread=1.10,
                    spot=0.12, new=0.006, smoke=0.8, nofly=1, battery=250, heat=0.70),
    "extreme": dict(fires=(4, 6), wind=(0.7, 1.0), shifts=2, gusts=0.15, relief=1.6, high_boost=0.30, spread=1.20,
                    spot=0.18, new=0.010, smoke=1.0, nofly=2, battery=180, heat=0.60),
}

NEIGH = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]


# ------------------------------------------------------------------ helpers
def box_blur(a, r=1, iters=1):
    for _ in range(iters):
        p = np.pad(a, r, mode="edge")
        out = np.zeros(a.shape, dtype=np.float64)
        k = 2 * r + 1
        for dy in range(k):
            for dx in range(k):
                out += p[dy:dy + a.shape[0], dx:dx + a.shape[1]]
        a = out / (k * k)
    return a


def smooth_noise(n, scale, rng):
    """Smooth random field in [0, 1] with blobs about `scale` cells wide."""
    scale = max(2, int(scale))
    m = n // scale + 2
    up = np.kron(rng.random((m, m)), np.ones((scale, scale)))[:n, :n]
    f = box_blur(up, r=max(1, scale // 3), iters=3)
    f -= f.min()
    return f / (f.max() + 1e-9)


def dilate(mask, r):
    out = mask.copy()
    for _ in range(r):
        out = out | np.roll(out, 1, 0) | np.roll(out, -1, 0) | np.roll(out, 1, 1) | np.roll(out, -1, 1)
    return out


def draw_line(mask, pts, width):
    n = mask.shape[0]
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        for t in np.linspace(0, 1, steps * 2):
            xi, yi = int(round(x0 + t * (x1 - x0))), int(round(y0 + t * (y1 - y0)))
            mask[max(0, yi - width // 2):min(n, yi + (width + 1) // 2),
                 max(0, xi - width // 2):min(n, xi + (width + 1) // 2)] = True


def meander(n, rng, horizontal, base, amp, k=14):
    xs = np.linspace(-2, n + 1, k)
    w = np.cumsum(rng.normal(0, amp, size=k))
    w -= w.mean()
    return [(x, base + d) if horizontal else (base + d, x) for x, d in zip(xs, w)]


def place_houses(t, homes, rng, cy, cx, count, spread):
    n = t.shape[0]
    for _ in range(count):
        hy = int(np.clip(cy + rng.normal(0, spread), 1, n - 3))
        hx = int(np.clip(cx + rng.normal(0, spread), 1, n - 3))
        block = t[hy:hy + 2, hx:hx + 2]
        if np.all((block == GRASS) | (block == FOREST) | (block == FIELD)):
            t[hy:hy + 2, hx:hx + 2] = HOUSE
            homes[hy:hy + 2, hx:hx + 2] = True


# ------------------------------------------------------------------ world
class World:
    """Terrain, elevation, wind and fire for one episode."""

    def __init__(self, size=128, cell_m=5.0, rng=None, map_type="random", difficulty="medium",
                 base_p=0.055, wind_k=1.8, slope_k=3.9, p_die=0.015, fire_model="native"):
        self.size, self.cell_m = size, cell_m
        self.rng = rng or np.random.default_rng()
        self.map_type_req, self.difficulty = map_type, difficulty
        self.fire_model, self.engine = fire_model, None
        self.cfg = DIFFICULTY[difficulty]
        self.base_p, self.wind_k, self.slope_k, self.p_die = base_p, wind_k, slope_k, p_die

    # ============================================================ map generators
    def _gen_rolling_hills(self):
        n, rng = self.size, self.rng
        elev = 40 * smooth_noise(n, n / 4, rng)
        t = np.full((n, n), GRASS, dtype=np.int8)
        fo = smooth_noise(n, n / 8, rng) + 0.3 * (elev / 40)
        t[fo > np.quantile(fo, 0.6)] = FOREST
        flat_low = (elev < np.quantile(elev, 0.35)) & (t == GRASS)
        t[flat_low & (smooth_noise(n, n / 10, rng) > 0.5)] = FIELD
        river = np.zeros((n, n), bool)
        draw_line(river, meander(n, rng, rng.random() < 0.5, rng.uniform(0.3, 0.7) * n, n * 0.04), max(2, n // 40))
        roads = self._roads(2)
        homes = np.zeros((n, n), bool)
        t[river] = WATER
        t[roads & ~river] = ROAD
        self._villages(t, homes, roads & ~river, rng.integers(3, 5), (6, 11))
        elev = np.where(river, elev - 3, elev)
        return t, elev, homes, roads & ~river

    def _gen_mountain_valley(self):
        n, rng = self.size, self.rng
        horizontal = rng.random() < 0.5
        c = rng.uniform(0.4, 0.6) * n
        yy, xx = np.mgrid[0:n, 0:n]
        d = np.abs((yy if horizontal else xx) - c - 6 * np.sin((xx if horizontal else yy) / n * 2 * np.pi))
        elev = 80 * (d / (n / 2)) ** 1.3 + 12 * smooth_noise(n, n / 6, rng)
        t = np.full((n, n), FOREST, dtype=np.int8)
        t[smooth_noise(n, n / 10, rng) > 0.7] = GRASS
        t[elev > np.quantile(elev, 0.88)] = ROCK
        valley = d < n * 0.08
        t[valley] = GRASS
        t[valley & (smooth_noise(n, n / 12, rng) > 0.55)] = FIELD
        river = np.zeros((n, n), bool)
        pts = [(x, c + 6 * np.sin(x / n * 2 * np.pi)) if horizontal else (c + 6 * np.sin(x / n * 2 * np.pi), x)
               for x in np.linspace(-2, n + 1, 30)]
        draw_line(river, pts, max(2, n // 45))
        road = np.zeros((n, n), bool)
        off = n * 0.04
        pts_r = [(x, y + off) if horizontal else (x + off, y) for x, y in pts]
        draw_line(road, pts_r, max(1, n // 64))
        homes = np.zeros((n, n), bool)
        t[river] = WATER
        t[road & ~river] = ROAD
        self._villages(t, homes, road & ~river, rng.integers(2, 4), (8, 14))
        return t, elev, homes, road & ~river

    def _gen_lake_shore(self):
        n, rng = self.size, self.rng
        elev = 30 * smooth_noise(n, n / 4, rng)
        t = np.full((n, n), GRASS, dtype=np.int8)
        fo = smooth_noise(n, n / 7, rng)
        t[fo > np.quantile(fo, 0.5)] = FOREST
        yy, xx = np.mgrid[0:n, 0:n]
        cy, cx = rng.uniform(0.3, 0.7, size=2) * n
        rx, ry = rng.uniform(0.18, 0.28, size=2) * n
        lake = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 + 0.6 * (smooth_noise(n, n / 6, rng) - 0.5) < 1
        elev = np.where(lake, 0, elev + 5)
        shore = dilate(lake, 4) & ~lake
        t[shore & (t == FOREST)] = GRASS
        roads = self._roads(1)
        ring = dilate(lake, 7) & ~dilate(lake, 6)
        roads |= ring
        homes = np.zeros((n, n), bool)
        t[lake] = WATER
        t[roads & ~lake] = ROAD
        ring_cells = ring & ~lake
        self._villages(t, homes, ring_cells, rng.integers(3, 6), (6, 12), spread=n * 0.03)
        return t, elev, homes, roads & ~lake

    def _gen_dense_forest(self):
        n, rng = self.size, self.rng
        elev = 60 * smooth_noise(n, n / 3, rng)
        t = np.full((n, n), FOREST, dtype=np.int8)
        cl = smooth_noise(n, n / 10, rng)
        t[cl > np.quantile(cl, 0.75)] = GRASS
        t[(elev > np.quantile(elev, 0.95))] = ROCK
        roads = self._roads(1)
        homes = np.zeros((n, n), bool)
        t[roads] = ROAD
        self._villages(t, homes, roads, rng.integers(1, 3), (4, 8))
        return t, elev, homes, roads

    def _gen_urban_interface(self):
        n, rng = self.size, self.rng
        yy, xx = np.mgrid[0:n, 0:n]
        side = rng.integers(4)          # which side the town is on
        coord = [xx, n - 1 - xx, yy, n - 1 - yy][side]
        town = coord < n * 0.35
        elev = 25 * (coord / n) + 10 * smooth_noise(n, n / 5, rng)    # rises towards the forest
        t = np.full((n, n), FOREST, dtype=np.int8)
        t[smooth_noise(n, n / 8, rng) > 0.65] = GRASS
        t[town] = GRASS
        roads = np.zeros((n, n), bool)
        step = max(10, n // 10)
        for k in range(step // 2, n, step):
            if side < 2:
                roads[k, :] |= town[k, :]
                cols = np.nonzero(town[0])[0]
                if k <= cols.max():
                    roads[:, k] |= town[:, k]
            else:
                roads[:, k] |= town[:, k]
                rows = np.nonzero(town[:, 0])[0]
                if k <= rows.max():
                    roads[k, :] |= town[k, :]
        roads = dilate(roads, 0) | self._roads(1)
        t[roads] = ROAD
        homes = np.zeros((n, n), bool)
        cells = np.argwhere(town & (t == GRASS))
        for _ in range(int(len(cells) * 0.05)):
            y, x = cells[rng.integers(len(cells))]
            place_houses(t, homes, rng, y, x, 1, 0.1)
        return t, elev, homes, roads

    def _gen_farmland(self):
        n, rng = self.size, self.rng
        elev = 10 * smooth_noise(n, n / 3, rng)
        t = np.full((n, n), GRASS, dtype=np.int8)
        # patchwork of rectangular fields separated by hedgerows (thin tree lines)
        sz = max(10, n // 9)
        for y0 in range(0, n, sz):
            for x0 in range(0, n, sz):
                h = int(sz * rng.uniform(0.7, 1.0)); w = int(sz * rng.uniform(0.7, 1.0))
                t[y0:y0 + h, x0:x0 + w] = FIELD if rng.random() < 0.65 else GRASS
                t[y0:y0 + h, x0] = FOREST            # hedgerows
                t[y0, x0:x0 + w] = FOREST
        wood = smooth_noise(n, n / 6, rng)
        t[wood > np.quantile(wood, 0.9)] = FOREST
        roads = self._roads(2)
        t[roads] = ROAD
        homes = np.zeros((n, n), bool)
        cells = np.argwhere(t == GRASS)
        for _ in range(rng.integers(5, 9)):          # scattered farms
            y, x = cells[rng.integers(len(cells))]
            place_houses(t, homes, rng, y, x, rng.integers(2, 4), 2)
        return t, elev, homes, roads

    def _roads(self, count):
        n, rng = self.size, self.rng
        roads = np.zeros((n, n), bool)
        dirs = [True, False] if count >= 2 else [rng.random() < 0.5]
        for horiz in dirs[:count]:
            c = rng.uniform(0.2, 0.8) * n
            k = rng.uniform(-0.25, 0.25)
            a, b = (-2, c), (n + 1, c + k * n)
            draw_line(roads, [a, b] if horiz else [(a[1], a[0]), (b[1], b[0])], max(1, n // 64))
        return roads

    def _villages(self, t, homes, along, count, houses, spread=None):
        n, rng = self.size, self.rng
        cells = np.argwhere(along)
        if len(cells) == 0:
            return
        for _ in range(count):
            cy, cx = cells[rng.integers(len(cells))]
            place_houses(t, homes, rng, cy, cx, rng.integers(*houses), spread or n * 0.04)
        t[dilate(homes, 2) & (t == FOREST)] = GRASS          # gardens around houses

    # ============================================================ setup
    def generate(self):
        rng, n = self.rng, self.size
        self.map_type = self.map_type_req if self.map_type_req != "random" else MAP_TYPES[rng.integers(len(MAP_TYPES))]
        t, elev, homes, road = getattr(self, "_gen_" + self.map_type)()
        elev = (elev - elev.min()) * self.cfg["relief"]
        self.terrain, self.elev, self.homes, self.road = t, elev.astype(np.float32), homes, road
        self.burnable = SPREAD[t] > 0
        self.home_near = dilate(homes, max(3, n // 20))
        high = elev >= np.quantile(elev, 0.7)
        self.high_ground = high
        self._spread_mult = SPREAD[t] * (1 + self.cfg["high_boost"] * high) * self.cfg["spread"]
        # slope effect for each of the 8 spread directions (source -> target)
        pe = np.pad(self.elev, 1, mode="edge")
        self._slope_fac = []
        for dx, dy in NEIGH:
            src = pe[1 - dy:1 - dy + n, 1 - dx:1 - dx + n]
            slope = (self.elev - src) / (np.hypot(dx, dy) * self.cell_m)
            self._slope_fac.append(np.clip(np.exp(self.slope_k * slope), 0.4, 3.0).astype(np.float32))
        # no-fly zones (e.g. corridors kept free for firefighting aircraft)
        self.nofly = np.zeros((n, n), bool)
        for _ in range(self.cfg["nofly"]):
            h, w = rng.integers(n // 10, n // 6, size=2)
            y0, x0 = rng.integers(n // 10, n - n // 10 - h), rng.integers(0, n - w)
            self.nofly[y0:y0 + h, x0:x0 + w] = True

    def _init_wind(self):
        n, rng, c = self.size, self.rng, self.cfg
        self.wind_dir = rng.uniform(0, 2 * np.pi)
        self.wind_speed = rng.uniform(*c["wind"])
        self.dir_var = (smooth_noise(n, n / 4, rng) - 0.5) * np.deg2rad(40)
        ridge = (self.elev - self.elev.min()) / (np.ptp(self.elev) + 1e-6)
        self.spd_var = (0.85 + 0.3 * smooth_noise(n, n / 4, rng)) * (1 + 0.3 * ridge)   # stronger on high ground
        self.shifts = sorted(rng.integers(50, 260, size=c["shifts"]).tolist())
        self._shift_left, self._gust = 0, 0.0
        self._update_wind_field()

    def _update_wind_field(self):
        ang = self.wind_dir + self.dir_var
        spd = np.clip((self.wind_speed + self._gust) * self.spd_var, 0, 1.4)
        self.wx, self.wy = (np.cos(ang) * spd).astype(np.float32), (np.sin(ang) * spd).astype(np.float32)
        s = np.hypot(self.wx, self.wy) + 1e-6
        ux, uy = self.wx / s, self.wy / s
        self._spread = []
        for k, (dx, dy) in enumerate(NEIGH):
            dist = np.hypot(dx, dy)
            cos = (dx * ux + dy * uy) / dist
            self._spread.append((self.base_p * np.exp(self.wind_k * s * cos) / dist * self._slope_fac[k])
                                .astype(np.float32))

    def wind_step(self, t):
        rng, c = self.rng, self.cfg
        self.wind_dir += rng.normal(0, 0.01)
        if t in self.shifts:                                  # weather front: wind turns over ~20 steps
            self._shift_left, self._shift = 20, rng.choice([-1, 1]) * rng.uniform(np.pi / 4, np.pi / 2)
        if self._shift_left > 0:
            self.wind_dir += self._shift / 20
            self._shift_left -= 1
        self.wind_speed = float(np.clip(self.wind_speed + rng.normal(0, 0.01), 0.15, 1.0))
        self._gust = rng.normal(0, c["gusts"]) if c["gusts"] > 0 else 0.0
        if t % 5 == 0 or self._shift_left > 0 or c["gusts"] > 0:
            self._update_wind_field()

    # ============================================================ fire
    def reset(self, ignitions=None):
        """ignitions: optional list of (row, col) start cells (default: random, from the difficulty level)"""
        self.generate()
        self._init_wind()
        n = self.size
        self.intensity = np.zeros((n, n), dtype=np.float32)
        self.burn_left = np.zeros((n, n), dtype=np.int16)
        self.burnt = np.zeros((n, n), dtype=bool)
        self.t = 0
        k = self.rng.integers(self.cfg["fires"][0], self.cfg["fires"][1] + 1)
        m = n // 8
        cand = np.argwhere(self.burnable[m:-m, m:-m] & ~self.homes[m:-m, m:-m]) + m
        cells = cand[self.rng.choice(len(cand), size=min(k, len(cand)), replace=False)] if ignitions is None \
            else np.asarray(ignitions, dtype=int).reshape(-1, 2)
        self.engine = make_fire_model(self.fire_model)
        self._fire_clock = 0.0
        if self.engine is not None:                          # external simulator owns the fire state
            self.engine.reset(self, [(int(y), int(x)) for y, x in cells])
            self._sync_engine()
            return
        for y, x in cells:
            self._ignite(np.array([y]), np.array([x]))

    @property
    def burning(self):
        return self.burn_left > 0

    def fire_map(self):
        return self.intensity * self.burning

    def _ignite(self, ys, xs):
        tt = self.terrain[ys, xs]
        self.intensity[ys, xs] = np.clip(INTENSITY[tt] * self.rng.uniform(0.85, 1.15, len(ys)), 0, 1)
        self.burn_left[ys, xs] = BURN_TIME[tt]

    def _try_ignite(self, y, x):
        n = self.size
        if 0 <= x < n and 0 <= y < n and self.burnable[y, x] and not self.burning[y, x] and not self.burnt[y, x]:
            if self.engine is not None:
                self.engine.ignite(self, y, x)
                self._sync_engine()
            else:
                self._ignite(np.array([y]), np.array([x]))

    def _sync_engine(self):
        """Copy the external engine's burning / burnt cells into the World's arrays."""
        e = self.engine
        was = self.burn_left > 0
        ys, xs = np.nonzero(e.burning & ~was)
        if len(ys):
            self._ignite(ys, xs)                             # intensity from terrain type, as in native
        ended = was & ~e.burning
        self.burn_left[ended] = 0
        self.intensity[ended] = 0
        self.burnt = e.burnt & ~e.burning

    def step(self):
        rng, n, c = self.rng, self.size, self.cfg
        self.t += 1
        self.wind_step(self.t)
        if self.engine is not None:
            self._fire_clock += self.engine.speed          # engine steps per kit step (time-scale calibration)
            while self._fire_clock >= 1.0:
                self.engine.step(self)
                self._fire_clock -= 1.0
            self._sync_engine()
            self._extra_ignitions()
            return
        new = np.zeros((n, n), dtype=bool)
        ys, xs = np.nonzero(self.burning)
        if len(ys):
            y0, y1 = max(0, ys.min() - 1), min(n, ys.max() + 2)
            x0, x1 = max(0, xs.min() - 1), min(n, xs.max() + 2)
            src = np.pad(self.fire_map(), 1)
            not_ign = np.ones((y1 - y0, x1 - x0), dtype=np.float32)
            for k, (dx, dy) in enumerate(NEIGH):
                nb = src[1 - dy + y0:1 - dy + y1, 1 - dx + x0:1 - dx + x1]
                not_ign *= 1 - np.minimum(self._spread[k][y0:y1, x0:x1] * nb, 1)
            p = (1 - not_ign) * self._spread_mult[y0:y1, x0:x1]
            can = self.burnable[y0:y1, x0:x1] & ~self.burning[y0:y1, x0:x1] & ~self.burnt[y0:y1, x0:x1]
            new[y0:y1, x0:x1] = can & (rng.random(p.shape) < p)

        self.burn_left[self.burning] -= 1
        fading = self.burning & (self.burn_left <= 4)
        self.intensity[fading] *= 0.7
        out = (self.burn_left == 0) & (self.intensity > 0)
        self.burnt |= out
        self.intensity[out] = 0
        ys, xs = np.nonzero(new)
        if len(ys):
            self._ignite(ys, xs)

        weak = self.burning & (self.intensity < 0.6)
        die = weak & (rng.random((n, n)) < self.p_die)
        self.burn_left[die] = 0
        self.intensity[die] = 0
        self.burnt |= die
        self._extra_ignitions()

    def _extra_ignitions(self):
        """Spot fires and new ignitions (kit feature, shared by every fire engine)."""
        rng, n, c = self.rng, self.size, self.cfg
        if self.burning.any():                               # spot fires downwind of intense fire
            for _ in range(rng.poisson(c["spot"] * self.wind_speed * 2)):
                cells = np.argwhere(self.burning & (self.intensity > 0.7))
                if len(cells) == 0:
                    break
                sy, sx = cells[rng.integers(len(cells))]
                d = rng.uniform(n * 0.04, n * 0.12) * self.wind_speed
                ang = np.arctan2(self.wy[sy, sx], self.wx[sy, sx]) + rng.normal(0, 0.3)
                self._try_ignite(int(round(sy + d * np.sin(ang))), int(round(sx + d * np.cos(ang))))
        if rng.random() < c["new"]:                          # lightning / human activity
            self._try_ignite(*rng.integers(0, n, size=2))

    # ============================================================ helpers for the drones / rendering
    def smoke(self):
        """Smoke: blurred fire intensity pushed downwind. 0 = clear, 1 = thick."""
        if self.cfg["smoke"] <= 0:
            return np.zeros((self.size, self.size), dtype=np.float32)
        s = box_blur(self.fire_map(), r=2)
        dx, dy = int(round(4 * np.cos(self.wind_dir))), int(round(4 * np.sin(self.wind_dir)))
        s = np.roll(np.roll(s, dy, 0), dx, 1)
        return np.clip(s * 2.5 * self.cfg["smoke"], 0, 1).astype(np.float32)

    def wind_text(self):
        deg = (np.degrees(np.arctan2(-np.sin(self.wind_dir), np.cos(self.wind_dir))) + 360) % 360
        names = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
        return names[int((deg + 22.5) // 45) % 8], self.wind_speed
