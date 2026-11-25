"""
Proof of concept: ONE drone keeps an up-to-date picture of the fire.

Deliberately simple (see EXPERIMENTS / report, "proof of concept"):
  * one drone, fixed medium altitude: perfect sensor that sees a (2R+1) x (2R+1) window (R = 7 cells),
    no missed fire, no false alarms, no smoke, no heat, no battery, no no-fly zones
  * actions: N, S, E, W, hover (2 cells per move)
  * every cell is equally important (houses / forest weighting comes later)
  * the fire can be slowed down: it advances once every `fire_every` drone steps (may be fractional,
    e.g. 1.5 = two fire steps every three drone steps)
  * besides the initial fire, new fires can start during the episode (`new_rate` per fire step), so
    there is a reason to keep searching

The drone's picture of the fire is a map B (cell believed to be burning, as last seen). The truth is F.
Quality of the picture = overlap (IoU) of B and F:   acc = |B and F| / |B or F|   (1 if both empty).
acc drops when a fire is missed, when the front advances unseen, or when B still shows fire that went out.

Reward per step (all terms roughly in [0, 1]):
    r = W_ACC * acc
      + W_FIX * sum over corrected cells of m(c) / max(|F|, Z_MIN)
      + W_EXP * (EXP_FLOOR + (1 - EXP_FLOOR) * (1 - acc)) * novelty
  novelty = fraction of the view that was never seen before (v1), or, with stale=1 (v2), the mean
  staleness of the view: min(1, steps since last seen / 60), never seen = 1. EXP_FLOOR (v2) keeps part of
  the search incentive alive when the fire picture is already good (v1: EXP_FLOOR = 0).
  corrected cell = a cell whose state in B changes when the drone looks at it (fire found, front advanced,
  fire went out). m(c) = 1 + BETA * min(1, d(c) / D) for a cell that becomes "burning" in B, where d(c) is
  its distance to the nearest cell already believed to be burning (a new, separate focus is worth up to
  1 + BETA times a cell on a known front); m(c) = 1 for other corrections.

v3 (preset "v3" in train_poc.py): reward for the AGE of the information the drone refreshes, plus anti-jitter
    r = W_ACC * acc + W_FIX * fix + W_AGE * age_view - W_REV * reversed
  age_view = FIRE_MULT * sum over fire cells in view of min(1, age(c) / T_FIRE) / max(|F|, Z_MIN)
           +             sum over other cells in view of min(1, age(c) / T_EMPTY) / view area
  age(c) = steps since the cell was last seen; fire cells = cells that burn or are believed to burn.
  Looking at something just seen is worth ~0, a fire seen a while ago is worth checking again. Keeping the
  whole fire fresh pays up to FIRE_MULT / T_FIRE per step whatever its size, more than roaming over empty land
  (at most ~0.13 per step), so the fire comes first; but once it is fresh, staying there pays little: the
  drone is paid to patrol (go, leave, come back) instead of sitting on the fire.
  reversed = 1 if the move is the opposite of the previous move (back-and-forth jitter), else 0.
  Optional "goto" actions (PoCEnv(action_mode="goto")): the drone picks one of GRID x GRID map sectors and
  flies to its centre for up to HORIZON steps before deciding again (commits to a trip, no mini jumps).

v4 (preset "v4"): v3 + reversal penalty, plus a per-step cost for known fire that is being forgotten
    r = v3 reward - W_FORGET * forgotten
  forgotten (FORGET_MODE "mean") = mean over the cells of the drone's fire map of min(1, age / T_FORGET)
  forgotten (FORGET_MODE "max")  = the same mean, but for the most forgotten separate fire (connected region),
                                   following the "minimise the maximum revisit time" reward of persistent
                                   surveillance (Bana et al. 2024)
  The cost is paid at every step, also while the drone is elsewhere, so staying on one fire while another one
  ages costs more and more, while following the only fire costs nothing. In addition the age reward of a fire
  cell keeps growing up to T_FIRE = 60 steps (v3: 20), so a fire that was left long ago is worth more when it is
  seen again (idleness that grows for many steps, as in non-homogeneous patrolling, Yanes Luis et al. 2025).
  There is no direct penalty for staying on a fire: penalising the time spent in one place leads to oscillation.

v5 (preset "v5", PoCCore(altitude=True)): v4 with the w1 weights (forget cost 1.0) plus ALTITUDE and HEAT DAMAGE
  Actions: N, S, E, W, hover, up, down. Three altitudes, with the view and sensor of the full environment:
      altitude   view            sensor (per cell)                       refresh quality Q
      low        7 x 7 cells     perfect                                 1.0
      medium     15 x 15 cells   misses fire 5 %, false alarm 0.2 %       0.8
      high       31 x 31 cells   misses fire 15 %, false alarm 0.4 %      0.4
  The drone's map takes what the sensor reports, so flying high fills the map faster but with errors, which
  lowers the IoU. A look of quality Q only removes a share Q of a cell's age (a high look does not count as a
  full check) and its age reward is multiplied by Q: per cell watched, a high look is worth 0.4 of a low one.
  High altitude still pays more for scanning empty land (4x more cells), medium pays more for keeping the
  fire itself up to date. The corrections term only pays for cells whose map entry becomes RIGHT and charges the cells
  that become WRONG, and the bonus for a new focus far from known fire is paid only the first time a burning
  cell is detected, so re-finding a cell that a sensor error had removed nets zero: noise cannot be farmed.
  Heat: at low altitude, if any cell within 1 of the drone burns with intensity above HEAT (0.7) the drone
  takes damage DMG_RATE * intensity per step (0.25 -> lost after about 4 steps over intense fire). Reward
  -W_DMG * damage, and -W_LOSS when it is lost. A lost drone ends the training episode; in evaluation the fire
  keeps burning until the end of the mission and the drone's map just gets older.
  Medium and high altitude are safe. Start at medium altitude. Observation: + altitude and health channels.

Evaluation (metrics(), the same for every policy):
    iou           mean acc over the episode (main metric, higher is better)
    t_first       steps until the first fire is in the drone's map (censored at max_steps)
    foci_found    fraction of fire foci (initial ignitions + new fires far from existing fire) that were seen
    foci_latency  mean steps from the start of a focus until it is seen (unseen foci count as censored)
    hover_frac    fraction of steps the drone did not move
    seen_frac     fraction of the map seen at least once
    fire_age      mean age (steps since last seen, capped at 60) of the cells that are burning (lower = fresher)
    dwell_frac    fraction of steps spent re-watching known fire seen <= 5 steps ago without learning anything
    revisit_gap   mean number of steps between two looks at the same burning cell (1 = watched all the time)
    rev_frac      fraction of moves that reverse the previous move (back-and-forth jitter)
    forgotten     mean share of the drone's fire map not seen for more than T_FORGET (30) steps
    oldest_fire   mean age of the most forgotten separate fire on the drone's map (steps, capped at 60)
    alt_low / alt_med / alt_high   share of steps at each altitude (altitude mode only)
    damage        total heat damage taken (1 = one drone's worth)
    lost          1 if the drone was lost to heat, else 0
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.ndimage import distance_transform_edt, label

from world import World, DIFFICULTY, MAP_TYPES

OBS = 32
N_CH = 8
R = 7                                     # view radius (medium altitude in the full environment)
MOVE = 2
N_, S_, E_, W_, HOVER = range(5)
MOVES = {N_: (0, -1), S_: (0, 1), E_: (1, 0), W_: (-1, 0), HOVER: (0, 0)}
DEFAULT_W = dict(acc=1.0, fix=1.0, exp=0.5, beta=3.0, d=10.0, z_min=10.0, exp_floor=0.0, stale=0.0,
                 age=0.0, fire_mult=10.0, t_fire=20.0, t_empty=60.0, rev=0.0,
                 forget=0.0, forget_mode="mean", t_forget=30.0, dmg=0.0, loss=0.0)
V3_W = dict(acc=0.3, fix=1.0, exp=0.0, age=1.0, stale=1.0)       # changes on top of DEFAULT_W for preset v3
V4_W = dict(V3_W, rev=0.1, t_fire=60.0, forget=0.5)               # changes on top of DEFAULT_W for preset v4
V5_W = dict(V4_W, forget=1.0, dmg=1.0, loss=5.0)                  # v4 w1 + heat damage, for preset v5
EIGHT = np.ones((3, 3), dtype=int)                                # 8-connectivity for separate fire regions
DWELL_AGE = 5
# altitude mode (values of the full environment, env.py): low, medium, high
R_ALT = [3, 7, 15]
MISS_ALT = [0.0, 0.05, 0.15]
FA_ALT = [0.0, 0.002, 0.004]
Q_ALT = [1.0, 0.8, 0.4]
HEAT = 0.7                                # intensity above which low flight damages the drone
DMG_RATE = 0.25                           # damage per step = DMG_RATE * intensity (health starts at 1)
UP, DOWN = 5, 6
FOCUS_D = 10                              # cells: a new fire farther than this from existing fire is a new focus
AGE_CAP = 60


def _as_list(x, allowed):
    if isinstance(x, str):
        x = allowed if x == "all" else x.split(",")
    return list(x)


class PoCCore:
    def __init__(self, size=64, max_steps=300, fire_every=4, new_rate=0.01, map_types="all", weights=None,
                 fire_model="native", altitude=False, end_on_loss=False):
        assert size % OBS == 0
        self.altitude, self.end_on_loss = altitude, end_on_loss
        self.n_ch = N_CH + (2 if altitude else 0)
        self.n_actions = 7 if altitude else 5
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
        self.det_ever = np.zeros((n, n), dtype=bool)         # burning cells correctly detected at least once
        self.burnt_known = np.zeros((n, n), dtype=bool)
        self.seen = np.zeros((n, n), dtype=bool)
        self.last_seen = np.full((n, n), -10 ** 6, dtype=float)
        self.F_ever = w.burning.copy()
        self.foci = [[int(y), int(x), 0, -1] for y, x in np.argwhere(w.burning)]   # y, x, t_start, t_seen
        self.acc_sum, self.hover = 0.0, 0
        self.prev_dir, self.moves, self.reversals = None, 0, 0
        self.dwell, self.gap_sum, self.gap_n = 0, 0.0, 0
        self.fire_age_sum, self.fire_steps = 0.0, 0
        self.forgot_sum, self.oldest_sum, self.known_steps = 0.0, 0.0, 0
        self.alt, self.health, self.alive = 1, 1.0, True
        self.alt_steps = np.zeros(3)
        self.damage_sum = 0.0
        self.sense_rng = np.random.default_rng(None if seed is None else seed + 1_000_003)   # sensor noise only
        self._fire_acc = 0.0
        self.t_first = None
        self.last_reward = {}
        self._sense()
        return self

    def acc(self):
        F = self.world.burning
        union = (self.B | F).sum()
        return 1.0 if union == 0 else float((self.B & F).sum() / union)

    def radius(self):
        return R_ALT[self.alt] if self.altitude else R

    def _window(self):
        x, y = self.pos
        n, r = self.size, self.radius()
        return max(0, y - r), min(n, y + r + 1), max(0, x - r), min(n, x + r + 1)

    def _sense(self):
        """The drone looks. Returns (corrected-cell mass, novelty of the view (v1/v2), age reward of the view (v3),
        True if it only re-watched fresh known fire)."""
        w, wt = self.world, self.w
        y0, y1, x0, x1 = self._window()
        F = w.burning[y0:y1, x0:x1]
        Bw = self.B[y0:y1, x0:x1]
        age = (self.t - self.last_seen[y0:y1, x0:x1]).astype(float)
        if self.altitude and self.alt > 0:                         # noisy sensor at medium / high altitude
            u = self.sense_rng.random(F.shape)
            obs = np.where(F, u >= MISS_ALT[self.alt], u < FA_ALT[self.alt])
        else:
            obs = F
        q = Q_ALT[self.alt] if self.altitude else 1.0
        changed = Bw != obs
        fix = 0.0
        if changed.any():
            improve = (obs == F) & (Bw != F)                       # map entry becomes right
            worsen = (obs != F) & (Bw == F)                        # map entry becomes wrong (sensor error)
            wgt = np.ones(F.shape)
            hot = improve & F & ~self.det_ever[y0:y1, x0:x1]       # first detection of a burning cell: focus bonus
            if hot.any():
                if self.B.any():
                    d = distance_transform_edt(~self.B)[y0:y1, x0:x1][hot]
                else:
                    d = np.full(int(hot.sum()), wt["d"])
                wgt[hot] = 1.0 + wt["beta"] * np.minimum(1.0, d / wt["d"])
            fix = float(wgt[improve].sum() - wgt[worsen].sum())
        if wt["stale"]:
            novelty = float(np.minimum(1.0, age / AGE_CAP).mean())
        else:
            novelty = float((~self.seen[y0:y1, x0:x1]).mean())
        fireish = F | Bw
        z = max(float(w.burning.sum()), wt["z_min"])
        age_c = np.minimum(age, AGE_CAP)
        age_new = (1.0 - q) * age_c                                # a look of quality q removes a share q of the age
        gain = lambda T, m: np.minimum(1.0, age_c[m] / T) - np.minimum(1.0, age_new[m] / T)
        age_r = q * float(wt["fire_mult"] * gain(wt["t_fire"], fireish).sum() / z
                          + gain(wt["t_empty"], ~fireish).sum() / (2 * R + 1) ** 2)
        known = F & Bw & self.seen[y0:y1, x0:x1]                  # burning cells already known and looked at again
        dwell = bool((known & (age <= DWELL_AGE)).any()) and not changed.any()
        g = age[known]
        self.gap_sum += float(np.minimum(g, AGE_CAP).sum())
        self.gap_n += int(g.size)
        self.B[y0:y1, x0:x1] = obs
        self.det_ever[y0:y1, x0:x1] |= obs & F
        self.burnt_known[y0:y1, x0:x1] = w.burnt[y0:y1, x0:x1]
        self.seen[y0:y1, x0:x1] = True
        self.last_seen[y0:y1, x0:x1] = self.t - age_new
        if self.t_first is None and self.B.any():
            self.t_first = self.t
        for f in self.foci:                                        # focus point inside the view
            if f[3] < 0 and y0 <= f[0] < y1 and x0 <= f[1] < x1:
                f[3] = self.t
        return fix, novelty, age_r, dwell

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
        action = int(action)
        if not self.alive:                                         # lost drone (evaluation): the fire goes on
            action = HOVER
        if action in (UP, DOWN):
            self.alt = int(np.clip(self.alt + (1 if action == UP else -1), 0, 2))
            dx, dy = 0, 0
        else:
            dx, dy = MOVES[action]
            old = self.pos.copy()
            self.pos = np.clip(self.pos + MOVE * np.array([dx, dy]), 0, n - 1)
            self.hover += int((self.pos == old).all())
        reversed_ = 0
        if (dx, dy) != (0, 0):
            self.moves += 1
            if self.prev_dir is not None and (dx, dy) == (-self.prev_dir[0], -self.prev_dir[1]):
                reversed_ = 1
                self.reversals += 1
            self.prev_dir = (dx, dy)
        self.t += 1
        self._fire_acc += 1.0 / self.fire_every
        while self._fire_acc >= 1.0 - 1e-9:
            self._fire_acc -= 1.0
            before = self.F_ever.copy()
            w.step()
            self.F_ever |= w.burning
            self._track_foci(before)
        dmg, lost = 0.0, 0
        if self.alive:
            self.alt_steps[self.alt] += 1
            fix, novelty, age_r, dwell = self._sense()
            if self.altitude and self.alt == 0:                    # heat at low altitude
                x, y = self.pos
                heat = (w.intensity * w.burning)[max(0, y - 1):y + 2, max(0, x - 1):x + 2].max()
                if heat > HEAT:
                    dmg = min(self.health, DMG_RATE * float(heat))
                    self.health -= dmg
                    self.damage_sum += dmg
                    if self.health <= 1e-9:
                        self.alive, lost = False, 1
        else:
            fix, novelty, age_r, dwell = 0.0, 0.0, 0.0, False
        self.dwell += int(dwell)
        acc = self.acc()
        if w.burning.any():
            self.fire_age_sum += float(np.minimum(AGE_CAP, self.t - self.last_seen[w.burning]).mean())
            self.fire_steps += 1
        forgotten = self._forgotten()
        z = max(float(w.burning.sum()), wt["z_min"])
        gate = wt["exp_floor"] + (1.0 - wt["exp_floor"]) * (1.0 - acc)
        parts = dict(acc=acc, fix=fix / z, exp=gate * novelty, age=age_r, rev=float(reversed_), forget=forgotten,
                     dmg=dmg, loss=float(lost))
        r = (wt["acc"] * parts["acc"] + wt["fix"] * parts["fix"] + wt["exp"] * parts["exp"]
             + wt["age"] * parts["age"] - wt["rev"] * parts["rev"] - wt["forget"] * parts["forget"]
             - wt["dmg"] * parts["dmg"] - wt["loss"] * parts["loss"])
        self.acc_sum += acc
        self.last_reward = parts
        done = self.t >= self.max_steps or (lost and self.end_on_loss)
        return r, done, parts

    def _forgotten(self):
        """Cost of the known fire that is being forgotten (also records the forgotten / oldest_fire metrics)."""
        wt = self.w
        if not self.B.any():
            return 0.0
        age = np.minimum(AGE_CAP, self.t - self.last_seen[self.B]).astype(float)
        self.forgot_sum += float((age > wt["t_forget"]).mean())
        lab, k = label(self.B, structure=EIGHT)
        ids = lab[self.B]
        region_age = np.bincount(ids, weights=age, minlength=k + 1)[1:] / np.maximum(np.bincount(ids, minlength=k + 1)[1:], 1)
        self.oldest_sum += float(region_age.max())
        self.known_steps += 1
        if not wt["forget"]:
            return 0.0
        if wt["forget_mode"] == "max":
            return float(min(1.0, region_age.max() / wt["t_forget"]))
        return float(np.minimum(1.0, age / wt["t_forget"]).mean())

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
        o = np.empty((self.n_ch, OBS, OBS), dtype=np.float32)
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
        if self.altitude:
            o[8] = self.alt / 2.0
            o[9] = self.health
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
                    seen_frac=float(self.seen.mean()),
                    fire_age=self.fire_age_sum / self.fire_steps if self.fire_steps else 0.0,
                    dwell_frac=self.dwell / T,
                    revisit_gap=self.gap_sum / self.gap_n if self.gap_n else float(AGE_CAP),
                    rev_frac=self.reversals / max(self.moves, 1),
                    forgotten=self.forgot_sum / self.known_steps if self.known_steps else 0.0,
                    oldest_fire=self.oldest_sum / self.known_steps if self.known_steps else 0.0,
                    alt_low=self.alt_steps[0] / T, alt_med=self.alt_steps[1] / T, alt_high=self.alt_steps[2] / T,
                    damage=self.damage_sum, lost=float(not self.alive))


class Goto:
    """'goto' actions: sector k of a GRID x GRID partition of the map; the drone flies to the sector centre for
    up to `horizon` steps (stops early when it arrives; if it is already there it hovers one step)."""

    def __init__(self, size, grid=8, horizon=8):
        self.size, self.grid, self.horizon = size, grid, horizon
        self.cell = size // grid
        self.k = None
        self.left = 0

    def target(self, k):
        k = int(k)
        return (k % self.grid) * self.cell + self.cell // 2, (k // self.grid) * self.cell + self.cell // 2

    def start(self, k):
        self.k, self.left = int(k), self.horizon

    def arrived(self, pos):
        tx, ty = self.target(self.k)
        return abs(pos[0] - tx) <= 1 and abs(pos[1] - ty) <= 1

    def low_action(self, pos):
        """Next primitive action of the current trip, or None if the trip is over."""
        if self.left <= 0:
            return None
        if self.arrived(pos):
            if self.left < self.horizon:            # arrived during the trip: the trip ends
                self.left = 0
                return None
            self.left = 0                           # already there when chosen: hover one step
            return HOVER
        self.left -= 1
        return step_towards(pos, self.target(self.k))


class PoCEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, size=64, max_steps=300, fire_every=4, new_rate=0.01, map_types="all", weights=None,
                 action_mode="primitive", grid=8, horizon=8, altitude=False):
        self.core = PoCCore(size=size, max_steps=max_steps, fire_every=fire_every, new_rate=new_rate,
                            map_types=map_types, weights=weights, altitude=altitude, end_on_loss=True)
        assert not (altitude and action_mode == "goto"), "goto actions are not combined with altitude"
        self.observation_space = spaces.Box(0.0, 1.0, shape=(self.core.n_ch, OBS, OBS), dtype=np.float32)
        self.action_mode = action_mode
        if action_mode == "goto":
            self.goto = Goto(size, grid, horizon)
            self.action_space = spaces.Discrete(grid * grid)
        else:
            self.action_space = spaces.Discrete(self.core.n_actions)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(2 ** 31))
        self.core.reset(seed=seed)
        return self.core.observe(), {}

    def step(self, action):
        if self.action_mode == "goto":
            self.goto.start(action)
            r, done, n_low = 0.0, False, 0
            while not done:
                a = self.goto.low_action(self.core.pos)
                if a is None:
                    break
                rr, done, parts = self.core.step(a)
                r += rr
                n_low += 1
            info = dict(parts, low_steps=n_low)
        else:
            r, done, parts = self.core.step(action)
            info = dict(parts, low_steps=1)
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
        return int(self.rng.integers(core.n_actions))


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
