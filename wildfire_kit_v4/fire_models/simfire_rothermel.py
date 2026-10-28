"""
Rothermel surface-fire spread as implemented in MITRE's SimFire.

  A. Tapley, M. Dotter, M. Doyle, A. Fennelly, D. Gandikota, S. Smith, M. Threet, T. Welsh,
  "Reinforcement learning for wildfire mitigation in simulated disaster environments",
  NeurIPS 2023 Workshop on Tackling Climate Change with ML. arXiv:2311.15925
  Code: github.com/mitrefireline/simfire (Apache-2.0), used by github.com/mitrefireline/simharness
  Physics: R. C. Rothermel, "A mathematical model for predicting fire spread in wildland fuels",
  USDA Forest Service INT-115, 1972; fuel models from P. L. Andrews, RMRS-GTR-371, 2018.

What is used unchanged from SimFire (third_party/simfire_world, commit 04905b1):
  - rothermel.compute_rate_of_spread()     the Rothermel equations (wind + slope factors)
  - presets.*                              Anderson fuel models (ShortGrass, TallGrass, ...)
  - parameters.FuelParticle                fuel particle constants
`RothermelGrid` re-implements SimFire's RothermelFireManager.update() with NumPy arrays instead of
one pygame sprite per burning pixel (SimFire's version needs Python 3.9 + pygame and is too slow
for RL). The update rule is reproduced exactly, including its details:
  - 8-neighbour spread, rate of spread computed for the target cell's fuel, wind and slope
  - a target reached by several burning cells keeps the value of the most recently ignited one
    (SimFire assigns with NumPy fancy indexing, so the last sprite in its list wins)
  - burn_amounts accumulate R * update_rate; a cell ignites once burn_amounts > pixel_scale
  - a burning cell is removed after max_fire_duration updates
tests/validate_simfire_reference.py runs SimFire's own RothermelFireManager side by side and checks
that both produce identical fire maps at every step.

Kit -> SimFire mapping (adapter choices):
  grassland -> ShortGrass (Anderson FM1)      farmland -> TallGrass (FM3)
  forest    -> TimberLitterUnderstory (FM10)  houses   -> NBUrban (non-burnable, as in SimFire)
  water / road / rock -> NBWater / NBUrban / NBBarren
  cell size 5 m -> pixel_scale in ft; elevation m -> ft; 1 kit wind unit = 20 mph midflame wind
  (SimFire's default wind range is 7-47 mph); dead fuel moisture per difficulty level (SimFire's
  functional configs use 0.03); one step = `update_rate` minutes (default 0.08 min = 4.8 s, chosen so
  that a 'medium' fire grows about as fast as the kit's native fire, see compare_simulators.py);
  max_fire_duration = SimFire's 5 minutes, i.e. 62 steps.

Known SimFire issue (found while testing, see tests/test_fire_models.py::test_slope): the sign of the
slope term in compute_rate_of_spread is inverted with respect to SimFire's own slope direction, so in
SimFire a fire runs faster DOWNhill. The adapter corrects this by default (fix_slope_sign=True) by
handing SimFire's function the slope aspect rotated by 180 degrees; SimFire's code is not modified.
Use SimFireRothermel(fix_slope_sign=False) to reproduce SimFire exactly.
"""
import numpy as np

from simfire_world.rothermel import compute_rate_of_spread
from simfire_world.parameters import FuelParticle
from simfire_world import presets

from .base import FireEngine

FT_PER_M = 3.28084
FTPM_PER_MPH = 88.0
UNBURNED, BURNING, BURNED = 0, 1, 2
# SimFire's 8-connected order in RothermelFireManager._get_new_locs: (dx, dy), image coordinates
SIMFIRE_DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
SIMFIRE_DIRS_4 = [(1, 0), (0, 1), (-1, 0), (0, -1)]


class RothermelGrid:
    """
    Array version of simfire.game.managers.fire.RothermelFireManager (NumPy only).

    fuel: dict of arrays w_0, delta, M_x, sigma (H x W); elevation: H x W (same length unit as
    pixel_scale, i.e. ft); U: midflame wind (ft/min), U_dir: degrees, both H x W (can be changed
    between steps); M_f: dead fuel moisture.
    """

    def __init__(self, fuel, elevation, pixel_scale, update_rate, max_fire_duration, M_f,
                 U, U_dir, fuel_particle=None, diagonal_spread=True, fix_slope_sign=False):
        self.fuel = {k: np.asarray(v) for k, v in fuel.items()}
        self.H, self.W = self.fuel["w_0"].shape
        self.pixel_scale, self.update_rate = pixel_scale, update_rate
        self.max_fire_duration, self.M_f = max_fire_duration, M_f
        self.U, self.U_dir = np.asarray(U, dtype=np.float64), np.asarray(U_dir, dtype=np.float64)
        self.fp = fuel_particle or FuelParticle()
        self.dirs = SIMFIRE_DIRS if diagonal_spread else SIMFIRE_DIRS_4
        # RothermelFireManager._compute_slopes
        grad_y, grad_x = np.gradient(np.asarray(elevation, dtype=np.float64), pixel_scale)
        self.slope_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        self.slope_dir = np.arctan2(grad_y, grad_x + 0.000001)
        if fix_slope_sign:
            # SimFire projects the slope as -|slope| * cos(slope_dir + travel_angle); with its own
            # slope_dir (steepest ascent, image coordinates) that is negative when travelling UPHILL,
            # so its fire spreads faster downhill. Rotating the aspect by 180 degrees restores
            # Rothermel's convention (uphill faster) without touching SimFire's function.
            self.slope_dir = self.slope_dir + np.pi
        self.fire_map = np.zeros((self.H, self.W), np.int8)
        self.burn_amounts = np.zeros((self.H, self.W), np.float64)
        self.duration = np.zeros((self.H, self.W), np.int64)
        self.order = np.full((self.H, self.W), -1, np.int64)     # position in SimFire's sprite list
        self._next = 0
        self._yy, self._xx = np.mgrid[0:self.H, 0:self.W]

    # ---------------------------------------------------------------- sprites
    def add_fire(self, x, y):
        """Equivalent of appending a Fire sprite (duration 0) and marking the pixel BURNING."""
        self.fire_map[y, x] = BURNING
        self.duration[y, x] = 0
        self.order[y, x] = self._next
        self._next += 1

    # ---------------------------------------------------------------- one update
    def update(self):
        fm = self.fire_map
        # FireManager._prune_sprites: sprites whose duration >= max_fire_duration -> BURNED
        expired = (fm == BURNING) & (self.duration >= self.max_fire_duration)
        fm[expired] = BURNED
        self.order[expired] = -1
        burning = fm == BURNING
        self.duration[burning] += 1                       # durations incremented after pruning
        if not burning.any():
            return False
        H, W = self.H, self.W
        # for every unburned target, the burning neighbour that SimFire would use (last sprite wins)
        best = np.full((H, W), -1, np.int64)
        src_x = np.zeros((H, W), np.int64)
        src_y = np.zeros((H, W), np.int64)
        unburned = fm == UNBURNED
        for dx, dy in self.dirs:
            # source s, target t = s + (dx, dy)  ->  for target t the source is t - (dx, dy)
            sy0, sy1 = max(0, -dy), H - max(0, dy)          # rows of sources that have an in-bound target
            sx0, sx1 = max(0, -dx), W - max(0, dx)
            s_ord = np.where(burning[sy0:sy1, sx0:sx1], self.order[sy0:sy1, sx0:sx1], -1)
            ty, tx = slice(sy0 + dy, sy1 + dy), slice(sx0 + dx, sx1 + dx)
            cand = unburned[ty, tx] & (s_ord > best[ty, tx])
            best[ty, tx] = np.where(cand, s_ord, best[ty, tx])
            src_x[ty, tx] = np.where(cand, self._xx[sy0:sy1, sx0:sx1], src_x[ty, tx])
            src_y[ty, tx] = np.where(cand, self._yy[sy0:sy1, sx0:sx1], src_y[ty, tx])
        ty, tx = np.nonzero(best >= 0)
        if len(ty) == 0:
            return True
        f32 = lambda a: np.asarray(a, dtype=np.float32)        # SimFire flattens all inputs to float32
        nloc = len(ty)
        R = compute_rate_of_spread(
            f32(src_x[ty, tx]), f32(src_y[ty, tx]), f32(tx), f32(ty),
            f32(self.fuel["w_0"][ty, tx]), f32(self.fuel["delta"][ty, tx]),
            f32(self.fuel["M_x"][ty, tx]), f32(self.fuel["sigma"][ty, tx]),
            f32(np.full(nloc, self.fp.h)), f32(np.full(nloc, self.fp.S_T)),
            f32(np.full(nloc, self.fp.S_e)), f32(np.full(nloc, self.fp.p_p)),
            f32(np.full(nloc, self.M_f)),
            f32(self.U[ty, tx]), f32(self.U_dir[ty, tx]),
            f32(self.slope_mag[ty, tx]), f32(self.slope_dir[ty, tx]),
        )
        R = R * self.update_rate
        self.burn_amounts[ty, tx] += R
        # FireManager._update_with_new_locs: new sprites in np.unique((y, x)) order
        new = self.burn_amounts[ty, tx] > self.pixel_scale
        ny, nx = ty[new], tx[new]                              # np.nonzero is already (y, x)-sorted
        fm[ny, nx] = BURNING
        self.duration[ny, nx] = 0
        self.order[ny, nx] = self._next + np.arange(len(ny))
        self._next += len(ny)
        return True


# ---------------------------------------------------------------------------------- kit adapter
def kit_fuels(terrain):
    from world import WATER, ROAD, ROCK, GRASS, FOREST, HOUSE, FIELD
    table = {WATER: presets.NBWater, ROAD: presets.NBUrban, ROCK: presets.NBBarren,
             GRASS: presets.ShortGrass, FOREST: presets.TimberLitterUnderstory,
             HOUSE: presets.NBUrban, FIELD: presets.TallGrass}
    out = {k: np.zeros(terrain.shape, np.float64) for k in ("w_0", "delta", "M_x", "sigma")}
    for t, fuel in table.items():
        m = terrain == t
        for k in out:
            out[k][m] = getattr(fuel, k)
    return out


MOISTURE = {"easy": 0.08, "medium": 0.05, "hard": 0.04, "extreme": 0.03}


class SimFireRothermel(FireEngine):
    name = "rothermel"
    citation = "Rothermel 1972 via MITRE SimFire (Tapley et al., NeurIPS 2023 CCAI workshop)"

    def __init__(self, update_rate=0.08, max_fire_minutes=5.0, mph_per_unit=20.0, moisture=None,
                 fix_slope_sign=True):
        self.update_rate, self.max_fire_minutes = update_rate, max_fire_minutes
        self.mph_per_unit, self.moisture = mph_per_unit, moisture
        self.fix_slope_sign = fix_slope_sign

    def _wind(self, world):
        speed = np.hypot(world.wx, world.wy).astype(np.float64)
        U = speed * self.mph_per_unit * FTPM_PER_MPH
        to_math = np.degrees(np.arctan2(-world.wy, world.wx))        # direction the wind blows to
        return U, (90.0 - to_math) % 360.0                             # SimFire: 0 = N, 90 = E

    def reset(self, world, cells):
        self._seed = self.seed_from(world)          # not used (SimFire is deterministic), keeps RNG use uniform
        U, U_dir = self._wind(world)
        self.grid = RothermelGrid(
            fuel=kit_fuels(world.terrain), elevation=world.elev.astype(np.float64) * FT_PER_M,
            pixel_scale=world.cell_m * FT_PER_M, update_rate=self.update_rate,
            max_fire_duration=int(round(self.max_fire_minutes / self.update_rate)),
            M_f=self.moisture if self.moisture is not None else MOISTURE.get(world.difficulty, 0.05),
            U=U, U_dir=U_dir, fix_slope_sign=self.fix_slope_sign)
        for y, x in cells:
            if self.grid.fuel["w_0"][y, x] > 0:
                self.grid.add_fire(int(x), int(y))
        self._read()

    def _read(self):
        self.burning = self.grid.fire_map == BURNING
        self.burnt = self.grid.fire_map == BURNED

    def step(self, world):
        self.grid.U, self.grid.U_dir = self._wind(world)               # SimFire allows wind maps
        self.grid.update()
        self._read()

    def ignite(self, world, y, x):
        if self.grid.fuel["w_0"][y, x] > 0 and self.grid.fire_map[y, x] == UNBURNED:
            self.grid.add_fire(int(x), int(y))
            self._read()
