"""
Sanity and equivalence checks for the fire engines (run: python tests/test_fire_models.py, ~2-4 min).

  1. wind:   every engine with a wind term spreads further downwind than upwind (east and south wind)
  2. slope:  engines with a slope term spread further uphill
  3. haksar: the fast NumPy engine and the original simulators.fires.LatticeForest give the same
             distribution of burnt area (two-sample Kolmogorov-Smirnov test, several time points)
  4. reproducibility: same seed -> same fire, whatever else uses the random generators in between
  5. integration: a full 3-drone episode with the greedy baseline runs on every engine
"""
import os
import sys

import numpy as np
import torch
from scipy.stats import ks_2samp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from world import World, FOREST, DIFFICULTY            # noqa: E402
from env import SwarmCore                               # noqa: E402
from baselines import Greedy                            # noqa: E402

N = 64


class TestWorld(World):
    """Uniform forest on a plane; fixed wind; for directional checks."""

    def __init__(self, wind_to=(1.0, 0.0), wind_speed=0.8, uphill=None, **kw):
        super().__init__(size=N, **kw)
        self.wind_to, self.wind_speed0, self.uphill = wind_to, wind_speed, uphill

    def _gen_test(self):
        t = np.full((N, N), FOREST, dtype=np.int8)
        yy, xx = np.mgrid[0:N, 0:N]
        elev = np.zeros((N, N)) if self.uphill is None else 0.25 * (self.uphill[0] * xx + self.uphill[1] * yy) * self.cell_m
        z = np.zeros((N, N), bool)
        return t, elev, z, z

    def _init_wind(self):
        super()._init_wind()
        self.cfg = dict(self.cfg, gusts=0.0, shifts=0, spot=0.0, new=0.0)
        self.shifts = []
        self.wind_dir = float(np.arctan2(self.wind_to[1], self.wind_to[0]))
        self.wind_speed = self.wind_speed0
        self.dir_var[:] = 0.0
        self.spd_var[:] = 1.0
        self._update_wind_field()


def spread_centroid(fm, steps=70, **kw):
    w = TestWorld(rng=np.random.default_rng(0), map_type="test", difficulty="easy", fire_model=fm, **kw)
    w.cfg = dict(DIFFICULTY["easy"], relief=1.0, high_boost=0.0)
    w.reset(ignitions=[(N // 2, N // 2)])
    for _ in range(steps):
        w.step()
    ys, xs = np.nonzero(w.burning | w.burnt)
    return xs.mean() - N // 2, ys.mean() - N // 2, len(xs)


def test_wind():
    for fm in ["native", "pyrorl", "rothermel"]:
        dx, dy, n = spread_centroid(fm, wind_to=(1.0, 0.0))
        assert dx > 2 and abs(dy) < dx, (fm, dx, dy)
        dx2, dy2, _ = spread_centroid(fm, wind_to=(0.0, 1.0))        # image y points down = south
        assert dy2 > 2 and abs(dx2) < dy2, (fm, dx2, dy2)
        print(f"  wind   {fm:10s} east wind: centroid shift x={dx:+.1f} y={dy:+.1f} cells ({n} cells)   "
              f"south wind: x={dx2:+.1f} y={dy2:+.1f}")
    dx, dy, n = spread_centroid("haksar", wind_to=(1.0, 0.0))
    print(f"  wind   haksar     (no wind term by design): centroid shift x={dx:+.1f} y={dy:+.1f}")


def test_slope():
    # native: whole-fire check with a weak cross-slope wind
    for up in [(1.0, 0.0), (0.0, -1.0)]:                           # rising to the east / to the north
        dx, dy, n = spread_centroid("native", wind_to=(up[1], up[0]), wind_speed=0.15, uphill=up)
        along = dx * up[0] + dy * up[1]
        assert along > 1.5, ("native", up, dx, dy)
        print(f"  slope  native     terrain rising to {'east ' if up[0] else 'north'} (14 deg), weak cross-slope "
              f"wind: centroid {along:+.1f} cells uphill")
    # rothermel: rate of spread from one burning cell to its uphill / downhill neighbour, no wind
    # (forest fuel without wind is too slow to show a shape within a short test)
    from fire_models.simfire_rothermel import RothermelGrid, FT_PER_M
    from simfire_world import presets
    f = presets.TimberLitterUnderstory
    fuel = {k: np.full((9, 9), getattr(f, k)) for k in ("w_0", "delta", "M_x", "sigma")}
    for up, name in [((1, 0), "east "), ((0, -1), "north")]:
        yy, xx = np.mgrid[0:9, 0:9]
        elev = 0.25 * 5 * FT_PER_M * (up[0] * xx + up[1] * yy)       # 14 deg plane, ft
        res = {}
        for fix in (True, False):
            g = RothermelGrid(fuel, elev, 5 * FT_PER_M, 1.0, 10, 0.05, np.zeros((9, 9)), np.zeros((9, 9)),
                              fix_slope_sign=fix)
            g.add_fire(4, 4)
            g.update()
            res[fix] = (g.burn_amounts[4 + up[1], 4 + up[0]], g.burn_amounts[4 - up[1], 4 - up[0]])
        assert res[True][0] > 1.5 * res[True][1], res
        print(f"  slope  rothermel  plane rising to {name}, no wind: ROS uphill {res[True][0]:.2f} vs downhill "
              f"{res[True][1]:.2f} ft/min   (SimFire as published: {res[False][0]:.2f} vs {res[False][1]:.2f})")


def burnt_area_samples(fm, n_eps, times=(30, 60, 120)):
    out = []
    for s in range(n_eps):
        w = World(size=N, rng=np.random.default_rng(10_000 + s), map_type="dense_forest",
                  difficulty="easy", fire_model=fm)
        w.reset()
        row = []
        for t in range(1, max(times) + 1):
            w.step()
            if t in times:
                row.append((w.burning | w.burnt).sum())
        out.append(row)
    return np.array(out)


def test_haksar_equivalence(n_eps=150):
    fast = burnt_area_samples("haksar", n_eps)
    ref = burnt_area_samples("haksar_ref", n_eps)
    for k, t in enumerate((30, 60, 120)):
        p = ks_2samp(fast[:, k], ref[:, k]).pvalue
        print(f"  haksar step {t:3d}: mean burnt cells fast {fast[:, k].mean():7.1f} | original "
              f"{ref[:, k].mean():7.1f}   KS p-value {p:.2f}")
        assert p > 0.01, (t, p)


def test_reproducible():
    for fm in ["native", "pyrorl", "haksar", "rothermel"]:
        maps = []
        for disturb in (False, True):
            w = World(size=N, rng=np.random.default_rng(5), map_type="rolling_hills", difficulty="hard",
                      fire_model=fm)
            w.reset()
            for _ in range(80):
                if disturb:                                  # other code using global RNGs (e.g. PPO)
                    torch.rand(1000)
                    np.random.rand(1000)
                w.step()
            maps.append(w.fire_map().copy())
        assert np.array_equal(maps[0], maps[1]), fm
    print("  same seed -> identical fire for every engine, even with other random draws in between")


def test_episode():
    for fm in ["native", "pyrorl", "haksar", "rothermel"]:
        core = SwarmCore(n_drones=3, max_steps=120, fire_models=fm, difficulty="hard")
        core.reset(seed=3)
        pol = Greedy()
        pol.reset(core)
        ret, done = 0.0, False
        while not done:
            r, done, info = core.step([pol.act(core, i) for i in range(3)])
            ret += r
        assert np.isfinite(ret)
        print(f"  episode {fm:10s} greedy x3, hard, 120 steps: return {ret:6.2f}")


if __name__ == "__main__":
    for f in [test_wind, test_slope, test_haksar_equivalence, test_reproducible, test_episode]:
        print(f.__name__)
        f()
    print("\nall fire-model checks passed")
