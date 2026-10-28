"""
Lattice forest-fire model of Haksar & Schwager, run through the original `simulators` package.

  R. N. Haksar, M. Schwager, "Distributed deep reinforcement learning for fighting forest fires with a
  network of aerial robots", IEEE/RSJ IROS 2018, pp. 1067-1074. doi:10.1109/IROS.2018.8593539
  Code: github.com/rhaksar/simulators (MIT), used by github.com/rhaksar/ddrl-firefighting and the
  authors' later papers on controlling / filtering large graph-based MDPs.

Model: every cell is a Tree in state healthy / on fire / burnt. A healthy tree with k burning
4-neighbours catches fire with probability 1 - (1 - alpha)^k; a burning tree stays on fire with
probability beta (mean burn time 1 / (1 - beta) steps) and otherwise burns out. Synchronous update.

Parameters: alpha = 0.2763 (the package default, used in the papers) for forest, scaled for the other
terrain types by the kit's flammability (incl. difficulty and high-ground boost); alpha = 0 for water,
road and rock. beta = exp(-1/10) (package default) for every burnable cell. The model has no wind and
no slope term: use it as the "no wind" ablation / a different family of fire dynamics.
Time scale: one lattice step every 1/SPEED kit steps (the only calibrated number).

Two engines:
  HaksarLatticeRef   builds the original simulators.fires.LatticeForest object and calls its update().
                     Faithful but slow in Python on a 128 x 128 map (use for checks and small runs).
  HaksarLatticeFast  the same Markov chain written with NumPy arrays (same transition probabilities,
                     same synchronous update); tests/test_fire_models.py checks that the two give the
                     same burnt-area statistics.
"""
import numpy as np

from .base import FireEngine

ALPHA_FOREST = 0.2763          # simulators.fires.LatticeForest default ('exponential' tree model)
BETA = float(np.exp(-1 / 10))  # simulators.fires.LatticeForest default
SPEED = 0.6                    # lattice steps per kit step (calibrated: see README, "Calibration")


def haksar_params(world):
    from world import SPREAD, FOREST
    alpha = np.clip(ALPHA_FOREST * world._spread_mult / SPREAD[FOREST], 0.0, 0.95)
    alpha[~world.burnable] = 0.0
    beta = np.where(world.burnable, BETA, 0.0)
    return alpha.astype(np.float64), beta.astype(np.float64)


class HaksarLatticeFast(FireEngine):
    name = "haksar"
    citation = "Haksar & Schwager, IROS 2018 (github.com/rhaksar/simulators)"

    def __init__(self, speed=None):
        self.speed = SPEED if speed is None else speed

    def reset(self, world, cells):
        self.rng = np.random.default_rng(self.seed_from(world))
        self.alpha, self.beta = haksar_params(world)
        n = world.size
        self.burning = np.zeros((n, n), bool)
        self.burnt = np.zeros((n, n), bool)
        for y, x in cells:
            self.burning[y, x] = True

    def step(self, world):
        b = self.burning
        k = np.zeros(b.shape, np.int8)                        # burning 4-neighbours
        k[1:, :] += b[:-1, :]
        k[:-1, :] += b[1:, :]
        k[:, 1:] += b[:, :-1]
        k[:, :-1] += b[:, 1:]
        healthy = ~b & ~self.burnt
        p_ign = 1.0 - (1.0 - self.alpha) ** k
        new = healthy & (k > 0) & (self.rng.random(b.shape) < p_ign)
        out = b & (self.rng.random(b.shape) >= self.beta)
        self.burning = (b & ~out) | new
        self.burnt = self.burnt | out

    def ignite(self, world, y, x):
        self.burning[y, x] = True


class HaksarLatticeRef(FireEngine):
    """Runs simulators.fires.LatticeForest itself (reference implementation)."""
    name = "haksar_ref"
    citation = HaksarLatticeFast.citation

    def __init__(self, speed=None):
        self.speed = SPEED if speed is None else speed

    def reset(self, world, cells):
        from simulators.fires.LatticeForest import LatticeForest
        a, b = haksar_params(world)
        n = world.size
        alpha = {(r, c): float(a[r, c]) for r in range(n) for c in range(n)}
        beta = {(r, c): float(b[r, c]) for r in range(n) for c in range(n)}
        self.sim = LatticeForest((n, n), rng=self.seed_from(world) % (2 ** 32 - 1),
                                 initial_fire=[(int(y), int(x)) for y, x in cells], alpha=alpha, beta=beta)
        self.burning = np.zeros((n, n), bool)
        self.burnt = np.zeros((n, n), bool)
        self._read()

    def _read(self):
        now = np.zeros_like(self.burning)
        if self.sim.fires:
            ys, xs = zip(*self.sim.fires)
            now[list(ys), list(xs)] = True
        self.burnt |= self.burning & ~now
        self.burning = now

    def step(self, world):
        if not self.sim.end:                                  # LatticeForest stops itself when no fire is left
            self.sim.update()
        self._read()

    def ignite(self, world, y, x):
        self.sim.group[(y, x)].set_on_fire()
        self.sim.fires.append((y, x))
        self.sim.stats[0] -= 1
        self.sim.stats[1] += 1
        self.sim.end = False
        self._read()
