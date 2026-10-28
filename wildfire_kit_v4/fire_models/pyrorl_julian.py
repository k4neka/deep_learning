"""
Julian & Kochenderfer (2019) stochastic wildfire model, run through PyroRL's own code.

  K. D. Julian, M. J. Kochenderfer, "Distributed wildfire surveillance with autonomous aircraft using
  deep reinforcement learning", J. Guidance, Control, and Dynamics 42(8), 2019. arXiv:1810.04244
  J. Guman, J. C. O'Brien, C. Pondoc, M. J. Kochenderfer, "PyroRL: A reinforcement learning environment
  for wildfire evacuation", JOSS 9(100), 6739, 2024.  github.com/sisl/PyroRL (MIT)

Model (as in the papers): every burning cell s' within 2 cells ignites an unburnt cell s with probability
P(s, s') ~ 0.094 / d(s, s')^2; p(s) = 1 - prod_s' (1 - P(s, s') B(s')). A burning cell loses one unit of
fuel per step and goes out when its fuel is gone. Wind scales each neighbour's term linearly with
the dot product between the wind and the neighbour direction.

What is used unchanged from PyroRL (third_party/pyrorl_environment, commit b48cae9):
  - FireWorld.sample_fire_propogation()    the whole spread step
  - set_fire_mask / linear_wind_transform  the 5x5 kernel and the wind bias
Adapter choices (the only additions):
  - fuel per cell = the kit's burn time for that terrain (+-15 %); water, road and rock get no fuel
  - a cell without fuel cannot burn (PyroRL lets a fuel-less cell flicker on for one step)
  - PyroRL supports one uniform wind vector: the kit's mean wind is passed in every step
  - PyroRL leaves wind-speed units open; 1 kit wind unit = `wind_scale` PyroRL units (default 100,
    i.e. at the kit's typical wind of 0.5 a neighbour's term changes by up to +-20 %)
  - time scale: one PyroRL step every 1/SPEED kit steps (the only calibrated number), fuel is
    converted to PyroRL steps accordingly
  - PyroRL draws random numbers from torch's global generator; the adapter gives the fire its own
    saved generator state so the fire is reproducible and independent of PPO's sampling
"""
import numpy as np
import torch

from pyrorl_environment.environment import FireWorld, FIRE_INDEX, FUEL_INDEX
from pyrorl_environment import environment_constant as pyrorl_const

from .base import FireEngine

SPEED = 0.3          # PyroRL steps per kit step (calibrated: see README, "Calibration")


class PyroRLFire(FireEngine):
    name = "pyrorl"
    citation = "Julian & Kochenderfer, JGCD 2019; PyroRL (Guman et al., JOSS 2024)"

    def __init__(self, propagation_rate=0.094, wind_scale=100.0, fuel_jitter=0.15, speed=None):
        self.rate, self.wind_scale, self.fuel_jitter = propagation_rate, wind_scale, fuel_jitter
        self.speed = SPEED if speed is None else speed

    def reset(self, world, cells):
        from world import BURN_TIME
        n = world.size
        seed = self.seed_from(world)
        rng = np.random.default_rng(seed)
        homes = np.argwhere(world.homes)
        road = np.argwhere(world.road)
        np_state = np.random.get_state()             # FireWorld draws its initial fuel from np.random:
        try:                                         # keep the global stream untouched (we overwrite it)
            self.fw = FireWorld(
                num_rows=n, num_cols=n,
                populated_areas=homes if len(homes) else np.array([[0, 0]]),
                paths=[road.tolist()] if len(road) else [[[0, 0]]],
                paths_to_pops={},
                custom_fire_locations=np.array(cells, dtype=int).reshape(-1, 2),
                fire_propagation_rate=self.rate,
            )
        finally:
            np.random.set_state(np_state)
        fuel = BURN_TIME[world.terrain].astype(np.float64) * self.speed   # burn time in engine steps
        fuel *= rng.uniform(1 - self.fuel_jitter, 1 + self.fuel_jitter, size=fuel.shape)
        fuel[~world.burnable] = 0.0
        self.fw.state_space[FUEL_INDEX] = np.maximum(np.round(fuel), 1.0) * world.burnable
        self._had_fuel = self.fw.state_space[FUEL_INDEX] > 0
        self._torch_state = torch.Generator().manual_seed(seed).get_state()
        self._refresh()

    def _refresh(self):
        fire = self.fw.state_space[FIRE_INDEX]
        fuel = self.fw.state_space[FUEL_INDEX]
        fire[fuel <= 0] = 0                                   # no fuel -> cannot burn
        self.burning = fire > 0
        self.burnt = (fuel <= 0) & self._had_fuel

    def step(self, world):
        speed, angle = self.mean_wind(world)
        pyrorl_const.set_fire_mask(self.rate)                 # PyroRL keeps the base kernel in a global
        self.fw.fire_mask = pyrorl_const.linear_wind_transform(speed * self.wind_scale, angle)
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self._torch_state)
            self.fw.sample_fire_propogation()                 # <- PyroRL's spread step, unchanged
            self._torch_state = torch.get_rng_state()
        self._refresh()

    def ignite(self, world, y, x):
        self.fw.state_space[FIRE_INDEX, y, x] = 1
        self._refresh()
