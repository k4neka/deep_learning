"""Common interface between world.World and an external fire-spread engine."""
import numpy as np


class FireEngine:
    """
    An engine owns the fire state (burning / burnt cells). The World asks it to advance one step and
    then copies `burning` and `burnt` into its own arrays, so the drones, smoke, rendering and reward
    code do not need to know which engine is running.

    Subclasses implement:
        reset(world, cells)   start fires at the (row, col) cells chosen by the World
        step(world)           advance one time step (world.wx / world.wy hold the current wind)
        ignite(world, y, x)   extra ignition (spot fire / lightning), cell is known to be unburnt
    and keep two boolean (size x size) arrays up to date: self.burning, self.burnt
    """
    name = "engine"
    citation = ""
    speed = 1.0          # engine steps per kit step; the only time-scale knob (see README, "Calibration")

    def reset(self, world, cells):
        raise NotImplementedError

    def step(self, world):
        raise NotImplementedError

    def ignite(self, world, y, x):
        raise NotImplementedError

    @staticmethod
    def seed_from(world):
        """A seed drawn from the World's own random stream (so the whole episode stays reproducible)."""
        return int(world.rng.integers(2 ** 31 - 1))

    @staticmethod
    def mean_wind(world):
        """Uniform wind for engines that only support one wind vector: (speed in kit units, math angle).
        Kit wind (wx, wy) is in image coordinates (x right, y down) and points where the wind blows TO."""
        wx, wy = float(np.mean(world.wx)), float(np.mean(world.wy))
        return float(np.hypot(wx, wy)), float(np.arctan2(-wy, wx))
