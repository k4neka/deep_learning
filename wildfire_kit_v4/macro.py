"""
Macro-actions for one drone: instead of N/S/E/W/up/down/hover, the policy picks WHERE to look next
(a sector of a G x G grid over the map) and AT WHICH ALTITUDE (low / medium / high). A simple
controller then flies there with the normal low-level actions:

  1. climb or descend until the chosen altitude is reached,
  2. fly towards the sector centre with SwarmCore.step_towards (goes around no-fly zones),
  3. stop when the drone is within ARRIVE cells of the centre, after `horizon` low-level steps,
     or when the episode ends.

Every low-level step is a normal SwarmCore.step(), so the world, sensing, battery safety layer and
reward are exactly the same as in SingleDroneEnv. The macro reward is the sum of the low-level rewards.

Why: with cell-by-cell moves on a 128 x 128 map the policy must chain dozens of correct moves before it
sees any fire. Choosing a sector is the decision the greedy baseline actually makes, and it shortens
the decision horizon by up to `horizon` times.

Action index a in [0, G*G*3):  altitude = a % 3,  sector = a // 3,  gx = sector % G,  gy = sector // G
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env import SwarmCore, N_CHANNELS, OBS, UP, DOWN, HOVER

ARRIVE = 2          # cells: "close enough" to the sector centre


def decode(a, grid, size):
    """Action index -> (target (x, y), target altitude)."""
    a = int(a)
    alt = a % 3
    sector = a // 3
    gx, gy = sector % grid, sector // grid
    cell = size / grid
    x = int(min(size - 1, (gx + 0.5) * cell))
    y = int(min(size - 1, (gy + 0.5) * cell))
    return (x, y), alt


def low_level_action(core, i, target, alt):
    """One primitive action of the controller, or None when the macro-action is finished."""
    if core.alt[i] < alt:
        return UP
    if core.alt[i] > alt:
        return DOWN
    x, y = core.pos[i]
    if abs(int(x) - target[0]) <= ARRIVE and abs(int(y) - target[1]) <= ARRIVE:
        return None
    a = core.step_towards(i, target)
    return None if a == HOVER else a


class MacroDroneEnv(gym.Env):
    """Gymnasium env for one drone with macro-actions. Same 22 x 32 x 32 observation as SingleDroneEnv.

    info["low_steps"] = number of low-level steps the macro-action took (used to count the training
    budget in low-level steps, so that macro and primitive runs are compared on the same number of
    simulator steps).
    """

    metadata = {"render_modes": []}

    def __init__(self, size=128, max_steps=300, seed=None, start_alt=1, map_types="random",
                 difficulty="medium", safety_return=True, fire_models="native", reward_w=None,
                 grid=8, horizon=8):
        self.core = SwarmCore(size=size, n_drones=1, max_steps=max_steps, seed=seed, start_alt=start_alt,
                              map_types=map_types, difficulty=difficulty, safety_return=safety_return,
                              fire_models=fire_models, reward_w=reward_w)
        self.grid, self.horizon = grid, horizon
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(N_CHANNELS, OBS, OBS), dtype=np.float32)
        self.action_space = spaces.Discrete(grid * grid * 3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.core.reset(seed=seed)
        return self.core.observe(0), {}

    def step(self, action):
        core = self.core
        target, alt = decode(action, self.grid, core.size)
        total, n, done, info = 0.0, 0, False, {}
        while n < self.horizon and not done:
            a = low_level_action(core, 0, target, alt)
            if a is None:
                if n > 0:
                    break
                a = HOVER                 # already there: hover one step so that time always advances
            r, done, info = core.step([a])
            total += r
            n += 1
        info = dict(info, low_steps=n)
        truncated = core.t >= core.max_steps
        return core.observe(0), total, done and not truncated, truncated, info


class MacroPolicyWrapper:
    """Lets a macro-action model fly any number of drones inside baselines.evaluate().

    Each drone keeps its current macro-action until it arrives, the horizon is used up, or the
    battery safety layer has taken over; only then is the model asked again. This mirrors
    MacroDroneEnv.step for one drone.
    """

    def __init__(self, model, deterministic=True, grid=8, horizon=8):
        self.model, self.deterministic = model, deterministic
        self.grid, self.horizon = grid, horizon

    def reset(self, core):
        self.plan = {}            # drone -> [target, alt, steps used]

    def act(self, core, i):
        p = self.plan.get(i)
        if p is not None and p[2] < self.horizon and not core.returning[i]:
            a = low_level_action(core, i, p[0], p[1])
            if a is not None:
                p[2] += 1
                return a
        a_macro, _ = self.model.predict(core.observe(i), deterministic=self.deterministic)
        target, alt = decode(a_macro, self.grid, core.size)
        self.plan[i] = [target, alt, 1]
        a = low_level_action(core, i, target, alt)
        return HOVER if a is None else a
