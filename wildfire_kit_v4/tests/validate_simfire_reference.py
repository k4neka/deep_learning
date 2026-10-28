"""
Check that fire_models.simfire_rothermel.RothermelGrid reproduces MITRE SimFire exactly.

Runs SimFire's own RothermelFireManager (pip package `simfire`, needs Python 3.9) and the kit's
array version side by side on kit maps (several map types, difficulties, multiple ignitions, a wind
change half-way) and compares the full fire map (unburned / burning / burned) after every step.

Setup (once, separate environment because simfire pins Python 3.9):
    uv venv -p 3.9 venv39 && VIRTUAL_ENV=venv39 uv pip install simfire "setuptools<70"
    SDL_VIDEODRIVER=dummy venv39/bin/python tests/validate_simfire_reference.py
"""
import os
import sys
import types

import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simfire.enums import BurnStatus                                   # noqa: E402
from simfire.game.managers.fire import RothermelFireManager             # noqa: E402
from simfire.game.sprites import Fire                                   # noqa: E402
from simfire.world.parameters import Environment, Fuel, FuelParticle    # noqa: E402

from world import World                                                 # noqa: E402
from fire_models.simfire_rothermel import RothermelGrid, kit_fuels, SimFireRothermel, FT_PER_M  # noqa: E402


def run_case(seed, map_type, difficulty, size=64, steps=150, update_rate=0.08, max_dur=12):
    w = World(size=size, rng=np.random.default_rng(seed), map_type=map_type, difficulty=difficulty)
    w.reset()                                       # native world: we only take map, elevation, wind, ignitions
    ign = [tuple(int(v) for v in yx) for yx in np.argwhere(w.burning)]
    fuel = kit_fuels(w.terrain)
    elev = w.elev.astype(np.float64) * FT_PER_M
    adapter = SimFireRothermel(update_rate=update_rate)
    U, U_dir = adapter._wind(w)
    M_f, ps = 0.05, w.cell_m * FT_PER_M

    # --- reference: SimFire's own manager
    fuels = np.empty(fuel["w_0"].shape, dtype=object)
    for (y, x), _ in np.ndenumerate(fuels):
        fuels[y, x] = Fuel(w_0=fuel["w_0"][y, x], delta=fuel["delta"][y, x],
                           M_x=fuel["M_x"][y, x], sigma=fuel["sigma"][y, x])
    terrain = types.SimpleNamespace(fuels=fuels, elevations=elev, screen_size=(size, size), image=None)
    y0, x0 = ign[0]
    ref = RothermelFireManager(init_pos=(x0, y0), fire_size=1, max_fire_duration=max_dur, pixel_scale=ps,
                               update_rate=update_rate, fuel_particle=FuelParticle(), terrain=terrain,
                               environment=Environment(M_f=M_f, U=U, U_dir=U_dir), headless=True,
                               diagonal_spread=True)
    fire_map = np.zeros((size, size), dtype=np.int64)
    fire_map[y0, x0] = BurnStatus.BURNING
    for y, x in ign[1:]:
        ref.sprites.append(Fire((x, y), 1, headless=True))
        ref.durations.append(0)
        fire_map[y, x] = BurnStatus.BURNING

    # --- kit array version
    grid = RothermelGrid(fuel=fuel, elevation=elev, pixel_scale=ps, update_rate=update_rate,
                         max_fire_duration=max_dur, M_f=M_f, U=U, U_dir=U_dir)
    for y, x in ign:
        grid.add_fire(x, y)

    for t in range(steps):
        if t == steps // 2:                         # weather change: new wind for both
            for _ in range(20):
                w.wind_step(1000)
            U, U_dir = adapter._wind(w)
            ref.U, ref.U_dir = U, U_dir
            grid.U, grid.U_dir = U, U_dir
        fire_map, _ = ref.update(fire_map)
        grid.update()
        a = np.asarray(fire_map, dtype=np.int64)
        if not np.array_equal(a, grid.fire_map.astype(np.int64)):
            return t, int((a != grid.fire_map).sum()), int((a > 0).sum())
    return None, 0, int((np.asarray(fire_map) > 0).sum())


if __name__ == "__main__":
    cases = [(s, m, d) for s, (m, d) in enumerate([
        ("rolling_hills", "medium"), ("mountain_valley", "hard"), ("lake_shore", "easy"),
        ("dense_forest", "extreme"), ("urban_interface", "medium"), ("farmland", "hard"),
        ("mountain_valley", "extreme"), ("rolling_hills", "easy")])]
    ok = 0
    for seed, m, d in cases:
        t, n_diff, touched = run_case(seed, m, d)
        status = "IDENTICAL" if t is None else f"DIFFERS at step {t} ({n_diff} cells)"
        ok += t is None
        print(f"seed {seed}  {m:16s} {d:8s}  cells reached by fire: {touched:5d}   {status}", flush=True)
    print(f"\n{ok}/{len(cases)} cases identical to SimFire's RothermelFireManager over 150 steps")
