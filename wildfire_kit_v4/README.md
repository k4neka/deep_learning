# Wildfire swarm — v4 (fire spread from published simulators)

## What's new in v4

v3's maps, wind, smoke, drones, sensing, reward and baselines are unchanged. The **fire-spread rule
is now pluggable**, and three of the four engines come from open-source wildfire simulators that
were used in published papers. Their code is included and runs unmodified (see `THIRD_PARTY.md`).

| `--fire-model` | Source | Used in | Physics |
| --- | --- | --- | --- |
| `native` | this kit (identical to v3, checked on 40 missions) | – (same structure as Alexandridis et al. 2008) | 8-neighbour stochastic CA, exp(wind) × exp(slope) |
| `pyrorl` | PyroRL, Stanford SISL (JOSS 2024) | Julian & Kochenderfer, *JGCD* 2019: drones monitoring a wildfire with deep RL | 5×5 stochastic kernel ∝ 1/d², fuel countdown, wind |
| `haksar` | `rhaksar/simulators` LatticeForest | Haksar & Schwager, *IROS* 2018: deep RL for aerial firefighting robots | 4-neighbour Markov lattice, no wind or slope |
| `rothermel` | MITRE SimFire (Rothermel 1972, Anderson fuel models) | Tapley et al., *NeurIPS 2023 CCAI workshop* (SimFire + SimHarness) | physical rate of spread with fuel type, moisture, wind and slope |

The same seed gives the **same map, wind and ignition points** for every engine, so engines can be
compared mission by mission. Spot fires, new ignitions and smoke are kit features and work with
every engine. `RELATED_WORK.md` lists the papers (what they do, which fire model they use) and
gives a paragraph you can adapt for the report.

## How each simulator was brought in (and verified)

| Engine | What runs unchanged | What we added | Verification |
| --- | --- | --- | --- |
| `pyrorl` | `FireWorld.sample_fire_propogation()`, the 5×5 kernel and the wind transform | fuel = kit burn time per terrain; cells without fuel can't burn; kit's mean wind passed each step; its own random stream | `tests/test_fire_models.py`: wind direction, reproducibility |
| `haksar_ref` / `haksar` | `haksar_ref` builds the original `LatticeForest` and calls `update()`; `haksar` is the same Markov chain on arrays (faster) | α per cell from terrain flammability (forest = package default 0.2763), β = package default e^(−1/10) | fast vs original: same burnt-area distribution at steps 30/60/120 (KS test p = 0.72 / 0.72 / 0.82, 150 missions each) |
| `rothermel` | SimFire's `compute_rate_of_spread()` (the Rothermel equations), fuel presets, fuel constants | `RothermelGrid`: SimFire's `RothermelFireManager.update()` rewritten with arrays (no pygame, no Python 3.9 pin); terrain → fuel model mapping | `tests/validate_simfire_reference.py` runs SimFire's own manager next to ours: **identical fire map at every step, 8/8 missions** (6 map types, all difficulty levels, wind change half-way) |

**A bug we found in SimFire:** its slope term has the wrong sign, so a SimFire fire runs faster
*downhill* (with no wind, the rate of spread is 9.45 ft/min downhill and 0 uphill on a 14° slope). The adapter corrects this by
default without editing SimFire's code (it gives the function the slope direction rotated by 180°).
`SimFireRothermel(fix_slope_sign=False)` reproduces SimFire exactly. You can mention this in the
report; it shows the simulators were actually tested.

### Calibration (one number per engine)

Each engine has its own notion of a time step, so each one gets **a single time-scale parameter**.
Each one is set so that a medium fire grows at about the native speed (median over 12 missions, all
map types). The papers' own spread parameters are kept.

| Engine | Parameter | Value |
| --- | --- | --- |
| `pyrorl` | PyroRL steps per kit step | 0.3 |
| `haksar` | lattice steps per kit step | 0.6 |
| `rothermel` | SimFire `update_rate` (minutes per step) | 0.08 (= 4.8 s) |

Result (`python compare_simulators.py`, medium, area reached by fire in ha):

| Engine | t=50 | t=100 | t=200 | t=300 |
| --- | --- | --- | --- | --- |
| native | 1.5 | 5.5 | 13.2 | 17.6 |
| pyrorl | 2.2 | 6.4 | 12.3 | 13.6 |
| haksar | 1.4 | 4.9 | 16.1 | 22.8 |
| rothermel | 2.7 | 5.3 | 9.3 | 15.2 |

### First results across simulators

Baselines, 3 drones, 12 missions, all map types, medium (higher is better):

| Engine | Random | Lawnmower | Greedy |
| --- | --- | --- | --- |
| native | 3.8 | 20.5 | 52.2 |
| pyrorl | 2.5 | 17.9 | 47.6 |
| haksar | 5.9 | 21.6 | 40.6 |
| rothermel | 5.4 | 22.6 | 43.5 |

The ranking of the strategies is the same under all four fire models, but the gap between greedy and
lawnmower shrinks on the engines the greedy rule was not tuned on. That is the kind of effect the
PPO cross-simulator test is designed to measure.

A quick, small forecast-CNN run (40 training missions, 6 epochs) trained on native fires already
generalises poorly to `haksar` and `rothermel`: its IoU falls below the "fire stays where it is"
baseline there. Training on all engines is the obvious next experiment.

## Files

| File | What it does |
| --- | --- |
| `world.py` | Map generators, elevation, difficulty, wind, native fire spread, smoke; v4: `fire_model=` |
| `env.py` | Drones, sensing, battery, no-fly zones, team map, reward; v4: `fire_models=` (one or a list) |
| `fire_models/` | The engine adapters: `pyrorl_julian.py`, `haksar_lattice.py`, `simfire_rothermel.py` |
| `third_party/` | Unmodified code from PyroRL, rhaksar/simulators, SimFire (+ their licences) |
| `compare_simulators.py` | Side-by-side picture, growth curves, baseline table per engine |
| `tests/test_fire_models.py` | Wind/slope direction, Haksar fast = original, reproducibility, full episodes |
| `tests/validate_simfire_reference.py` | Our Rothermel engine vs SimFire's own code (needs a Python 3.9 env) |
| `baselines.py` | Random, lawnmower, greedy + `evaluate()`; `--fire-model` |
| `train_ppo.py` | PPO; `--fire-models` for training, final test on every engine (`cross_simulator`) |
| `forecast_test.py` | Fire forecast CNN; `--fire-model` for training data, `--test-fire-models` |
| `render.py`, `maps_gallery.py` | Video / pictures; `render.py --fire-model` |
| `jobs/train_gpu_allfire.slurm` | VSC job: train on all four engines, test on each |
| `RELATED_WORK.md`, `THIRD_PARTY.md` | Papers and simulators; what was copied from where |

## On your laptop

```bash
pip install torch gymnasium stable-baselines3 matplotlib pillow imageio-ffmpeg scipy
cd wildfire_kit_v4
python tests/test_fire_models.py                       # ~3 min, all checks should pass
python compare_simulators.py                           # ~4 min -> runs/compare/*.png
python baselines.py --drones 3 --episodes 6 --fire-model rothermel
python render.py --policy greedy --drones 3 --map mountain_valley --difficulty hard --fire-model rothermel --out mv_rothermel.mp4
python forecast_test.py --k 10 --fire-model native --test-fire-models native,pyrorl,haksar,rothermel
```

Optional, to re-check the SimFire equivalence yourself (SimFire needs Python 3.9):

```bash
uv venv -p 3.9 venv39 && VIRTUAL_ENV=venv39 uv pip install simfire "setuptools<70"
SDL_VIDEODRIVER=dummy venv39/bin/python tests/validate_simfire_reference.py
```

## On VSC

```bash
cd $VSC_DATA/dl-project
module purge
module load PyTorch/2.9.1-foss-2025a-CUDA-12.8.0-whl
source venv-fire/bin/activate
pip install scipy                               # once (only the tests use it)
unzip wildfire_kit_v4.zip && cd wildfire_kit_v4
python train_ppo.py --steps 20000 --n-envs 8 --subproc --eval-episodes 3 --fire-models native,rothermel --out runs/smoke
sbatch jobs/train_gpu.slurm            # trains on native fire, tests on all four engines
sbatch jobs/train_gpu_allfire.slurm    # trains on all four engines (one per episode), tests on each
```

`results.json` now also has `cross_simulator`: score of every policy on each engine. A suggested main
table for the report: rows = trained on {native, all engines}, columns = tested on each engine.

## Things to say honestly in the report

- None of the engines has been validated against real fires here. Using them shows that the results
  do not depend on one home-made fire model and makes the work comparable to earlier papers.
- The time scale of each engine is calibrated (one number each, see above); everything else uses the
  published defaults, except for the terrain → fuel mapping and the adapter choices listed in each
  file's docstring.
- PyroRL and the Haksar lattice only support one uniform wind (or none); spatial wind variation
  only matters for `native` and `rothermel`.
- In `rothermel` houses do not burn (SimFire treats urban cells as non-burnable), but fire next to
  houses still counts as critical.
