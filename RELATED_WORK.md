# Related work and simulators

Checked October 2026. "Integrated" means the simulator's own code runs inside this kit (see
`fire_models/` and `third_party/`).

## 1. Drones / aircraft monitoring wildfires with learning or planning (closest to our project)

| Paper | What they do | Fire model | Relation to us |
| --- | --- | --- | --- |
| K. D. Julian, M. J. Kochenderfer, *Distributed wildfire surveillance with autonomous aircraft using deep reinforcement learning*, J. Guidance, Control, and Dynamics 42(8):1768–1778, 2019. doi:10.2514/1.G004106, arXiv:1810.04244 | 2–4 fixed-wing aircraft, DQN; a "belief" version keeps a shared map of burning belief + time since last visit; scales to more aircraft than trained for | stochastic cellular model, P ∝ 1/d², fuel countdown, wind bias | Same task and same idea of a shared, ageing map (our `belief` / `staleness` channels). **Fire model integrated** (`pyrorl`) |
| R. N. Haksar, M. Schwager, *Distributed deep reinforcement learning for fighting forest fires with a network of aerial robots*, IEEE/RSJ IROS 2018, pp. 1067–1074. doi:10.1109/IROS.2018.8593539 | Multi-agent DQN, aerial robots that suppress fire; decentralised policy trained with few agents, run with many | lattice Markov model (`rhaksar/simulators`) | Fire suppression instead of monitoring, but same simulator family. **Integrated** (`haksar`, `haksar_ref`) |
| A. Viseras, M. Meissner, J. Marchal, *Wildfire front monitoring with multiple UAVs using deep Q-learning*, IEEE Access, 2021. doi:10.1109/ACCESS.2021.3055651 | Multiple UAVs, deep Q-learning to keep the fire front observed | own simulator | Same task; useful baseline discussion |
| C. Diaz-Vilor, A. Lozano, H. Jafarkhani, *A reinforcement learning approach for wildfire tracking with UAV swarms*, arXiv:2407.05473, 2024 | 1–4 UAVs, TD3; altitude vs. coverage trade-off; separate charging policy with finite battery | FARSITE (elliptical spread) | Same altitude trade-off and battery constraint as our hard/extreme levels |
| S. Khoshdel, Q. Luo, F. Afghah, *PyroTrack: belief-based deep reinforcement learning path planning for aerial wildfire monitoring in partially observable environments*, arXiv:2403.11095, 2024 | One low-altitude drone, POMDP, Bayesian belief of ignition per cell | not named | Belief-map input like ours; single drone |
| E. Seraj, M. Gombolay, *Coordinated control of UAVs for human-centered active sensing of wildfires*, ACC 2020, arXiv:2006.07969 | Distributed control, Kalman-filter uncertainty + firefighter safety | – | Safety of people near the fire ≈ our criticality weighting |
| E. Seraj et al., *FireCommander: an interactive, probabilistic multi-agent environment for heterogeneous robot teams*, arXiv:2011.00165, 2020 | Multi-agent perception–action environment for wildfire | FARSITE-style | Public repository no longer accessible (needs login, checked 10/2026) – not integrated |
| R. Bailon-Ruiz, A. Bit-Monnot, S. Lacroix, *Real-time wildfire monitoring with a fleet of UAVs*, Robotics and Autonomous Systems 152:104071, 2022. doi:10.1016/j.robot.2022.104071 | Fixed-wing fleet, Variable Neighbourhood Search planning with terrain, wind and fire-propagation models; mixed-reality field test | propagation model with wind and terrain | Classical planning baseline for the same problem |

## 2. Simulators and RL environments for wildfire

| Simulator | Paper | Licence | Status here |
| --- | --- | --- | --- |
| **SimFire** + SimHarness (MITRE) | A. Tapley et al., *Reinforcement learning for wildfire mitigation in simulated disaster environments*, NeurIPS 2023 Workshop on Tackling Climate Change with ML, arXiv:2311.15925 | Apache-2.0 | **Integrated** (`rothermel`): SimFire's `compute_rate_of_spread`, fuel models and fuel constants used unchanged; its fire manager re-written with arrays and verified cell-for-cell identical to the original on 8/8 test missions |
| **PyroRL** (Stanford SISL) | J. Guman, J. C. O'Brien, C. Pondoc, M. J. Kochenderfer, *PyroRL: a reinforcement learning environment for wildfire evacuation*, JOSS 9(100):6739, 2024. doi:10.21105/joss.06739 | MIT | **Integrated** (`pyrorl`): `FireWorld.sample_fire_propogation` and the wind kernel called unchanged. Its fire model is the Julian & Kochenderfer one |
| **rhaksar/simulators** (Stanford MSL) | Haksar & Schwager 2018 (above) and the authors' follow-up papers | MIT | **Integrated** (`haksar_ref` = original `LatticeForest` object; `haksar` = array version, same statistics, KS test p ≈ 0.7–0.8) |
| Cell2Fire | C. Pais et al., *Cell2Fire: a cell-based forest fire growth model to support strategic landscape management planning*, Frontiers in Forests and Global Change 4:692706, 2021, doi:10.3389/ffgc.2021.692706; arXiv:1905.09317 | GPL-3.0 | Not integrated: C++ program driven by input files, too slow to call inside an RL loop. Good candidate for an offline check of the forecast CNN |
| free-range-zoo (wildfire domain) | C. Patino et al., *Inaugural MOASEI Competition at AAMAS'2025: a technical report*, arXiv:2507.05469 | AGPL-3.0 | Not integrated: different task (fire *suppression* by agents that join and leave the team, i.e. open agent systems) |
| FARSITE | M. A. Finney, *FARSITE: Fire Area Simulator – model development and evaluation*, USDA Forest Service RMRS-RP-4, 1998 | – | Reference model used by Diaz-Vilor et al. and FireCommander; no maintained Python package |

## 3. Fire-spread physics behind the engines

- R. C. Rothermel, *A mathematical model for predicting fire spread in wildland fuels*, USDA Forest Service INT-115, 1972. – the `rothermel` engine.
- P. L. Andrews, *The Rothermel surface fire spread model and associated developments: a comprehensive explanation*, USDA Forest Service RMRS-GTR-371, 2018. – fuel parameters used by SimFire.
- A. Alexandridis, D. Vakalis, C. I. Siettos, G. V. Bafas, *A cellular automata model for forest fire spread prediction: the case of the wildfire that swept through Spetses Island in 1990*, Applied Mathematics and Computation 204:191–201, 2008. – the kit's **native** model has the same structure: ignition probability multiplied by an exponential wind factor (in the cosine between wind and spread direction) and an exponential slope factor; their slope coefficient a = 0.078 per degree gives ×2.2 per 10° of upslope, our `slope_k` gives ×2.0.

## 4. Deep learning for fire-spread prediction (for `forecast_test.py`)

- F. Huot et al., *Next Day Wildfire Spread: a machine learning dataset to predict wildfire spreading from remote-sensing data*, IEEE Trans. Geoscience and Remote Sensing, 2022. arXiv:2112.02447
- S. Gerard et al., *WildfireSpreadTS: a dataset of multi-modal time series for wildfire spread prediction*, NeurIPS 2023 Datasets and Benchmarks.

## Suggested wording for the report

> Fire spread is simulated with four interchangeable engines that share the same maps, wind field and
> ignition points: our own stochastic cellular model (same structure as Alexandridis et al. 2008), the
> stochastic model of Julian & Kochenderfer (2019) as implemented in PyroRL, the lattice model of
> Haksar & Schwager (2018) from the authors' `simulators` package, and the Rothermel (1972) model as
> implemented in MITRE's SimFire. The external engines run the original code; where we re-implemented
> a component for speed, we verified it against the original (identical fire maps for SimFire;
> statistically indistinguishable burnt areas for the lattice model). Each engine has a single
> time-scale parameter, calibrated so that a medium fire grows at a comparable rate. Policies are
> trained on [one / all] engine(s) and evaluated on all four.

Be honest in the report about what this does and does not show: running published simulators shows
that results do not depend on one home-made fire model and makes them comparable to earlier work.
None of these engines (including SimFire) has been validated here against real fires.
