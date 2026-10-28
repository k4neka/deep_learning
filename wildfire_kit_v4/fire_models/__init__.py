"""
Pluggable fire-spread engines for the wildfire swarm kit (v4).

The kit's maps, wind, smoke, drones, sensing and reward stay exactly the same; only the rule
that decides which cells catch fire and when they burn out is swapped. Three of the engines
come from published, open-source wildfire simulators (vendored unmodified in ../third_party):

  name        source simulator (paper)                                         physics
  ---------   --------------------------------------------------------------   -----------------------------
  native      this kit (v3)                                                     8-neighbour stochastic CA,
                                                                                exp(wind) x exp(slope)
  pyrorl      PyroRL, Stanford SISL (Guman et al., JOSS 2024) - implements the  5x5 stochastic kernel ~1/d^2,
              fire model of Julian & Kochenderfer (JGCD 2019), the reference    fuel countdown, linear wind
              paper on wildfire surveillance with DRL-controlled aircraft
  haksar      rhaksar/simulators LatticeForest (Haksar & Schwager, IROS 2018,   4-neighbour Markov lattice,
              DRL firefighting with a network of aerial robots)                 1-(1-a)^k ignition, persistence b
  rothermel   MITRE SimFire RothermelFireManager (Tapley et al., NeurIPS 2023   Rothermel (1972) rate of spread,
              CCAI workshop; SimHarness), Rothermel 1972 / Andrews 2018         Anderson fuel models, wind+slope

`haksar_ref` runs the original, unvectorised LatticeForest object itself (slow, used to check that
the fast `haksar` engine is statistically identical; see tests/test_fire_models.py).
`rothermel` is a vectorised re-implementation of SimFire's RothermelFireManager.update() that calls
SimFire's own compute_rate_of_spread(); tests/validate_simfire_reference.py shows it produces
cell-for-cell identical burn maps to the original SimFire code.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_TP = os.path.join(os.path.dirname(_HERE), "third_party")
for _p in (_TP, os.path.join(_TP, "haksar_simulators")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FIRE_MODELS = ["native", "pyrorl", "haksar", "rothermel"]          # used in experiments
ALL_FIRE_MODELS = FIRE_MODELS + ["haksar_ref"]                      # + slow reference implementation


def make_fire_model(name, **kwargs):
    """Return a FireEngine for `name`, or None for the kit's native model."""
    if name in (None, "native"):
        return None
    if name == "pyrorl":
        from .pyrorl_julian import PyroRLFire
        return PyroRLFire(**kwargs)
    if name == "haksar":
        from .haksar_lattice import HaksarLatticeFast
        return HaksarLatticeFast(**kwargs)
    if name == "haksar_ref":
        from .haksar_lattice import HaksarLatticeRef
        return HaksarLatticeRef(**kwargs)
    if name == "rothermel":
        from .simfire_rothermel import SimFireRothermel
        return SimFireRothermel(**kwargs)
    raise ValueError(f"unknown fire model {name!r}; choose from {ALL_FIRE_MODELS}")
