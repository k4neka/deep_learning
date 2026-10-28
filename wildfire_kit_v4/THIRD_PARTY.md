# Third-party code in `third_party/`

All files below are copied **byte-for-byte unmodified** from the upstream repositories (the commit is
given so anyone can diff them). Only the folder names were chosen by us so the pieces can be imported
without installing the full packages (SimFire needs Python 3.9 + pygame; PyroRL's package import
pulls in pygame and registers Gymnasium environments).

| Folder | Upstream | Commit | Files | Licence |
| --- | --- | --- | --- | --- |
| `pyrorl_environment/` | github.com/sisl/PyroRL, `pyrorl/pyrorl/envs/environment/` | b48cae9 | `environment.py`, `environment_constant.py` | MIT (`LICENSE`) |
| `haksar_simulators/simulators/` | github.com/rhaksar/simulators | ed874b5 | `Element.py`, `Simulator.py`, `fires/LatticeForest.py`, `fires/ForestElements.py` (+ empty `__init__.py`) | MIT (`LICENSE`) |
| `simfire_world/` | github.com/mitrefireline/simfire, `simfire/world/` (also PyPI `simfire` 2.0.1) | 04905b1 | `rothermel.py`, `parameters.py`, `presets.py` | Apache-2.0 (`LICENSE`) |

MITRE asks to be notified at opensource@mitre.org if SimFire code is used in a deployment or embedded
in another project (SimHarness README); not needed for coursework, but worth knowing.
