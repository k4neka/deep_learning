import numpy as np
from env import SwarmCore  # noqa: F401  (preveri, da se env.py naloži)
from baselines import evaluate, Lawnmower, Greedy, RandomPolicy

MAPS = "rolling_hills,mountain_valley,dense_forest,farmland"
SEEDS = range(5000, 5020)
KW = dict(seeds=SEEDS, map_types=MAPS, difficulty="medium", fire_models="rothermel")

rows = [
    ("random", evaluate(RandomPolicy(), n_drones=1, **KW)),
    ("lawnmower", evaluate(Lawnmower(), n_drones=1, **KW)),
    ("greedy", evaluate(Greedy(), n_drones=1, **KW)),
    ("greedy_3drones", evaluate(Greedy(), n_drones=3, **KW)),
]
print(f"{'politika':16s} {'return':>9s} {'staleness':>10s} {'heat/ep':>8s}")
for name, m in rows:
    print(f"{name:16s} {m['return']:9.2f} {m['staleness']:10.3f} {m['lost_heat_per_ep']:8.2f}")

r = {n: m["return"] for n, m in rows}
ok = r["greedy_3drones"] > r["greedy"] > r["lawnmower"] > r["random"]
print("\nvrstni red Greedy-3 > Greedy-1 > Lawnmower > Random:", "DA" if ok else "NE")