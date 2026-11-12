import numpy as np
import torch
from stable_baselines3 import PPO
from env import SingleDroneEnv

MODEL = "runs/test_B_full/model.zip"
SEEDS = range(1000, 1010)          # 10 epizod
NAMES = ["N", "S", "E", "W", "UP", "DOWN", "HOVER"]

model = PPO.load(MODEL, device="cpu")


def run(deterministic):
    acts, alts = np.zeros(7), np.zeros(3)
    ent_sum, steps, edge, stag = 0.0, 0, 0, 0
    uniq, t_exist, t_detect = [], [], []
    lost, ep_len = [], []
    xs, ys = [], []
    for s in SEEDS:
        env = SingleDroneEnv(size=128, max_steps=300, map_types="mountain_valley",
                             difficulty="medium", fire_models="rothermel")
        obs, _ = env.reset(seed=s)
        core, n = env.core, env.core.size
        visited, te, td, done = set(), None, None, False
        while not done:
            obs_t, _ = model.policy.obs_to_tensor(obs)
            with torch.no_grad():
                ent_sum += float(model.policy.get_distribution(obs_t).entropy().mean())
            a, _ = model.predict(obs, deterministic=deterministic)
            a = int(a)
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            x, y = core.pos[0]
            acts[a] += 1
            alts[core.alt[0]] += 1
            steps += 1
            stag += info["stagnation"]
            edge += int(x <= 2 or y <= 2 or x >= n - 3 or y >= n - 3)
            visited.add((int(x), int(y)))
            xs.append(x)
            ys.append(y)
            if te is None and core.world.fire_map().sum() > 0:
                te = core.t
            if td is None and core.detected_ever.any():
                td = core.t
        uniq.append(len(visited))
        t_exist.append(te)
        t_detect.append(td)
        lost.append((core.lost_reason[0] or "ni izgubljen", core.t))
        ep_len.append(core.t)
    print(f"\n=== deterministic={deterministic} ({steps} korakov) ===")
    print("akcije (%):", {k: round(100 * float(v) / steps, 1) for k, v in zip(NAMES, acts)})
    print("višina 0/1/2 (%):", (100 * alts / steps).round(1).tolist())
    print("ob robu (%):", round(100 * edge / steps, 1))
    print("povprečni x, y:", round(float(np.mean(xs)), 1), round(float(np.mean(ys)), 1))
    print("različnih celic na epizodo:", uniq)
    print("zagorelo ob t:", t_exist)
    print("prvi zaznan požar ob t:", t_detect)
    print("povp. entropija:", round(ent_sum / steps, 3), "(maks = 1.946)")
    print("kazni za zastoj na epizodo:", round(stag / len(SEEDS), 1))
    print("dolžine epizod:", ep_len)
    print("izguba (razlog, korak):", lost)


run(True)
run(False)