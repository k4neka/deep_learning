import io
import torch
from stable_baselines3 import PPO
import train_ppo  # noqa: F401  (SmallCNN must be importable, as in diag_ppo.py)

# 1) torch can't read straight from a zip stream (which stable-baselines3 passes it): read it into memory first.
_orig_load = torch.load


def _load(f, *a, **k):
    if hasattr(f, "read") and not isinstance(f, io.BytesIO):
        f = io.BytesIO(f.read())
    return _orig_load(f, *a, **k)


torch.load = _load

# 2) the model stores SmallCNN as a by-value copy from the cluster's Python 3.13: use the local class instead.
_orig_ppo_load = PPO.load.__func__


def _ppo_load(cls, path, *a, **k):
    k.setdefault("custom_objects", dict(
        policy_kwargs=dict(features_extractor_class=train_ppo.SmallCNN,
                           features_extractor_kwargs=dict(features_dim=128),
                           normalize_images=False)))
    return _orig_ppo_load(cls, path, *a, **k)


PPO.load = classmethod(_ppo_load)

# usage: same arguments as render.py, e.g.
#   python render_b4.py --policy ppo --model runs\B4_re20\model.zip --drones 1 --map rolling_hills --seed 5019 ...
import render  # noqa: E402

render.main()
