"""Patch env.py and train_ppo.py for B4 (--re-weight). Makes .bak_B3 backups. Run from wildfire_kit_v4."""
import shutil, sys

def patch(path, edits):
    raw = open(path, newline="").read()
    crlf = "\r\n" in raw
    s = raw.replace("\r\n", "\n")
    for old, new in edits:
        n = s.count(old)
        if n != 1:
            sys.exit(f"NAPAKA v {path}: iskani niz se pojavi {n}x (pricakovano 1x):\n{old}")
        s = s.replace(old, new)
    shutil.copyfile(path, path + ".bak_B3")
    open(path, "w", newline="").write(s.replace("\n", "\r\n") if crlf else s)
    print("OK:", path, f"({len(edits)} zamenjav, kopija {path}.bak_B3)")

patch("env.py", [
    ('                 fire_models="native"):\n        assert size % OBS == 0',
     '                 fire_models="native", reward_w=None):\n        assert size % OBS == 0'),
    ('        self.record, self.safety_return = record, safety_return\n',
     '        self.record, self.safety_return = record, safety_return\n'
     '        self.reward_w = {**REWARD_W, **(reward_w or {})}      # per-instance weights; default = v6\n'),
    ('        rw = REWARD_W\n', '        rw = self.reward_w\n'),
    ('difficulty="medium", safety_return=True, fire_models="native"):\n        self.core = SwarmCore(',
     'difficulty="medium", safety_return=True, fire_models="native", reward_w=None):\n        self.core = SwarmCore('),
    ('fire_models=fire_models)\n        self.observation_space',
     'fire_models=fire_models, reward_w=reward_w)\n        self.observation_space'),
])
patch("train_ppo.py", [
    ('    ap.add_argument("--seed", type=int, default=0)\n',
     '    ap.add_argument("--re-weight", type=float, default=2.0,\n'
     '                    help="reward weight of R_E (exploration) in TRAINING only; 2.0 = v6 default (B1-B3)")\n'
     '    ap.add_argument("--seed", type=int, default=0)\n'),
    ('                                       fire_models=args.fire_models))\n    if args.vecnorm:',
     '                                       fire_models=args.fire_models,\n'
     '                                       reward_w=dict(E=args.re_weight)))\n    if args.vecnorm:'),
    ('| ent_coef: {args.ent_coef}"', '| ent_coef: {args.ent_coef} | re_weight: {args.re_weight}"'),
])
