#!/usr/bin/env python3
# sweep.py
"""
Launch a grid of runs for any config subclass defined in configs.py.

Examples
--------
# TinyStories Pre-Norm, 2×2 grid
python sweep.py --cfg_cls TSPreNormRMS --lr 3e-4 1e-4 --bs 16 32

# Rope-less ablation, single run
python sweep.py --cfg_cls configs:TSRemoveRope --lr 5e-5 --bs 32
"""

import argparse, importlib, itertools, json, subprocess, uuid
from pathlib import Path

# ────────────────────────────────────────────────────────────────
# 1. Parse sweep-level CLI flags
# ----------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--cfg_cls",  default="TSPreNormRMS",
               help="Either <ClassName> (assumed in configs.py) or <module:ClassName>")
p.add_argument("--lr",  type=float, nargs="+", default=[3e-4],
               help="One or more learning-rates")
p.add_argument("--bs",  type=int,   nargs="+", default=[16],
               help="One or more batch-sizes")
p.add_argument("--steps", type=int, default=10_000,
               help="num_training_steps override")
p.add_argument("--device", default="cuda")
p.add_argument("--print_every", type=int, default=100)
p.add_argument("--validation_every", type=int, default=500)
p.add_argument("--weight_decay", type=float, nargs="+", default=[1e-2])
args = p.parse_args()

# ────────────────────────────────────────────────────────────────
# 2. Resolve the config class object
# ----------------------------------------------------------------
if ":" in args.cfg_cls:
    module_path, cls_name = args.cfg_cls.split(":")
else:                                   # default module is 'configs'
    module_path, cls_name = "configs", args.cfg_cls

CfgClass = getattr(importlib.import_module(module_path), cls_name)

# ────────────────────────────────────────────────────────────────
# 3. Sweep over the grid and spawn runs
# ----------------------------------------------------------------
for lr, bs, weight_decay in itertools.product(args.lr, args.bs, args.weight_decay):
    # build the dataclass 
    cfg = CfgClass(lr=lr,
                   batch_size=bs,
                   num_training_steps=args.steps,
                   print_every=args.print_every,
                   validation_every=args.validation_every,
                   device=args.device,
                   weight_decay=weight_decay)

    cls_flag = f"{CfgClass.__module__}:{CfgClass.__name__}"
    subprocess.run(
        ["python", "train.py",
         "--cfg_cls",  cls_flag,
         "--cfg_json", json.dumps(cfg.as_dict())],
        check=True
    )
