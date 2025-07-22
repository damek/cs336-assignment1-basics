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
import math
import optimization

# ────────────────────────────────────────────────────────────────
# 1. Parse sweep-level CLI flags
# ----------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--cfg_cls",  default="TSPreNormRMS",
               help="Either <ClassName> (assumed in configs.py) or <module:ClassName>")
subs = p.add_subparsers(dest='lr_scheduler', required=True)
constant = subs.add_parser("constant")
constant.add_argument("--lr",  type=float, nargs="+", default=[3e-4],
               help="[constant] one or more learning-rates")
constant.set_defaults(lr=None, min_lr=None, max_lr=None, warmup_end=None, cosine_end=None, stable_end=None, decay_end=None)

cosine = subs.add_parser("cosine")
cosine.add_argument("--max_lr", type=float, nargs="+", default=[1e-1],
               help="[cosine] Maximum learning rate")
cosine.add_argument("--min_lr", type=float, nargs="+", default=[1e-6],
               help="[cosine] Minimum learning rate")
cosine.add_argument("--warmup_end", type=int, nargs="+", default=[0],
               help="[cosine] Number of warmup steps")
cosine.add_argument("--cosine_end", type=int, nargs="+", default=[None],
               help="[cosine] Final iteration of the cosine cycle")
cosine.set_defaults(lr=[None], stable_end=[None], decay_end=[None])

wsd = subs.add_parser("wsd")
wsd.add_argument("--min_lr", type=float, nargs="+", default=[1e-4],
               help="[wsd] Minimum learning rate")
wsd.add_argument("--max_lr", type=float, nargs="+", default=[1e-2],
               help="[wsd] Maximum learning rate")
wsd.add_argument("--warmup_end", type=int, nargs="+", default=[0],
               help="[wsd] Number of warmup steps")
wsd.add_argument("--stable_end", type=int, nargs="+", default=[None],
               help="[wsd] Final iteration of the stable phase")
wsd.add_argument("--decay_end", type=int, nargs="+", default=[None],
               help="[wsd] Final iteration of the decay phase")
wsd.set_defaults(lr=[None], cosine_end=[None])

p.add_argument("--grad_clip", type=float, nargs="+", default=[None])
p.add_argument("--bs",  type=int,   nargs="+", default=[32],
               help="One or more batch-sizes")
p.add_argument("--run_until_step", type=int, default=10_000,
               help="num_training_steps override")
p.add_argument("--device", default="cuda")
p.add_argument("--print_every", type=int, default=100)
p.add_argument("--validation_every", type=int, default=None)
p.add_argument("--weight_decay", type=float, nargs="+", default=[1e-2])
p.add_argument("--d_model", type=int, nargs="+", default=[512])
p.add_argument("--context_length", type=int, nargs="+", default=[256])
p.add_argument("--num_layers", type=int, nargs="+", default=[4])
p.add_argument("--num_heads", type=int, nargs="+", default=[16])
p.add_argument("--d_ff", type=int, nargs="+", default=[None])
p.add_argument("--resume_from", type=str, default=None)
p.add_argument("--compile", type=lambda x: x.lower() == 'true', default=False)
p.add_argument("--time_limit_hours", type=float, nargs="+", default=None)
p.add_argument("--rope_theta", type=float, nargs="+", default=[10000])
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

for lr, bs, weight_decay, d_model, context_length, max_lr, min_lr, warmup_end, cosine_end, grad_clip, num_layers, num_heads, rope_theta, stable_end, decay_end in itertools.product(args.lr, args.bs, args.weight_decay, args.d_model, args.context_length, args.max_lr, args.min_lr, args.warmup_end, args.cosine_end, args.grad_clip, args.num_layers, args.num_heads,args.rope_theta, args.stable_end, args.decay_end):
    # build the dataclass 
    if args.lr_scheduler == "cosine":
        lr = None
        wsd_decay = None
        if cosine_end is None:
            cosine_end = args.run_until_step
        if warmup_end is None:
            warmup_end = 0
        cosine_decay = {"max_lr": max_lr, "min_lr": min_lr, "warmup_end": warmup_end, "cosine_end": cosine_end}
    elif args.lr_scheduler == "wsd":
        lr = None
        cosine_decay = None
        if stable_end is None:
            stable_end = 0
        if decay_end is None:
            decay_end = args.run_until_step
        wsd_decay = {"min_lr": min_lr, "max_lr": max_lr, "warmup_end": warmup_end, "stable_end": stable_end, "decay_end": decay_end}
    else:
        cosine_decay = None
        wsd_decay = None
    print(f"num_heads: {num_heads}, num_layers: {num_layers}")
    cfg = CfgClass(lr=lr,
                   batch_size=bs,
                   run_until_step=args.run_until_step,
                   print_every=args.print_every,
                   validation_every=args.validation_every,
                   device=args.device,
                   weight_decay=weight_decay,
                   d_model=d_model,
                   context_length=context_length,
                   resume_from=args.resume_from,
                   cosine_decay=cosine_decay,
                   lr_scheduler=args.lr_scheduler,
                   grad_clip=grad_clip, 
                   compile=args.compile,
                   time_limit_hours=args.time_limit_hours,
                   num_heads=num_heads,
                   num_layers=num_layers,
                   rope_theta_parameter=rope_theta,
                   wsd_decay=wsd_decay)

    cls_flag = f"{CfgClass.__module__}:{CfgClass.__name__}"
    subprocess.run(
        ["python", "train.py",
         "--cfg_cls",  cls_flag,
         "--cfg_json", json.dumps(cfg.as_dict())],
        check=True
    )
