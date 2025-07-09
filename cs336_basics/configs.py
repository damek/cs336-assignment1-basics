from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple, Optional
import math

 
@dataclass
class TransformerCfg:
    # ─── Dataset ───────────────────────────────────────────────────────────
    train_data:      str = None
    val_data:        str = None
    vocab_path:      str = None
    merges_path:     str = None
    context_length:  int = 256
    vocab_size:      int = 10000 

    # ─── Model architecture ───────────────────────────────────────────────
    d_model:         int = 512
    num_layers:        int = 4
    num_heads:         int = 16
    d_ff:            int = int(math.ceil(8*d_model // 3 / 64) * 64)   
    rope_theta_parameter:   float = 10000
    eps:                 float = 1e-6

    # ─── Training loop ────────────────────────────────────────────────────
    batch_size:         int = 32
    num_training_steps: int = 10_000
    validation_freq:    int = 500
    print_freq:         int = 100
    seed:               int = 1337

    # ─── Optimiser & schedule ─────────────────────────────────────────────
    lr:              float = 1e-4
    betas:           Tuple[float, float] = (0.9, 0.95)
    weight_decay:    float = 1e-2
    # warmup_steps:    int   = 400
    # cosine_decay:    bool  = True          # disable for constant LR
    # grad_clip:       Optional[float] = 1.0 # None → no clipping

    # ─── Runtime / hardware ───────────────────────────────────────────────
    device:          str = "cuda"
    dtype:           str = "float32"      
    compile:         bool = False          

    # ─── Logging & checkpoints ────────────────────────────────────────────
    wandb_project:   str = ""
    wandb_base_name: str = ""
    save_freq:      int = 1000           
    keep_latest:     bool = True
    resume_from:     str = None
               
    def as_dict(self):
        return asdict(self)
    

@dataclass
class TinyStoriesCfg(TransformerCfg):
    # ─── Dataset ───────────────────────────────────────────────────────────
    train_data:      str = "data/tiny_stories/train.bin"
    val_data:        str = "data/tiny_stories/val.bin"
    vocab_path:      str = "data/tiny_stories/vocab.txt"
    merges_path:     str = "data/tiny_stories/merges.txt"
    wandb_project:   str = "tiny-stories"
    wandb_base_name: str = "lr {lr} batch_size {batch_size} "
    checkpoint_dir:  str = "tiny_stories_runs/"

# utils_cfg.py  (import this from train.py, sweep.py, etc.)
import argparse, json, importlib, pathlib
from configs import TransformerCfg          # base class

def load_cfg() -> TransformerCfg:
    """
    1. Choose subclass with --cfg_cls  (e.g. 'configs:TinyStoriesCfg').
    2. Apply overrides from --cfg_json (inline or @file).
    3. Apply quick scalars   --lr  --batch_size  --wandb_name ...
    """
    p = argparse.ArgumentParser()
    p.add_argument("--cfg_cls",  type=str,
                   default="configs:TransformerCfg",
                   help="MODULE:Class for initial config")
    p.add_argument("--cfg_json", type=str, default=None,
                   help="Raw JSON or '@/path/to/file.json' with overrides")
    # common ad-hoc scalars
    p.add_argument("--lr", type=float)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--wandb_name", type=str)
    args, unknown = p.parse_known_args()    # let training script add more flags

    # 1 — instantiate the chosen subclass
    module_path, class_name = args.cfg_cls.split(":")
    CfgClass = getattr(importlib.import_module(module_path), class_name)
    cfg = CfgClass()

    # 2 — JSON overrides
    if args.cfg_json:
        blob = (pathlib.Path(args.cfg_json[1:]).read_text()
                if args.cfg_json.startswith("@") else args.cfg_json)
        for k, v in json.loads(blob).items():
            setattr(cfg, k, v)

    # 3 — scalar CLI overrides
    if args.lr is not None:          cfg.lr = args.lr
    if args.batch_size is not None:  cfg.batch_size = args.batch_size
    if args.wandb_name is not None:  cfg.wandb_name = args.wandb_name

    return cfg, unknown   # pass unknown back so train.py can parse the rest
