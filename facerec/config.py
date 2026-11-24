"""Project paths and training configuration."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "proc"
OUTPUTS = ROOT / "outputs"
VGG_DIR = RAW / "VGGFace2"
META_DIR = RAW / "meta"
LFW_DIR = RAW / "lfw"

IMG_SIZE = 112


@dataclass
class TrainConfig:
    name: str = "r18_arcface"
    backbone: str = "r18"            # r18 | r50
    head: str = "arcface"            # arcface | softmax
    variant: str = "aligned"         # aligned | croponly
    emb_dim: int = 512
    s: float = 64.0
    m: float = 0.5
    epochs: int = 20
    batch_size: int = 32             # fits in 4 GB at fp32 without spilling into shared memory
    accum_steps: int = 1
    lr: float = 0.035
    momentum: float = 0.9
    weight_decay: float = 5e-4
    warmup_epochs: float = 1.0
    workers: int = 6
    amp: bool = False                # TU117 (Quadro T2000) has no tensor cores: fp16 is ~4x slower than fp32
    dropout: float = 0.0
    seed: int = 0
    degrade_aug: bool = False        # train-time blur / downscale / JPEG / small scale-shift (augment.Degrade)
    # debugging / smoke tests
    limit_classes: int = 0           # >0: use only the first N train classes
    max_steps_per_epoch: int = 0     # >0: truncate epochs
    dev_pairs: int = 6000
    extra: dict = field(default_factory=dict)

    @property
    def out_dir(self) -> Path:
        return OUTPUTS / self.name

    def to_dict(self):
        return asdict(self)


def load_config(path: str | Path, **overrides) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    d.update({k: v for k, v in overrides.items() if v is not None})
    return TrainConfig(**d)
