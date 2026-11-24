"""Memmap-backed face datasets. Images are stored as uint8 (N,112,112,3) RGB arrays;
normalisation to [-1,1] happens on the GPU (see `to_input`)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import IMG_SIZE, PROC


def open_memmap(path: Path, n: int, mode="r"):
    return np.memmap(path, dtype=np.uint8, mode=mode, shape=(n, IMG_SIZE, IMG_SIZE, 3))


def load_vgg(variant: str = "aligned"):
    """Returns (index DataFrame of kept images, memmap) for the processed VGGFace2 data.
    Index columns: row (memmap row), path, folder, class_id, split, ..."""
    idx = pd.read_csv(PROC / "vgg_index.csv")
    n_rows = int((PROC / "vgg_nrows.txt").read_text())
    mm = open_memmap(PROC / f"vgg_{variant}.u8", n_rows)
    return idx[idx.keep].reset_index(drop=True), mm


class FaceDataset(Dataset):
    """Training dataset: returns (uint8 HWC tensor, label). Horizontal flip augmentation."""

    def __init__(self, mm_path: Path, n_rows: int, rows: np.ndarray, labels: np.ndarray, flip=True, aug=None):
        self.mm_path, self.n_rows = mm_path, n_rows
        self.rows, self.labels, self.flip = rows.astype(np.int64), labels.astype(np.int64), flip
        self.aug = aug  # optional callable (img_u8, rng) -> img_u8, e.g. augment.Degrade
        self._mm = None
        self._rng = None

    def __getstate__(self):  # never pickle an open memmap into worker processes
        d = self.__dict__.copy()
        d["_mm"] = None
        d["_rng"] = None
        return d

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self._mm is None:  # opened lazily so it works with Windows spawn workers
            self._mm = open_memmap(self.mm_path, self.n_rows)
        img = np.array(self._mm[self.rows[i]])
        if self.flip and np.random.rand() < 0.5:
            img = img[:, ::-1].copy()
        if self.aug is not None:
            if self._rng is None:  # torch gives every worker (and every epoch) a different seed
                self._rng = np.random.default_rng(torch.initial_seed() % 2**32)
            img = self.aug(img, self._rng)
        return torch.from_numpy(img), self.labels[i]


def to_input(batch_u8: torch.Tensor, device="cuda") -> torch.Tensor:
    """uint8 (B,H,W,3) -> float (B,3,H,W) in [-1,1] on device."""
    # .contiguous(): permute() alone yields a channels-last layout, which is ~10x slower on this GPU
    x = batch_u8.to(device, non_blocking=True).permute(0, 3, 1, 2).contiguous().float()
    return x.div_(127.5).sub_(1.0)


def sample_pairs(labels: np.ndarray, n_pairs: int, seed: int = 0):
    """Balanced verification pairs (n/2 genuine, n/2 impostor) over item indices with given labels.
    Returns (i, j, same) arrays."""
    rng = np.random.default_rng(seed)
    by_cls = {}
    for k, c in enumerate(labels):
        by_cls.setdefault(c, []).append(k)
    multi = [c for c, v in by_cls.items() if len(v) >= 2]
    classes = list(by_cls)
    I, J, S = [], [], []
    for _ in range(n_pairs // 2):
        c = multi[rng.integers(len(multi))]
        a, b = rng.choice(by_cls[c], 2, replace=False)
        I.append(a); J.append(b); S.append(1)
    for _ in range(n_pairs // 2):
        c1, c2 = rng.choice(len(classes), 2, replace=False)
        I.append(rng.choice(by_cls[classes[c1]])); J.append(rng.choice(by_cls[classes[c2]])); S.append(0)
    return np.array(I), np.array(J), np.array(S)
