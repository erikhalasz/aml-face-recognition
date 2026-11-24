"""Batched embedding extraction with flip test-time augmentation."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import to_input


@torch.no_grad()
def embed_array(model, imgs: np.ndarray, batch=128, device="cuda", flip_tta=True, amp=False) -> np.ndarray:
    """imgs: uint8 (N,112,112,3) array or memmap (any indexable). Returns L2-normalised float32 (N,D)."""
    model.eval()
    out = []
    for s in range(0, len(imgs), batch):
        x = to_input(torch.from_numpy(np.ascontiguousarray(imgs[s:s + batch])), device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and device == "cuda"):
            e = model(x).float()
            if flip_tta:
                e = e + model(torch.flip(x, dims=[3])).float()
        out.append(F.normalize(e, dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 512), np.float32)


def embed_rows(model, mm, rows: np.ndarray, batch=128, **kw) -> np.ndarray:
    """Embed selected memmap rows (keeps memory bounded)."""
    rows = np.asarray(rows)
    order = np.argsort(rows)  # sequential reads are much faster on a memmap
    res = np.empty((len(rows), model.fc.out_features), np.float32)
    for s in range(0, len(rows), 4096):
        sel = order[s:s + 4096]
        res[sel] = embed_array(model, mm[rows[sel]], batch=batch, **kw)
    return res
