"""Shared detection/alignment pass used for both VGGFace2 and LFW."""
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .align import FILL, Aligner, bbox_crop, center_crop, quality_stats, similarity_warp
from .config import PROC
from .dataset import open_memmap

def imread_rgb(path: Path, size=None):
    buf = np.fromfile(str(path), dtype=np.uint8)  # unicode-safe on Windows
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size and img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


class Imgs(Dataset):
    def __init__(self, root, paths, size):
        self.root, self.paths, self.size = root, paths, size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = imread_rgb(self.root / self.paths[i], self.size)
        if img is None:
            img = np.zeros((self.size, self.size, 3), np.uint8)
        return torch.from_numpy(img)


def process(root, paths, out_prefix, size, crop_frac, min_face, batch, workers, limit=0, pad=0):
    """Shared by VGGFace2 and LFW. Returns DataFrame of per-image stats."""
    n = len(paths) if not limit else min(limit, len(paths))
    paths = list(paths)[:n]
    stats_path = PROC / f"{out_prefix}_stats.npy"
    prog_path = PROC / f"{out_prefix}_progress.txt"
    al_path, cr_path = PROC / f"{out_prefix}_aligned.u8", PROC / f"{out_prefix}_croponly.u8"
    start = int(prog_path.read_text()) if prog_path.exists() else 0
    mode = "r+" if start > 0 and al_path.exists() else "w+"
    al, cr = open_memmap(al_path, n, mode), open_memmap(cr_path, n, mode)
    # stats: prob, blur, bright, lm(10)
    stats = np.load(stats_path) if start > 0 and stats_path.exists() else np.full((n, 13), np.nan, np.float32)

    aligner = Aligner(min_face=min_face)
    dl = DataLoader(Imgs(root, paths[start:], size), batch_size=batch, num_workers=workers)
    t0, done = time.time(), start
    for b in dl:
        imgs = list(b.numpy())
        # tight crops: add a black border so the detector sees the whole face and the
        # warp to the template does not need to sample outside the image
        padded = [cv2.copyMakeBorder(im, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(FILL,) * 3) for im in imgs] if pad else imgs
        dets = aligner.detect(padded)
        for k, (img, pimg, d) in enumerate(zip(imgs, padded, dets)):
            i = done + k
            if d is None:
                cr[i] = center_crop(img, crop_frac)  # no detection: fallback (image is dropped / flagged)
                continue
            p, lm, box = d
            cr[i] = bbox_crop(pimg, box)
            face = similarity_warp(pimg, lm)
            lm = lm - pad  # store landmarks in original image coordinates
            if face is None:
                continue
            al[i] = face
            blur, bright = quality_stats(face)
            stats[i, :3] = (p, blur, bright)
            stats[i, 3:] = lm.reshape(-1)
        done += len(imgs)
        if (done // batch) % 40 == 0 or done == n:
            al.flush(); cr.flush(); np.save(stats_path, stats); prog_path.write_text(str(done))
            r = (done - start) / (time.time() - t0)
            print(f"{done}/{n}  {r:.0f} img/s  eta {(n - done) / max(r, 1e-6) / 60:.1f} min", flush=True)
    al.flush(); cr.flush(); np.save(stats_path, stats); prog_path.write_text(str(done))
    df = pd.DataFrame(stats, columns=["det_prob", "blur", "bright"] + [f"lm{j}" for j in range(10)])
    return df, n


# ---------------------------------------------------------------------------------------------
# Fixed face mask. VGGFace2 images in this subset are pre-masked (white outside the face), LFW
# images are not. One mask per variant is estimated from VGGFace2 *train* images only and applied
# to every image (train, dev, LFW), so all inputs share the same white background outside it.

def compute_mask(mm, rows, thr=0.9, chunk=4096):
    """Pixels that are face (not white) in at least `thr` of the given images."""
    cnt = np.zeros(mm.shape[1:3], np.float64)
    rows = np.sort(rows)
    for s in range(0, len(rows), chunk):
        x = mm[rows[s:s + chunk]]
        cnt += (~(x >= 245).all(-1)).sum(0)
    frac = cnt / len(rows)
    mask = frac >= thr
    # keep the largest connected component and fill holes
    n, lab = cv2.connectedComponents(mask.astype(np.uint8))
    if n > 2:
        sizes = [(lab == k).sum() for k in range(1, n)]
        mask = lab == (1 + int(np.argmax(sizes)))
    inv = (~mask).astype(np.uint8)
    n, lab = cv2.connectedComponents(inv)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    holes = np.isin(lab, [k for k in range(1, n) if k not in border])
    return mask | holes, frac


def apply_mask(mm, mask, chunk=8192):
    """In place: set all pixels outside `mask` to the fill colour (idempotent)."""
    out = ~mask
    for s in range(0, mm.shape[0], chunk):
        x = np.array(mm[s:s + chunk])
        x[:, out] = FILL
        mm[s:s + chunk] = x
    mm.flush()
