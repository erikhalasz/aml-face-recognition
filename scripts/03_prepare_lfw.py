"""Step 3: download LFW (+ official pairs.txt), align with the same pipeline as VGGFace2.

Outputs in data/proc:
  lfw_aligned.u8 / lfw_croponly.u8   uint8 memmaps, one row per LFW image
  lfw_index.csv                      row, path, person, det_prob, ...
  lfw_pairs.csv                      the 6,000 official pairs: row_a, row_b, same, fold
Test images are never dropped: if MTCNN misses a face, the crop-only image is used instead
(and flagged), so every one of the 6,000 pairs is evaluated.
"""
import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from facerec.config import LFW_DIR, PROC  # noqa: E402
from facerec.dataset import open_memmap  # noqa: E402
from facerec.preprocess import apply_mask, process  # noqa: E402

# figshare mirror used by scikit-learn's fetch_lfw_*
URLS = {"lfw.tgz": "https://ndownloader.figshare.com/files/5976018",
        "pairs.txt": "https://ndownloader.figshare.com/files/5976006"}


def download():
    LFW_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        dst = LFW_DIR / name
        if not dst.exists():
            print("downloading", name)
            req = urllib.request.Request(url, headers={"User-Agent": "python-urllib"})
            with urllib.request.urlopen(req) as r, open(dst, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
    img_root = LFW_DIR / "lfw"
    if not img_root.exists():
        print("extracting lfw.tgz")
        with tarfile.open(LFW_DIR / "lfw.tgz") as t:
            t.extractall(LFW_DIR, filter="data")
    return img_root


def parse_pairs(path: Path):
    lines = path.read_text().strip().splitlines()
    n_folds, n = map(int, lines[0].split())
    out, k = [], 1
    for fold in range(n_folds):
        for same in (1, 0):
            for _ in range(n):
                p = lines[k].split(); k += 1
                if same:
                    a, b = (p[0], int(p[1])), (p[0], int(p[2]))
                else:
                    a, b = (p[0], int(p[1])), (p[2], int(p[3]))
                out.append((f"{a[0]}/{a[0]}_{a[1]:04d}.jpg", f"{b[0]}/{b[0]}_{b[1]:04d}.jpg", same, fold))
    return pd.DataFrame(out, columns=["path_a", "path_b", "same", "fold"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--crop-frac", type=float, default=0.5)
    args = ap.parse_args()

    root = download()
    paths = sorted(str(p.relative_to(root)).replace("\\", "/") for p in root.glob("*/*.jpg"))
    st, n = process(root, paths, "lfw", 250, args.crop_frac, 40, args.batch, args.workers)
    idx = pd.DataFrame({"row": np.arange(n), "path": paths, "person": [p.split("/")[0] for p in paths]})
    idx = pd.concat([idx, st], axis=1)
    idx["fallback_crop"] = idx.det_prob.isna()
    if idx.fallback_crop.any():  # fill missing aligned rows with the crop-only image
        al = open_memmap(PROC / "lfw_aligned.u8", n, "r+")
        cr = open_memmap(PROC / "lfw_croponly.u8", n, "r")
        for r in idx.row[idx.fallback_crop]:
            al[r] = cr[r]
        al.flush()
    # same fixed face mask as the training data (estimated on VGGFace2 train only)
    for variant in ("aligned", "croponly"):
        apply_mask(open_memmap(PROC / f"lfw_{variant}.u8", n, "r+"), np.load(PROC / f"mask_{variant}.npy"))
    idx.to_csv(PROC / "lfw_index.csv", index=False)
    (PROC / "lfw_nrows.txt").write_text(str(n))

    pairs = parse_pairs(LFW_DIR / "pairs.txt")
    p2r = dict(zip(idx.path, idx.row))
    pairs["row_a"] = pairs.path_a.map(p2r)
    pairs["row_b"] = pairs.path_b.map(p2r)
    assert pairs[["row_a", "row_b"]].notna().all().all(), "pair image missing"
    pairs.to_csv(PROC / "lfw_pairs.csv", index=False)
    print(f"LFW: {n} images, {idx.person.nunique()} people, {len(pairs)} pairs, "
          f"{int(idx.fallback_crop.sum())} images without detection (crop fallback)")


if __name__ == "__main__":
    main()
