"""Step 2: detect -> 5 landmarks -> align (112x112) -> quality filter, plus the crop-only variant.

Writes data/proc/vgg_aligned.u8, data/proc/vgg_croponly.u8 (uint8 memmaps, one row per image of
vgg_index_raw.csv) and data/proc/vgg_index.csv with detection/quality stats and a `keep` flag.
Resumable: progress is checkpointed every few thousand images.
Finally a fixed face mask per variant is estimated on train images and applied to all images.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from facerec.config import PROC, VGG_DIR  # noqa: E402
from facerec.dataset import open_memmap  # noqa: E402
from facerec.preprocess import apply_mask, compute_mask, process  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--crop-frac", type=float, default=1.0)  # images are already tight crops
    ap.add_argument("--pad", type=int, default=64)
    ap.add_argument("--min-face", type=int, default=100)
    ap.add_argument("--min-prob", type=float, default=0.9)
    ap.add_argument("--blur-pct", type=float, default=2.0)
    ap.add_argument("--mask-thr", type=float, default=0.8)
    args = ap.parse_args()

    raw = pd.read_csv(PROC / "vgg_index_raw.csv")
    st, n = process(VGG_DIR, raw.path.values, "vgg", 256, args.crop_frac, args.min_face, args.batch, args.workers, args.limit, pad=args.pad)
    idx = pd.concat([raw.iloc[:n].reset_index(drop=True), st], axis=1)
    idx.insert(0, "row", np.arange(n))

    detected = idx.det_prob.notna()
    blur_thr = np.nanpercentile(idx.blur[detected], args.blur_pct)
    reason = np.where(~detected, "no_face",
             np.where(idx.det_prob < args.min_prob, "low_prob",
             np.where(idx.blur < blur_thr, "blurry",
             np.where((idx.bright < 30) | (idx.bright > 225), "exposure", ""))))
    idx["drop_reason"] = reason
    idx["keep"] = reason == ""
    idx.to_csv(PROC / "vgg_index.csv", index=False)
    (PROC / "vgg_nrows.txt").write_text(str(n))
    print("blur threshold", blur_thr)
    print(idx.drop_reason.replace("", "kept").value_counts())

    # fixed face mask per variant, estimated from train images only, applied to every image
    rng = np.random.default_rng(0)
    tr_rows = idx.row[idx.keep & (idx.split == "train")].values
    sample = rng.choice(tr_rows, min(20000, len(tr_rows)), replace=False)
    for variant in ("aligned", "croponly"):
        mm = open_memmap(PROC / f"vgg_{variant}.u8", n, "r+")
        mask, frac = compute_mask(mm, sample, thr=args.mask_thr)
        np.save(PROC / f"mask_{variant}.npy", mask)
        np.save(PROC / f"maskfrac_{variant}.npy", frac)
        apply_mask(mm, mask)
        print(f"mask {variant}: {mask.mean() * 100:.1f}% of pixels kept")


if __name__ == "__main__":
    main()
