"""Step 1: (extract and) index the VGGFace2 subset.

Maps each `Name_Gender` folder to its VGGFace2 Class_ID via meta/identity_meta_cleaned.csv,
splits identities into train / dev (official VGGFace2 test identities -> dev) and writes
data/proc/vgg_index_raw.csv (one row per image).
"""
import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from facerec.config import META_DIR, PROC, RAW, ROOT, VGG_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=str(ROOT / "VGGFace2.zip"))
    ap.add_argument("--min-dev-ids", type=int, default=300)
    args = ap.parse_args()

    if not VGG_DIR.exists():
        print("extracting", args.zip)
        zipfile.ZipFile(args.zip).extractall(RAW)
    PROC.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(META_DIR / "identity_meta_cleaned.csv", sep="\t", skipinitialspace=True)
    meta.columns = [c.strip() for c in meta.columns]
    meta["Name"] = meta["Name"].str.strip().str.strip('"')
    meta["Gender"] = meta["Gender"].str.strip()
    gmap = {"m": "Male", "f": "Female"}
    meta["folder"] = meta["Name"] + "_" + meta["Gender"].map(gmap)
    f2id = dict(zip(meta.folder, meta.Class_ID.str.strip()))

    test_ids = {l.split("/")[0] for l in (META_DIR / "test_list.txt").read_text().split()}

    rows, unmapped = [], []
    for d in sorted(p for p in VGG_DIR.iterdir() if p.is_dir()):
        cid = f2id.get(d.name)
        if cid is None:
            unmapped.append(d.name)
            cid = "x_" + d.name  # still a valid identity, just no official ID
        for f in sorted(d.glob("*.jpg")):
            rows.append((f"{d.name}/{f.name}", d.name, cid))
    df = pd.DataFrame(rows, columns=["path", "folder", "class_id"])
    df["split"] = np.where(df.class_id.isin(test_ids), "dev", "train")

    n_dev = df[df.split == "dev"].class_id.nunique()
    if n_dev < args.min_dev_ids:  # fallback: random identity-disjoint dev split
        rng = np.random.default_rng(0)
        ids = df.class_id.unique()
        dev_ids = set(rng.choice(ids, 500, replace=False))
        df["split"] = np.where(df.class_id.isin(dev_ids), "dev", "train")
        print(f"only {n_dev} official test ids present -> random 500-id dev split")

    df.to_csv(PROC / "vgg_index_raw.csv", index=False)
    s = df.groupby("split").agg(images=("path", "size"), ids=("class_id", "nunique"))
    print(s)
    print(f"unmapped folders: {len(unmapped)} {unmapped[:10]}")


if __name__ == "__main__":
    main()
