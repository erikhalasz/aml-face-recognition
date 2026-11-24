"""Step 4: identity-disjointness between VGGFace2 (train/dev) and LFW (test).

Mode 1 (before training):  name-list overlap. Every VGGFace2 identity whose normalised name
                           matches an LFW person is removed (keep=False, drop_reason=lfw_overlap).
Mode 2 (--run <dir>, after training): embedding-based scan. Each LFW person's mean embedding is
                           compared with every VGGFace2 train-class centroid; pairs above a cosine
                           threshold are listed as possible identity leaks (outputs/leakage_scan.csv).
"""
import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from facerec.config import OUTPUTS, PROC  # noqa: E402


def norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", s)


def name_overlap():
    vgg = pd.read_csv(PROC / "vgg_index.csv", keep_default_na=False)
    lfw = pd.read_csv(PROC / "lfw_index.csv")
    lfw_names = {norm(p) for p in lfw.person.unique()}
    vgg["name"] = vgg.folder.str.replace(r"_(Male|Female)$", "", regex=True)
    ov = vgg.name.map(norm).isin(lfw_names)
    ov_ids = sorted(vgg.folder[ov].unique())
    vgg.loc[ov, "keep"] = False
    vgg.loc[ov, "drop_reason"] = "lfw_overlap"
    vgg.drop(columns="name").to_csv(PROC / "vgg_index.csv", index=False)
    rep = {"overlapping_identities": len(ov_ids), "removed_images": int(ov.sum()),
           "by_split": vgg[ov].groupby("split").folder.nunique().to_dict(), "names": ov_ids}
    (PROC / "leakage_names.json").write_text(json.dumps(rep, indent=2))
    print(f"name overlap: {len(ov_ids)} identities, {int(ov.sum())} images removed; {rep['by_split']}")
    k = vgg[vgg.keep.astype(str) == "True"] if vgg.keep.dtype == object else vgg[vgg.keep]
    print(k.groupby("split").agg(images=("path", "size"), ids=("class_id", "nunique")))


def embedding_scan(run: Path, thr: float):
    import torch  # noqa: F401
    from facerec.dataset import load_vgg, open_memmap
    from facerec.embed import embed_rows
    from facerec.train import load_model

    model, cfg = load_model(run)
    idx, mm = load_vgg(cfg["variant"])
    tr = idx[idx.split == "train"]
    # centroid per class from up to 20 images (keeps the scan to ~10 min)
    sub = tr.groupby("class_id").head(20)  # (groupby.apply drops the key column in pandas >= 2.2)
    e = embed_rows(model, mm, sub.row.values)
    cls = sub.class_id.values
    names = sub.groupby("class_id").folder.first()
    uc = np.unique(cls)
    C = np.stack([e[cls == c].mean(0) for c in uc]); C /= np.linalg.norm(C, axis=1, keepdims=True)

    lfw = pd.read_csv(PROC / "lfw_index.csv")
    lmm = open_memmap(PROC / f"lfw_{cfg['variant']}.u8", int((PROC / "lfw_nrows.txt").read_text()))
    le = embed_rows(model, lmm, lfw.row.values)
    people = lfw.person.values
    up = np.unique(people)
    P = np.stack([le[people == p].mean(0) for p in up]); P /= np.linalg.norm(P, axis=1, keepdims=True)
    S = P @ C.T
    j = S.argmax(1)
    df = pd.DataFrame({"lfw_person": up, "vgg_identity": names.loc[uc[j]].values, "cosine": S.max(1)})
    df = df.sort_values("cosine", ascending=False)
    flagged = df[df.cosine >= thr]
    OUTPUTS.mkdir(exist_ok=True)
    df.head(200).to_csv(OUTPUTS / "leakage_scan.csv", index=False)
    print(f"embedding scan: {len(flagged)} LFW people with centroid cosine >= {thr}")
    print(flagged.head(30).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="trained run dir for the embedding scan")
    ap.add_argument("--thr", type=float, default=0.7)
    a = ap.parse_args()
    if a.run:
        embedding_scan(Path(a.run), a.thr)
    else:
        name_overlap()


if __name__ == "__main__":
    main()
