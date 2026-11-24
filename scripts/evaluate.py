"""Evaluate a model on LFW: verification (10-fold) + open-set identification + size/latency.

  python scripts/evaluate.py --run outputs/r18_arcface
  python scripts/evaluate.py --pretrained facenet      # pipeline sanity check (reference only)
  python scripts/evaluate.py --classical eigenfaces|lbp

Writes <out_dir>/results.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from facerec import identify  # noqa: E402
from facerec.bench import latency_ms, n_params  # noqa: E402
from facerec.config import OUTPUTS, PROC  # noqa: E402
from facerec.dataset import load_vgg, open_memmap  # noqa: E402
from facerec.embed import embed_rows  # noqa: E402
from facerec.verify import verification_report  # noqa: E402


def lfw_data(variant):
    idx = pd.read_csv(PROC / "lfw_index.csv")
    mm = open_memmap(PROC / f"lfw_{variant}.u8", int((PROC / "lfw_nrows.txt").read_text()))
    pairs = pd.read_csv(PROC / "lfw_pairs.csv")
    return idx, mm, pairs


def dev_identification_tau(emb_fn, variant, target_far=0.01):
    """Threshold for FAR<=target chosen on VGGFace2 dev identities (never on LFW)."""
    idx, mm = load_vgg(variant)
    dev = idx[idx.split == "dev"].reset_index(drop=True)
    labels = dev.class_id.values
    g_idx, g_lab, kp, unk = identify.build_dev_protocol(labels)
    used = np.unique(np.concatenate([g_idx, kp, unk]))
    emb = np.zeros((len(dev), 0), np.float32)
    e_used = emb_fn(mm, dev.row.values[used])
    emb = np.zeros((len(dev), e_used.shape[1]), np.float32); emb[used] = e_used
    sc = identify.score_probes(emb, labels, g_idx, g_lab, kp, unk)
    tau = identify.threshold_for_far(sc, target_far)
    far, frr = identify.far_frr(sc, tau)
    return tau, dict(dev_tau=tau, dev_far=far, dev_frr=frr, dev_rank1=sc["rank1"],
                     dev_gallery_ids=int(len(np.unique(g_lab))), dev_unknown_probes=int(len(unk)))


def evaluate_embedder(emb_fn, variant, sim="cos", out_dir=None, tau_from_dev=True):
    idx, mm, pairs = lfw_data(variant)
    emb = emb_fn(mm, idx.row.values)
    a, b = emb[pairs.row_a.values], emb[pairs.row_b.values]
    scores = (a * b).sum(1) if sim == "cos" else sim(a, b)
    res = {"verification": verification_report(scores, pairs.same.values, pairs.fold.values)}
    res["verification"]["n_pairs"] = int(len(pairs))
    res["n_lfw_detection_fallback"] = int(idx.fallback_crop.sum())
    if sim == "cos":
        tau, devinfo = dev_identification_tau(emb_fn, variant) if tau_from_dev else (None, {})
        res["identification"] = identify.identification_report(emb, idx.person.values, tau=tau)
        res["identification"].update(devinfo)
    # failure cases for the report: hardest misclassified pairs at the mean threshold
    if "thresholds" in res["verification"]:
        t = float(np.mean(res["verification"]["thresholds"]))
        wrong = np.flatnonzero((scores >= t) != pairs.same.values.astype(bool))
        worst = wrong[np.argsort(-np.abs(scores[wrong] - t))][:12]
        res["failures"] = [dict(path_a=pairs.path_a[i], path_b=pairs.path_b[i], row_a=int(pairs.row_a[i]),
                                row_b=int(pairs.row_b[i]), same=int(pairs.same[i]), score=float(scores[i]))
                           for i in worst]
    res["scores"] = scores.astype(float).tolist()
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results.json").write_text(json.dumps(res))
    v = res["verification"]
    print(f"LFW acc {v['acc_mean']*100:.2f} ± {v['acc_std']*100:.2f}  AUC {v['auc']:.4f}  EER {v['eer']*100:.2f}%  "
          f"TAR@FAR=0.1% {v['tar_at_far_1e3']*100:.2f}%")
    if "identification" in res:
        i = res["identification"]
        print(f"Ident: rank1 {i['rank1']*100:.2f}%  tau {i['tau']:.3f}  FAR {i['far']*100:.2f}%  FRR {i['frr']*100:.2f}%")
    return res


class FacenetWrap(nn.Module):
    """facenet-pytorch InceptionResnetV1 (trained on VGGFace2) on our 112 crops, resized to 160."""

    def __init__(self):
        super().__init__()
        from facenet_pytorch import InceptionResnetV1
        self.net = InceptionResnetV1(pretrained="vggface2").eval()
        self.fc = nn.Linear(1, 512)  # only to expose out_features=512 for embed_rows

    def forward(self, x):  # x in [-1,1] -> facenet's fixed standardisation is (x*255-127.5)/128 ~= x
        x = F.interpolate(x, size=(160, 160), mode="bilinear", align_corners=False)
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--pretrained", choices=["facenet"])
    ap.add_argument("--classical", choices=["eigenfaces", "lbp"])
    ap.add_argument("--variant", default=None)
    a = ap.parse_args()

    if a.run:
        from facerec.train import load_model
        run = Path(a.run)
        model, cfg = load_model(run)
        variant = a.variant or cfg["variant"]
        res = evaluate_embedder(lambda mm, rows: embed_rows(model, mm, rows), variant,
                                out_dir=run / ("" if variant == cfg["variant"] else f"eval_{variant}"))
        cpu_model = model.float().cpu()
        res["bench"] = dict(params=n_params(model), gpu_ms=latency_ms(model, "cuda"),
                            cpu_ms=latency_ms(cpu_model, "cpu", n=30, warmup=5))
        res["config"] = cfg
        res["dev_log"] = pd.read_csv(run / "log.csv").to_dict(orient="list")
        (run / "results.json").write_text(json.dumps(res))
        print(res["bench"])
    elif a.pretrained:
        model = FacenetWrap().cuda().eval()
        out = OUTPUTS / "ref_facenet_vggface2"
        res = evaluate_embedder(lambda mm, rows: embed_rows(model, mm, rows, amp=False), "aligned", out_dir=out)
        res["bench"] = dict(params=n_params(model.net), gpu_ms=latency_ms(model, "cuda"))
        (out / "results.json").write_text(json.dumps(res))
    else:
        from facerec import classical
        idx, mm = load_vgg("aligned")
        rng = np.random.default_rng(0)
        tr_rows = np.sort(rng.choice(idx[idx.split == "train"].row.values, 50000, replace=False))
        out = OUTPUTS / f"classical_{a.classical}"
        if a.classical == "eigenfaces":
            ef = classical.Eigenfaces(200).fit(mm[tr_rows])
            fn = lambda m, rows: ef.embed(m[np.sort(rows)][np.argsort(np.argsort(rows))])  # noqa: E731
            res = evaluate_embedder(fn, "aligned", out_dir=out)
        else:
            fn = lambda m, rows: classical.lbp_hist(m[np.sort(rows)][np.argsort(np.argsort(rows))])  # noqa: E731
            res = evaluate_embedder(fn, "aligned", sim=classical.chi2_similarity, out_dir=out)


if __name__ == "__main__":
    main()
