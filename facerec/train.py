"""Training loop: backbone + (ArcFace | softmax) head, SGD, warmup + cosine LR, AMP,
per-epoch dev verification (early-stopping by best dev accuracy), checkpoint/resume."""
from __future__ import annotations

import csv
import json
import math
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import PROC, TrainConfig
from .dataset import FaceDataset, load_vgg, sample_pairs, to_input
from .embed import embed_rows
from .heads import build_head
from .models import build_backbone
from .verify import verification_report


def lr_at(step, total, warmup, base):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * p))


def dev_eval(model, mm, dev_rows, dev_labels, n_pairs, seed=0):
    I, J, S = sample_pairs(dev_labels, n_pairs, seed)
    uniq, inv = np.unique(np.concatenate([I, J]), return_inverse=True)
    emb = embed_rows(model, mm, dev_rows[uniq])
    ei, ej = emb[inv[: len(I)]], emb[inv[len(I):]]
    rep = verification_report((ei * ej).sum(1), S)
    return rep["acc_best"], rep["eer"], rep["auc"]


def run(cfg: TrainConfig):
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    # cudnn.benchmark + fp16/channels_last produced NaNs on this GPU and gave no speed-up in fp32
    torch.backends.cudnn.benchmark = False
    out = cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))

    idx, mm = load_vgg(cfg.variant)
    n_rows = mm.shape[0]
    tr = idx[idx.split == "train"]
    if cfg.limit_classes:
        keep = sorted(tr.class_id.unique())[: cfg.limit_classes]
        tr = tr[tr.class_id.isin(keep)]
    classes = sorted(tr.class_id.unique())
    cmap = {c: i for i, c in enumerate(classes)}
    labels = tr.class_id.map(cmap).values
    dev = idx[idx.split == "dev"]
    if cfg.limit_classes:  # smoke test: dev = held-out images of the same few classes is meaningless, keep real dev
        dev = dev[dev.class_id.isin(sorted(dev.class_id.unique())[:100])]
    dev_rows, dev_labels = dev.row.values, dev.class_id.values
    print(f"[{cfg.name}] train imgs={len(tr)} classes={len(classes)} dev imgs={len(dev)} ids={dev.class_id.nunique()}")

    aug = None
    if cfg.degrade_aug:
        from .augment import Degrade
        aug = Degrade(np.load(PROC / f"mask_{cfg.variant}.npy"))
    ds = FaceDataset(PROC / f"vgg_{cfg.variant}.u8", n_rows, tr.row.values, labels, aug=aug)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers,
                    pin_memory=True, drop_last=True, persistent_workers=cfg.workers > 0)

    dev_ = "cuda"
    model = build_backbone(cfg.backbone, cfg.emb_dim, cfg.dropout).to(dev_)
    head = build_head(cfg.head, cfg.emb_dim, len(classes), cfg.s, cfg.m).to(dev_)
    # no weight decay on BN / bias / PReLU
    decay, no_decay = [], []
    for p in list(model.parameters()) + list(head.parameters()):
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    opt = torch.optim.SGD([{"params": decay, "weight_decay": cfg.weight_decay},
                           {"params": no_decay, "weight_decay": 0.0}], lr=cfg.lr, momentum=cfg.momentum)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    steps_per_epoch = len(dl) // cfg.accum_steps
    if cfg.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, cfg.max_steps_per_epoch)
    total = steps_per_epoch * cfg.epochs
    warmup = int(cfg.warmup_epochs * steps_per_epoch)

    start_epoch, step, best = 0, 0, -1.0
    ck = out / "last.pt"
    if ck.exists():
        st = torch.load(ck, map_location=dev_, weights_only=False)
        model.load_state_dict(st["model"]); head.load_state_dict(st["head"])
        opt.load_state_dict(st["opt"]); scaler.load_state_dict(st["scaler"])
        start_epoch, step, best = st["epoch"] + 1, st["step"], st["best"]
        print(f"resumed from epoch {start_epoch}")

    log_path = out / "log.csv"
    new_log = not log_path.exists()
    logf = open(log_path, "a", newline="")
    logw = csv.writer(logf)
    if new_log:
        logw.writerow(["epoch", "step", "lr", "train_loss", "train_acc", "dev_acc", "dev_eer", "dev_auc", "secs"])

    for epoch in range(start_epoch, cfg.epochs):
        model.train(); head.train()
        t0 = time.time()
        tot_loss = tot_acc = n = 0
        opt.zero_grad(set_to_none=True)
        for it, (x, y) in enumerate(dl):
            if it // cfg.accum_steps >= steps_per_epoch:
                break
            x = to_input(x, dev_)
            y = y.to(dev_, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=cfg.amp):
                emb = model(x)
            loss, logits = head(emb.float(), y)
            scaler.scale(loss / cfg.accum_steps).backward()
            if (it + 1) % cfg.accum_steps == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(step, total, warmup, cfg.lr)
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 5.0)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
            tot_loss += loss.item(); tot_acc += (logits.argmax(1) == y).float().mean().item(); n += 1
            if it % 200 == 0:
                el = time.time() - t0
                print(f"ep {epoch} it {it}/{steps_per_epoch * cfg.accum_steps} loss {tot_loss / n:.3f} "
                      f"acc {tot_acc / n:.3f} lr {opt.param_groups[0]['lr']:.4f} {n * cfg.batch_size / max(el, 1e-6):.0f} img/s",
                      flush=True)
        dacc, deer, dauc = dev_eval(model, mm, dev_rows, dev_labels, cfg.dev_pairs)
        secs = time.time() - t0
        print(f"== epoch {epoch}: loss {tot_loss / n:.3f} dev_acc {dacc:.4f} dev_eer {deer:.4f} ({secs / 60:.1f} min)", flush=True)
        logw.writerow([epoch, step, opt.param_groups[0]["lr"], tot_loss / n, tot_acc / n, dacc, deer, dauc, secs]); logf.flush()
        if dacc > best:
            best = dacc
            torch.save({"model": model.state_dict(), "epoch": epoch, "dev_acc": dacc, "cfg": cfg.to_dict()}, out / "best.pt")
        torch.save({"model": model.state_dict(), "head": head.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "step": step, "best": best}, ck)
    logf.close()
    print(f"[{cfg.name}] done, best dev acc {best:.4f}")
    return best


def load_model(run_dir, device="cuda"):
    st = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    c = st["cfg"]
    m = build_backbone(c["backbone"], c["emb_dim"], c.get("dropout", 0.0)).to(device)
    m.load_state_dict(st["model"]); m.eval()
    return m, c
