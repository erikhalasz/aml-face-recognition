"""Open-set identification (1:N) with an enrolled gallery and an 'Unknown' reject option.

Gallery: one template per enrolled person = normalised mean of n_enroll embeddings.
Known probes: remaining images of enrolled people. Unknown probes: people not in the gallery.
  Rank-1   = closed-set accuracy on known probes (argmax over gallery).
  FAR(FPIR)= fraction of unknown probes whose best score >= tau (accepted as someone).
  FRR(FNIR)= fraction of known probes that are not correctly identified at tau
             (rejected, or best match is the wrong person).
"""
from __future__ import annotations

import numpy as np


def build_protocol(labels: np.ndarray, n_enroll=3, min_imgs=5, seed=0):
    """Split items into gallery / known-probe / unknown-probe index arrays.
    Persons with >= min_imgs images are enrolled (n_enroll random images in the gallery),
    persons with exactly 1 image become unknowns."""
    rng = np.random.default_rng(seed)
    by = {}
    for k, c in enumerate(labels):
        by.setdefault(c, []).append(k)
    g_idx, g_lab, kp, unk = [], [], [], []
    for c, items in by.items():
        items = list(items)
        if len(items) >= min_imgs:
            rng.shuffle(items)
            g_idx += items[:n_enroll]; g_lab += [c] * n_enroll; kp += items[n_enroll:]
        elif len(items) == 1:
            unk += items
    return np.array(g_idx), np.array(g_lab), np.array(kp), np.array(unk)


def build_dev_protocol(labels: np.ndarray, n_enroll=3, max_probes=20, seed=0):
    """Mirror protocol on validation identities (all have many images): half the identities are
    enrolled (n_enroll gallery images, up to max_probes known probes), the other half are unknowns
    (up to max_probes probes each). Used only to choose the decision threshold."""
    rng = np.random.default_rng(seed)
    ids = np.unique(labels)
    rng.shuffle(ids)
    enrolled = set(ids[: len(ids) // 2])
    g_idx, g_lab, kp, unk = [], [], [], []
    for c in ids:
        items = np.flatnonzero(labels == c)
        rng.shuffle(items)
        if c in enrolled:
            g_idx += list(items[:n_enroll]); g_lab += [c] * min(n_enroll, len(items))
            kp += list(items[n_enroll:n_enroll + max_probes])
        else:
            unk += list(items[:max_probes])
    return np.array(g_idx), np.array(g_lab), np.array(kp), np.array(unk)


def templates(emb, g_idx, g_lab):
    classes = np.unique(g_lab)
    T = np.stack([emb[g_idx[g_lab == c]].mean(0) for c in classes])
    return T / np.linalg.norm(T, axis=1, keepdims=True), classes


def score_probes(emb, labels, g_idx, g_lab, kp, unk):
    T, classes = templates(emb, g_idx, g_lab)
    sk = emb[kp] @ T.T
    su = emb[unk] @ T.T if len(unk) else np.zeros((0, len(classes)))
    best_k = classes[sk.argmax(1)]
    return dict(known_max=sk.max(1), known_correct=(best_k == labels[kp]),
                unk_max=su.max(1) if len(unk) else np.zeros(0),
                rank1=float((best_k == labels[kp]).mean()),
                rank5=float(np.mean([labels[p] in classes[np.argsort(-r)[:5]] for p, r in zip(kp, sk)])))


def far_frr(sc, tau):
    far = float((sc["unk_max"] >= tau).mean()) if len(sc["unk_max"]) else float("nan")
    frr = float(1 - ((sc["known_max"] >= tau) & sc["known_correct"]).mean())
    return far, frr


def threshold_for_far(sc, target_far=0.01):
    """Smallest tau with FAR <= target (minimises FRR subject to the FAR constraint)."""
    u = np.sort(sc["unk_max"])
    k = int(np.ceil(len(u) * (1 - target_far)))
    return float(u[min(k, len(u) - 1)] + 1e-6) if len(u) else 0.5


def det_curve(sc, n=200):
    lo = min(sc["known_max"].min(), sc["unk_max"].min() if len(sc["unk_max"]) else 1)
    hi = max(sc["known_max"].max(), sc["unk_max"].max() if len(sc["unk_max"]) else -1)
    taus = np.linspace(lo, hi, n)
    pts = [far_frr(sc, t) for t in taus]
    return taus.tolist(), [p[0] for p in pts], [p[1] for p in pts]


def identification_report(emb, labels, tau=None, n_enroll=3, min_imgs=5, seed=0, target_far=0.01):
    labels = np.asarray(labels)
    g_idx, g_lab, kp, unk = build_protocol(labels, n_enroll, min_imgs, seed)
    sc = score_probes(emb, labels, g_idx, g_lab, kp, unk)
    tau_own = threshold_for_far(sc, target_far)
    if tau is None:
        tau = tau_own
    far, frr = far_frr(sc, tau)
    taus, fars, frrs = det_curve(sc)
    return dict(n_gallery_ids=int(len(np.unique(g_lab))), n_known_probes=int(len(kp)),
                n_unknown_probes=int(len(unk)), rank1=sc["rank1"], rank5=sc["rank5"],
                tau=float(tau), far=far, frr=frr, oracle_tau_far1=tau_own,
                det_tau=taus, det_far=fars, det_frr=frrs)
