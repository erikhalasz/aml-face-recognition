"""Verification (1:1) metrics following the LFW protocol."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def best_threshold(scores, same):
    """Threshold maximising accuracy on (scores, same). Candidate thresholds = midpoints of sorted scores."""
    s = np.sort(np.unique(scores))
    cands = np.concatenate([[s[0] - 1e-6], (s[1:] + s[:-1]) / 2, [s[-1] + 1e-6]])
    # vectorised accuracy for all candidates via sorting
    order = np.argsort(scores)
    sc, sm = scores[order], same[order].astype(np.int64)
    # number of positives with score <= t and negatives with score <= t
    pos_le = np.cumsum(sm)
    neg_le = np.cumsum(1 - sm)
    idx = np.searchsorted(sc, cands, side="right")  # count of scores <= cand
    P, N = sm.sum(), (1 - sm).sum()
    tp = P - np.where(idx > 0, pos_le[np.maximum(idx - 1, 0)], 0)
    tn = np.where(idx > 0, neg_le[np.maximum(idx - 1, 0)], 0)
    acc = (tp + tn) / len(scores)
    k = int(np.argmax(acc))
    return float(cands[k]), float(acc[k])


def eer(scores, same):
    fpr, tpr, thr = roc_curve(same, scores)
    fnr = 1 - tpr
    k = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[k] + fnr[k]) / 2), float(thr[k])


def tar_at_far(scores, same, far=1e-3):
    fpr, tpr, _ = roc_curve(same, scores)
    ok = fpr <= far
    return float(tpr[ok].max()) if ok.any() else 0.0


def kfold_accuracy(scores, same, folds):
    """LFW protocol: for each fold, fit threshold on the other 9 folds, test on this one."""
    accs, ths = [], []
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        t, _ = best_threshold(scores[tr], same[tr])
        accs.append(float(((scores[te] >= t) == same[te].astype(bool)).mean()))
        ths.append(t)
    return np.array(accs), np.array(ths)


def verification_report(scores, same, folds=None):
    scores, same = np.asarray(scores, np.float64), np.asarray(same).astype(int)
    rep = {}
    if folds is not None:
        accs, ths = kfold_accuracy(scores, same, np.asarray(folds))
        rep.update(acc_mean=float(accs.mean()), acc_std=float(accs.std()),
                   fold_accs=accs.tolist(), thresholds=ths.tolist())
    else:
        t, a = best_threshold(scores, same)
        rep.update(acc_best=a, threshold=t)
    e, et = eer(scores, same)
    fpr, tpr, _ = roc_curve(same, scores)
    rep.update(auc=float(roc_auc_score(same, scores)), eer=e, eer_threshold=et,
               tar_at_far_1e3=tar_at_far(scores, same, 1e-3),
               tar_at_far_1e2=tar_at_far(scores, same, 1e-2),
               roc_fpr=fpr[:: max(1, len(fpr) // 400)].tolist(),
               roc_tpr=tpr[:: max(1, len(tpr) // 400)].tolist())
    return rep
