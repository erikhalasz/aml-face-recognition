"""Classical lower bounds: Eigenfaces (PCA) and LBP histograms on aligned grayscale faces."""
from __future__ import annotations

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.decomposition import PCA


def to_gray(imgs: np.ndarray) -> np.ndarray:
    return np.stack([cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2GRAY) for im in imgs]).astype(np.float32)


class Eigenfaces:
    def __init__(self, n_components=200):
        self.pca = PCA(n_components=n_components, whiten=True, svd_solver="randomized", random_state=0)

    def fit(self, imgs):
        g = to_gray(imgs).reshape(len(imgs), -1) / 255.0
        self.pca.fit(g)
        return self

    def embed(self, imgs):
        g = to_gray(imgs).reshape(len(imgs), -1) / 255.0
        z = self.pca.transform(g)
        return z / np.linalg.norm(z, axis=1, keepdims=True)


def lbp_hist(imgs, P=8, R=1, grid=7):
    """Uniform LBP (59 bins with 'nri_uniform') histograms over a grid x grid layout, concatenated."""
    feats = []
    for g in to_gray(imgs):
        code = local_binary_pattern(g.astype(np.uint8), P, R, method="nri_uniform")
        h, w = code.shape
        hs = []
        for i in range(grid):
            for j in range(grid):
                cell = code[i * h // grid:(i + 1) * h // grid, j * w // grid:(j + 1) * w // grid]
                hist, _ = np.histogram(cell, bins=59, range=(0, 59))
                hs.append(hist / max(1, cell.size))
        feats.append(np.concatenate(hs))
    return np.asarray(feats, np.float32)


def chi2_similarity(a, b):
    """Negative chi-square distance (higher = more similar), row-wise."""
    return -0.5 * (((a - b) ** 2) / (a + b + 1e-10)).sum(1)
