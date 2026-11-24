"""Train-time degradation augmentation to close the VGGFace2 -> LFW quality/alignment gap.

The VGGFace2 crops are sharp, high-resolution faces; LFW faces are ~100 px, blurrier, JPEG-compressed
and sit slightly differently in the aligned frame. Each transform below fires independently with its
own probability; the fixed face mask is re-applied at the end so the masked region is identical to
what the model sees at test time.
"""
from __future__ import annotations

import cv2
import numpy as np

FILL = 255


class Degrade:
    def __init__(self, mask: np.ndarray, p_geom=0.7, scale=(0.90, 1.10), shift=4.0, rot=4.0,
                 p_down=0.5, down=(0.3, 0.9), p_blur=0.2, sigma=(0.5, 1.5),
                 p_jpeg=0.3, quality=(30, 90), p_color=0.3, color=0.2):
        self.out = ~mask.astype(bool)
        self.p_geom, self.scale, self.shift, self.rot = p_geom, scale, shift, rot
        self.p_down, self.down = p_down, down
        self.p_blur, self.sigma = p_blur, sigma
        self.p_jpeg, self.quality = p_jpeg, quality
        self.p_color, self.color = p_color, color

    def __call__(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        h, w = img.shape[:2]
        if rng.random() < self.p_geom:
            M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-self.rot, self.rot), rng.uniform(*self.scale))
            M[:, 2] += rng.uniform(-self.shift, self.shift, 2)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(FILL, FILL, FILL))
        if rng.random() < self.p_down:
            f = rng.uniform(*self.down)
            small = cv2.resize(img, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA)
            img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
        if rng.random() < self.p_blur:
            img = cv2.GaussianBlur(img, (0, 0), rng.uniform(*self.sigma))
        if rng.random() < self.p_color:
            a = 1 + rng.uniform(-self.color, self.color)          # contrast
            b = 255 * rng.uniform(-self.color, self.color) / 2     # brightness
            img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        if rng.random() < self.p_jpeg:
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(*self.quality))])
            if ok:
                img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        img = np.ascontiguousarray(img)
        img[self.out] = FILL
        return img
