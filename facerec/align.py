"""Face detection (MTCNN), 5-point similarity alignment to the ArcFace 112x112 template,
quality metrics and the crop-only variant used by Ablation 2."""
from __future__ import annotations

import cv2
import numpy as np

# Standard ArcFace / InsightFace 5-point template for 112x112 (left eye, right eye, nose, mouth L, mouth R)
ARCFACE_TEMPLATE = np.array(
    [[38.2946, 51.6963],
     [73.5318, 51.5014],
     [56.0252, 71.7366],
     [41.5493, 92.3655],
     [70.7299, 92.2041]], dtype=np.float32)
# The VGGFace2 subset is masked to the face only (no hair/background), so the standard template
# would waste ~70% of the 112x112 frame. We zoom it by 1.2 around (56, 60).
TEMPLATE_ZOOM = 1.2
ARCFACE_TEMPLATE = (ARCFACE_TEMPLATE - np.float32([56, 60])) * TEMPLATE_ZOOM + np.float32([56, 60])


FILL = 255  # background colour: VGGFace2 crops are masked with white outside the face


def similarity_warp(img: np.ndarray, landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    """Warp an RGB uint8 image so that its 5 landmarks land on the ArcFace template."""
    M, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), ARCFACE_TEMPLATE, method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(img, M, (size, size), borderMode=cv2.BORDER_CONSTANT, borderValue=(FILL,) * 3)


def bbox_crop(img: np.ndarray, box, size: int = 112, scale: float = 1.0) -> np.ndarray:
    """Crop-only baseline (no landmarks): square crop around the detector box, resized."""
    x1, y1, x2, y2 = box
    cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * scale
    M = np.float32([[size / s, 0, size / 2 - cx * size / s], [0, size / s, size / 2 - cy * size / s]])
    return cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_AREA,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(FILL,) * 3)


def center_crop(img: np.ndarray, frac: float = 0.7, size: int = 112) -> np.ndarray:
    """Crop-only baseline: central square crop (no landmarks), resized to size x size."""
    h, w = img.shape[:2]
    c = int(round(min(h, w) * frac))
    y0, x0 = (h - c) // 2, (w - c) // 2
    return cv2.resize(img[y0:y0 + c, x0:x0 + c], (size, size), interpolation=cv2.INTER_AREA)


def quality_stats(face: np.ndarray) -> tuple[float, float]:
    """(blur = variance of Laplacian, brightness = mean gray) of the inner region of an aligned RGB face."""
    g = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)[24:96, 24:88]  # inner face region only (no border)
    return float(cv2.Laplacian(g, cv2.CV_64F).var()), float(g.mean())


class Aligner:
    """Batched MTCNN on the GPU. For each image keeps the face closest to the centre
    (weighted by confidence), since VGGFace2/LFW crops are centred on the subject."""

    def __init__(self, device: str = "cuda", min_face: int = 40):
        from facenet_pytorch import MTCNN
        self.mtcnn = MTCNN(keep_all=True, device=device, min_face_size=min_face,
                           post_process=False, select_largest=False)

    def detect(self, imgs: list[np.ndarray]):
        """imgs: list of equally-sized RGB uint8 arrays. Returns list of (prob, landmarks[5,2], box[4]) or None."""
        boxes, probs, lms = self.mtcnn.detect(np.stack(imgs), landmarks=True)
        out = []
        for img, b, p, l in zip(imgs, boxes, probs, lms):
            if b is None or len(b) == 0:
                out.append(None)
                continue
            b = np.asarray(b, dtype=np.float64)
            p = np.asarray(p, dtype=np.float64)
            l = np.asarray(l, dtype=np.float64)
            h, w = img.shape[:2]
            cx = (b[:, 0] + b[:, 2]) / 2
            cy = (b[:, 1] + b[:, 3]) / 2
            dist = np.hypot(cx - w / 2, cy - h / 2) / max(h, w)
            score = np.asarray(p, dtype=np.float64) - 2.0 * dist
            i = int(np.argmax(score))
            out.append((float(p[i]), l[i].astype(np.float32), b[i].astype(np.float32)))
        return out
