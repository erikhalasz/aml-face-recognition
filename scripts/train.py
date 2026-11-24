"""Train one model: python scripts/train.py --config configs/r18_arcface.yaml [--epochs N ...]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from facerec.config import load_config  # noqa: E402
from facerec.train import run  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--m", type=float)
    ap.add_argument("--workers", type=int)
    ap.add_argument("--limit-classes", type=int)
    ap.add_argument("--max-steps-per-epoch", type=int)
    ap.add_argument("--dev-pairs", type=int)
    a = vars(ap.parse_args())
    cfg = load_config(a.pop("config"), **a)
    run(cfg)
