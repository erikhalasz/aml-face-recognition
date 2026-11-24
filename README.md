# Face Recognition with Deep Learning — ResNet + ArcFace, VGGFace2 → LFW

Implementation of the project plan in `Face_Recognition_presentation_EN (1).pdf`:
a 512-d face embedding (IR-ResNet-18/50) trained with the ArcFace loss on VGGFace2, used for
**verification** (cosine + threshold, LFW 6,000 pairs, 10-fold) and **open-set identification**
(LFW gallery: 3 enrolled images per person, rest probes, people with one image as unknowns).

**Not in the repo:** the datasets (VGGFace2 subset and LFW; `03_prepare_lfw.py` downloads LFW) and the model
weights. The trained `best.pt` checkpoints for all four runs are attached to the GitHub release `v1.0`. Put each
one in `outputs/<run>/best.pt` to evaluate it with `scripts/evaluate.py --run outputs/<run>`.

## Pipeline (Windows / PowerShell, run from this folder)

```powershell
./scripts/00_setup.ps1                                   # venv + PyTorch (CUDA 12.1) + deps
$py = ".\.venv\Scripts\python.exe"
& $py scripts/01_extract_index.py                        # unzip, Name_Gender -> Class_ID, train/dev split
& $py scripts/02_preprocess_vgg.py                       # MTCNN -> 5 landmarks -> 112x112, quality filter, mask (~2 h)
& $py scripts/03_prepare_lfw.py                          # download LFW + pairs.txt, same preprocessing
& $py scripts/04_leakage_check.py                        # remove VGGFace2 identities that are LFW people
& $py scripts/evaluate.py --pretrained facenet           # sanity check of the evaluation code (reference only)
& $py scripts/evaluate.py --classical eigenfaces         # classical lower bounds
& $py scripts/evaluate.py --classical lbp
& $py scripts/train.py --config configs/r18_arcface.yaml # main model (resumable: rerun the same command)
& $py scripts/train.py --config configs/r18_softmax.yaml # baseline + Ablation 1
& $py scripts/train.py --config configs/r18_arcface_croponly.yaml   # Ablation 2
& $py scripts/train.py --config configs/r18_arcface_aug.yaml     # FINAL model: degradation aug, 14 epochs
& $py scripts/evaluate.py --run outputs/r18_arcface      # (same for each run)
& $py scripts/04_leakage_check.py --run outputs/r18_arcface_aug      # embedding near-duplicate scan
& $py scripts/make_report.py; & $py -m jupyter nbconvert --to notebook --execute --inplace notebooks/report.ipynb
```

## Results (LFW)
| Model | LFW acc (10-fold) | EER | TAR@FAR=0.1% | Rank-1 | FAR / FRR @ τ (from dev) |
|---|---|---|---|---|---|
| **ResNet-18 + ArcFace + degradation aug, 14 ep (final)** | **97.87 ± 0.40%** | **2.37%** | 90.5% | **98.5%** | 15.5% / 3.1% |
| ResNet-18 + ArcFace, 8 ep | 94.52 ± 1.33% | 5.65% | 61.2% | 94.6% | 63.0% / 6.3% |
| ResNet-18 + softmax, 8 ep (Ablation 1) | 94.78 ± 1.06% | 5.10% | 76.2% | 94.9% | 41.6% / 6.9% |
| ResNet-18 + ArcFace crop-only, 8 ep (Ablation 2) | 92.95 ± 1.71% | 7.13% | 62.9% | 89.7% | 43.6% / 13.0% |
| Eigenfaces | 70.37 ± 2.17% | 29.9% | 6.9% | 25.7% | – |
| LBP | 67.53 ± 2.36% | 32.6% | 11.4% | – | – |
| Pretrained FaceNet (pipeline check only) | 97.67 ± 0.75% | 2.33% | 88.2% | 96.9% | 5.9% / 11.6% |

Targets: accuracy ≥ 97% ✓, EER ≤ 3% ✓, Rank-1 ≥ 90% ✓, FRR ≤ 10% ✓, FAR ≤ 1% ✗ (a threshold chosen on VGGFace2 dev
identities does not transfer to LFW; see the report). Hardware: Quadro T2000 4 GB, about 47 min per R18 epoch.

Leakage: the embedding scan (`outputs/leakage_scan.csv`) found 21 LFW people who are in the VGGFace2 train set under a
different name spelling, which the name filter missed. They appear in 53 of the 6,000 pairs. Without those pairs the
final model scores 97.85 ± 0.40% (`outputs/leakage_clean_eval.json`), so the effect is negligible.

## Layout
| Path | Content |
|---|---|
| `facerec/align.py` | MTCNN, 5-point similarity alignment, crop-only, quality stats |
| `facerec/preprocess.py` | batched detection/alignment pass, fixed face mask |
| `facerec/models/iresnet.py` | IR-ResNet-18/50 (ArcFace style), 512-d embedding |
| `facerec/augment.py` | train-time degradation augmentation (blur, downscale, JPEG, colour, scale/shift) |
| `facerec/heads.py` | ArcFace (s=64, m=0.5) and plain softmax heads |
| `facerec/train.py` | SGD + warmup/cosine, AMP, dev-set early stopping, resume |
| `facerec/verify.py` | LFW 10-fold accuracy, ROC-AUC, EER, TAR@FAR |
| `facerec/identify.py` | open-set gallery: Rank-1, FAR/FRR at threshold, DET |
| `facerec/classical.py` | Eigenfaces, LBP histograms |
| `notebooks/report.ipynb` | all tables and figures for the report |

## Notes on the data (deviations from the plan)
* `VGGFace2.zip` is a subset of the VGGFace2 *train* split: 483,919 images, 7,382 identities. It contains none of
  the official test identities, so **dev = 500 randomly held-out identities** (identity-disjoint).
* The images are pre-aligned, tight face crops with a **white mask outside the face**. To remove the
  resulting train/test domain gap, a fixed face mask (face in ≥80% of VGGFace2 train images) is applied to
  all images, train and LFW alike. The alignment template is zoomed ×1.2 for the same reason.
* Leakage controls: VGGFace2 identities whose name matches an LFW person are removed; an embedding scan lists
  any remaining look-alike identities; LFW thresholds are fitted on 9 folds and tested on the 10th; the
  identification threshold is chosen on VGGFace2 dev identities; nothing is tuned on LFW.
