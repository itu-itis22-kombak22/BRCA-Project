# brca-demo — Breast Histopathology Patch Classifier + WSI Heatmap Demo

Graduation project demo: train a binary tumor / non-tumor patch classifier
on **PCam** (PatchCamelyon, Kaggle Histopathologic Cancer Detection), then
apply it over a **TCGA-BRCA** Whole Slide Image to generate a qualitative
tumor-probability heatmap.

PCam is used for training because it provides patch-level ground truth;
TCGA-BRCA is used only as a qualitative demo target (all TCGA-BRCA slides
are from cancer patients, so slide-level binary labels are meaningless).

## Pipeline

1. **Training (Colab, T4 GPU):** `colab/train_pcam.ipynb` fine-tunes a
   pretrained ResNet-18 on PCam for ~2 epochs (BCEWithLogitsLoss, AdamW,
   AMP). Target val AUC ≈ 0.94–0.96. Output: `resnet18_pcam.pth`.
2. **Inference (MacBook, MPS):** `src/inference.py` loads the weights,
   predicts tumor probability for individual patches or batches.
3. **Heatmap (MacBook, MPS):** `src/heatmap.py` slides 96×96 windows over
   a TCGA-BRCA `.svs`, batches them through the classifier, and overlays
   probabilities on a slide thumbnail.

## Repo layout

```
brca-demo/
├── README.md
├── requirements.txt
├── .gitignore
├── colab/
│   └── train_pcam.ipynb       # Phase 1 — Colab training
├── src/
│   ├── __init__.py
│   ├── inference.py           # Phase 2 — patch prediction
│   └── heatmap.py             # Phase 2 — WSI sliding window + overlay
├── models/                    # gitignored — put resnet18_pcam.pth here
├── data/                      # gitignored — .svs files
└── outputs/                   # gitignored — heatmap PNGs
```

## Setup (MacBook)

```bash
brew install openslide

cd ~/brca-demo
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Sanity check
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
python -c "import openslide; print('openslide OK')"
```

## Phase 1 — Training on Colab

Two trainers are provided. Both live under `colab/`, both target the
free T4 GPU, both produce a 43 MB ResNet-18 checkpoint in the same
architecture so any of them is a drop-in replacement in the inference
pipeline. Train either or both — the UI has a model selector.

### Option A — `train_pcam.ipynb` (PatchCamelyon, lymph-node metastasis)

1. Accept the Kaggle competition rules once at
   <https://www.kaggle.com/c/histopathologic-cancer-detection/rules>.
2. Open the notebook in Colab, set `Runtime → T4 GPU`, run 4 cells.
3. ~15–20 min training, target val AUC ≈ 0.97.
4. Download `resnet18_pcam.pth` into `models/`.

### Option B — `train_breakhis.ipynb` (BreaKHis, primary breast tissue)

1. No rule-accept step; it's a public Kaggle dataset.
2. Open the notebook in Colab, set `Runtime → T4 GPU`, run 4 cells.
3. ~30–45 min training on the 40× tier, target val AUC ≈ 0.90.
4. Download `resnet18_breakhis.pth` into `models/`.

BreaKHis is **domain-matched** to primary breast slides (the TCGA-BRCA
use case); PCam is trained on lymph-node metastases. On a primary-tumour
slide the BreaKHis checkpoint generally yields much higher absolute
probabilities, while PCam's heatmap gives spatial structure only.

## Phase 2 — Heatmap demo (MacBook)

After `models/resnet18_pcam.pth` is in place:

```bash
python -m src.heatmap --slide data/<your_slide>.svs --out outputs/
```

### Streamlit UI (drag & drop)

```bash
streamlit run src/app.py
```

The app accepts PNG / JPG / TIF / ZIP (of images) / PDF and returns a
heatmap overlay plus a short Turkish 2–3 sentence summary.

## Notes

- **Magnification matters.** PCam patches are ≈0.972 µm/px (~10×); TCGA-BRCA
  slides are typically 0.25 µm/px (40×). The heatmap code must resample
  patches to PCam's scale, otherwise predictions are meaningless.
- **No large files in git.** `.svs`, `.pth`, `.zip` stay out of the repo.
