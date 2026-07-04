# SeaSaw — Training Guide

## Overview

11 years of GOES + Wind data at 5-min resolution ≈ **1.15 million rows**.
This guide tells you exactly what to run, in what order, on what hardware.

---

## Step 0: Fetch the Data

### Automated (Wind + GOES)

```bash
pip install cdasws requests tqdm

# First, discover exact dataset IDs for your date range
python -m src.ingestion.auto_fetcher --list-datasets

# Full fetch: 11 years (will take several hours, runs in background)
python -m src.ingestion.auto_fetcher --start 2013-01-01 --end 2024-01-01

# If only Wind data (to test pipeline first)
python -m src.ingestion.auto_fetcher --start 2013-01-01 --end 2014-01-01 --wind-only
```

### GRASP (Manual — ISRO PRADAN)
1. Register at https://pradan.issdc.gov.in/
2. Login → GSAT/GRASP → Select 1-2 years of data → Download ZIPs
3. Place all ZIPs in `data/raw/grasp/` — DO NOT extract them
4. The reader handles bulk extraction automatically

### What gets downloaded where:
```
data/raw/
├── goes/        ← GOES-13, 15, 16 CDF files
├── wind_mfi/    ← Wind MFI monthly CDF files (1-min IMF)
├── wind_swe/    ← Wind SWE monthly CDF files (proton speed/density)
└── grasp/       ← GRASP ZIPs (manual download from PRADAN)
```

---

## Step 1: Inspect Before Running

Before Phase 1, run the inspector on ONE file from each source.
This confirms auto-detection will work for your specific files.

```bash
# Pick any CDF file from each folder and inspect it
python -m src.ingestion.cdf_inspector data/raw/goes/some_file.cdf
python -m src.ingestion.cdf_inspector data/raw/wind_mfi/some_file.cdf
python -m src.ingestion.cdf_inspector data/raw/wind_swe/some_file.cdf
```

Look for:
- The **epoch/time variable** (almost always `Epoch`)
- The **>2 MeV electron flux variable** (for GOES — note the exact name)
- The **BGSE or B_GSE variable** (for Wind MFI)
- The **Proton_V_nonlin or Vp variable** (for Wind SWE)

If auto-detection fails, pass the correct names explicitly in `run_phase1_ingestion.py`.

---

## Step 2: Smoke Test on 1 Year First

DO NOT run 11 years immediately.
Start with 1 year to confirm the full pipeline works end-to-end.

```bash
# Edit run_phase1_ingestion.py to temporarily point at 1-year data
# Or just put 1 year of files in the data/raw/ folders first

python run_phase1_ingestion.py   # → data/processed/training_raw.csv
python run_phase2_preprocessing.py
python run_phase3_features.py
python run_phase4_dataset_builder.py

# Quick XGBoost train to verify shapes
python run_phase5_training.py --xgb-only --horizon A
```

If this works, scale to 11 years.

---

## Step 3: Full Pipeline (11 Years)

Run phases sequentially. Each produces files the next phase reads.

```bash
python run_phase1_ingestion.py        # Hours depending on file count
python run_phase2_preprocessing.py    # ~5–15 min on 1M rows
python run_phase3_features.py         # ~10–30 min (dynamic lag is row-by-row)
python run_phase4_dataset_builder.py  # ~5 min
```

After Phase 4 you have:
```
data/processed/
├── dataset_A_45min/   X_train.npy, X_val.npy, X_test.npy, y_train.npy, ...
├── dataset_B_6hr/
└── dataset_C_12hr/
```

---

## Step 4: Train XGBoost (CPU — Your Laptop is Fine)

XGBoost is fast. 1M rows trains in minutes.

```bash
python run_phase5_training.py --xgb-only

# Trains 3 models: xgb_horizon_A.pkl, xgb_horizon_B.pkl, xgb_horizon_C.pkl
# Total time: 10–30 minutes on a modern laptop
```

Review validation MAE/RMSE output before proceeding to LSTM.
XGBoost gives a strong baseline. If its scores are already good, LSTM will improve them further.

---

## Step 5: Train LSTM (USE GPU — Do Not Train on CPU)

### Why GPU is required:
- Sequence length 288 × ~900K samples = massive computation
- CPU estimate: 3–7 days per horizon (impractical)
- GPU estimate: 1–3 hours per horizon (practical)

### Free GPU options:

**Option A — Google Colab (recommended, free T4 GPU)**
```
1. Upload data/processed/dataset_A_45min/ to Google Drive
2. Open Colab: colab.research.google.com
3. Runtime → Change runtime type → T4 GPU
4. Mount Drive, run the training script
5. Download models/ folder back
```

**Option B — Kaggle Notebooks (free P100 GPU, 30h/week)**
```
1. Upload .npy files as a Kaggle Dataset
2. Create a new notebook, enable GPU accelerator
3. Run training script
4. Download output models
```

**Option C — Your own NVIDIA GPU**
```bash
pip install tensorflow[gpu]   # or tensorflow-gpu for older versions
python run_phase5_training.py --lstm-only
```

### Colab setup snippet (paste at top of Colab notebook):
```python
from google.colab import drive
drive.mount('/content/drive')

import subprocess
subprocess.run(['pip', 'install', 'tensorflow', 'xgboost'], check=True)

import sys
sys.path.insert(0, '/content/drive/MyDrive/seasaw')

from src.models.lstm_trainer import LSTMTrainer
import numpy as np

# Load data
X_train = np.load('/content/drive/MyDrive/seasaw/data/processed/dataset_A_45min/X_train.npy')
y_train = np.load('/content/drive/MyDrive/seasaw/data/processed/dataset_A_45min/y_train.npy')
X_val   = np.load('/content/drive/MyDrive/seasaw/data/processed/dataset_A_45min/X_val.npy')
y_val   = np.load('/content/drive/MyDrive/seasaw/data/processed/dataset_A_45min/y_val.npy')

trainer = LSTMTrainer(horizon='A')
trainer.train(X_train, y_train, X_val, y_val)
# Save model to Drive: /content/drive/MyDrive/seasaw/models/lstm_horizon_A.h5
```

---

## Step 6: Ensemble + Validation

Once both XGBoost (.pkl) and LSTM (.h5) models are saved locally:

```bash
python run_phase6_ensemble.py      # optimizes α weights, saves ensemble_weights.json
python run_phase7_validation.py    # compares predictions vs GRASP flux
```

---

## Step 7: Dashboard

```bash
streamlit run src/dashboard/app.py
```

---

## Hardware Requirements Summary

| Task | Hardware | Time Estimate |
|------|----------|---------------|
| Phases 1–4 (data + features) | Any laptop, 8 GB RAM | 30 min – 2 hours |
| XGBoost training (3 models) | Any laptop CPU | 10–30 minutes |
| LSTM training (3 models) | GPU required | 3–9 hours total (T4 Colab) |
| Ensemble + validation | Any laptop | < 5 minutes |
| Dashboard | Any laptop | Instant |

---

## What "Good" Looks Like

After training, your validation metrics (on test set, log scale) should be approximately:

| Horizon | Good R² | Acceptable R² |
|---------|---------|---------------|
| 30–45 min | > 0.85 | > 0.70 |
| 6 hours   | > 0.65 | > 0.50 |
| 12 hours  | > 0.50 | > 0.35 |

Persistence baseline (predict "current flux = future flux") typically scores:
- R² ≈ 0.90 for 30-min (very hard to beat at short horizons)
- R² ≈ 0.50 for 6h
- R² ≈ 0.30 for 12h

Your model must beat persistence for the 6h and 12h horizons to be meaningful.

---

## Common Issues and Fixes

**Phase 1 fails with "variable not found"**
→ Run `cdf_inspector.py` on the failing file. Pass the correct name explicitly.

**LSTM loss is NaN from epoch 1**
→ Log transform was not applied. Check `log_electron_flux` column exists in training data.

**LSTM overfits (val_loss rises after few epochs)**
→ Increase Dropout, reduce learning rate, or use more data.

**XGBoost predicts negative flux**
→ XGBoost operates on log scale. Predictions will be log values — inverse transform them.

**GRASP validation shows large offset vs GOES**
→ Expected: GRASP observes Indian longitude, GOES observes ~75°W. Some spatial difference is normal. Validate the trend/shape, not exact values.
