# SeaSaw — Space Environment Adaptive Solar Activity Warning using MACHINE LEARNING

AI/ML space weather forecasting system built for **ISRO Hackathon Problem Statement 14**.
Predicts energetic electron radiation (**>2 MeV flux**) at Geostationary Earth Orbit (GEO) to
protect ISRO satellites, at three forecast horizons: **30–45 min**, **6 hours**, and **12 hours**.

---

## Project Status

| Phase | Description | Status |
|-------|--------------|--------|
| 1 | Data Ingestion (GOES + Wind + GRASP) | ✅ Complete |
| 2 | Preprocessing (spikes, gaps, log transform, scaling) | ✅ Complete |
| 3 | Feature Engineering (Dynamic Lag, lag/rolling/delta features) | ✅ Complete |
| 4 | Multi-Horizon Dataset Builder (3 chronological train/val/test splits) | ✅ Complete |
| 5 | Model Training (XGBoost + LSTM per horizon) | ✅ Complete |
| 6 | Weighted Ensemble (α-blend of XGBoost + LSTM) | ⬜ Not started |
| 7 | Validation against GRASP (Indian-longitude ground truth) | ⬜ Not started |
| 8 | Streamlit Dashboard | ⬜ Not started |

Phases 1–5 are implemented, tested against real downloaded data where possible, and ready to
run on a full multi-year dataset. Phases 6–8 are next.

---

## Pipeline

```
GOES CDF  +  Wind CDF  +  GRASP ZIP
        │
        ▼
  Phase 1  Data Ingestion            run_phase1_ingestion.py
        │  → data/processed/training_raw.csv, grasp_validation.csv
        ▼
  Phase 2  Preprocessing             run_phase2_preprocessing.py
        │  → data/processed/training_preprocessed.csv, models/scalers.pkl
        ▼
  Phase 3  Feature Engineering       run_phase3_features.py
        │  → data/processed/training_features.csv
        ▼
  Phase 4  Multi-Horizon Dataset     run_phase4_dataset_builder.py
        │  → data/processed/dataset_{A_45min,B_6hr,C_12hr}/*.npy
        ▼
  Phase 5  XGBoost + LSTM Training   run_phase5_training.py
        │  → models/xgb_horizon_{A,B,C}.pkl, models/lstm_horizon_{A,B,C}.h5
        ▼
  Phase 6  Weighted Ensemble         (not yet built)
        ▼
  Phase 7  Validation vs GRASP       (not yet built)
        ▼
  Phase 8  Streamlit Dashboard       (not yet built)
```

---

## Setup

```bash
pip install -r requirements.txt
```

Requires: `cdflib`, `pandas`, `numpy`, `scipy`, `scikit-learn`, `xgboost`, `tensorflow`,
`matplotlib`, `plotly`, `streamlit`. For automated data fetching, also:
`pip install cdasws requests tqdm`.

### Getting the data

**Automated (Wind + GOES):**
```bash
python -m src.ingestion.auto_fetcher --start 2013-01-01 --end 2024-01-01
```
Downloads Wind MFI, Wind SWE, and GOES-13/15/16 CDF files directly from NASA CDAWeb / NOAA.
See `TRAINING_GUIDE.md` for the full walkthrough, including 1-year smoke-test recommendations
before committing to an 11-year fetch.

**Manual (GRASP/GSAT — ISRO PRADAN):**
Register at [pradan.issdc.gov.in](https://pradan.issdc.gov.in/), download ZIPs for
Data → GSAT → date range, and place them (un-extracted) in `data/raw/grasp/`.

Before running Phase 1, inspect at least one file from each source:
```bash
python -m src.ingestion.cdf_inspector data/raw/goes/<file>.cdf
python -m src.ingestion.cdf_inspector data/raw/wind_mfi/<file>.cdf
python -m src.ingestion.cdf_inspector data/raw/wind_swe/<file>.cdf
```

### Running the pipeline

```bash
python run_phase1_ingestion.py
python run_phase2_preprocessing.py
python run_phase3_features.py
python run_phase4_dataset_builder.py
python run_phase5_training.py                    # trains XGBoost + LSTM, all 3 horizons

# Or run pieces individually while iterating:
python run_phase5_training.py --xgb-only --horizon A
python run_phase5_training.py --lstm-only --epochs 30 --sequence-length 288
```

XGBoost trains fine on a laptop CPU (~10–30 min for all three horizons). LSTM should be
trained on a GPU — see `TRAINING_GUIDE.md` for free Colab/Kaggle options; the CPU path still
works but is only practical for small smoke tests, not the full multi-year dataset.

---

## Key Engineering Decisions

**Log transform (mandatory, no exceptions).** Electron flux spans ~5 orders of magnitude.
Every model sees `log10(flux + 1e-10)`, never raw flux. Predictions are inverse-transformed
back (`10**x`) before being reported.

**Dynamic Lag (signature innovation).** Solar wind doesn't take a fixed time to travel from
the Wind spacecraft (L1) to Earth — it depends on solar wind speed at that moment
(`Δt = ΔX / Vsw`, `ΔX ≈ 1.5×10⁶ km`). Phase 3 computes this per-row and looks *backward* that
many steps for the aligned Wind features — a first-order approximation (using the
contemporaneous, arrival-side speed rather than the true departure-side speed), chosen
deliberately because it's a well-defined, gap-free, and deployable operation. An earlier
forward-scatter design (shifting each Wind reading forward to its estimated arrival time) was
tried and rejected: variable per-row shifts aren't surjective, so it left scattered
destination gaps and dropped ~60% of rows once combined with the later `dropna()`.

**Direct single-output LSTM, one per horizon.** No recursive multi-step prediction, no
seq2seq — each LSTM takes a fixed 288-step (24h) input window and predicts a single scalar
(`log10(flux)`) at `t + horizon`. Recursive prediction was rejected because error compounds
badly over a 6h/12h rollout.

**Chronological splits everywhere.** Never random shuffle for time series — Phase 4's
80/10/10 train/val/test split is a straight chronological cut, preserving temporal order so
validation never leaks future information into training.

**Spike detection uses MAD, not rolling std.** A single extreme value inside its own rolling
window inflates plain `std()` enough to hide itself from a `>5σ` threshold test. Phase 2 uses
median absolute deviation (rescaled by 1.4826 to a normal-consistent sigma) instead, which
stays robust to the very outlier it's trying to detect.

---

## Data Sources

| Source | Format | Purpose |
|--------|--------|---------|
| GOES satellite (13/15/16) | CDF | Training target: >2 MeV electron flux |
| Wind spacecraft MFI | CDF | Input: IMF Bx, By, Bz (nT, GSE frame) |
| Wind spacecraft SWE | CDF | Input: solar wind speed (km/s), plasma density (cm⁻³) |
| GRASP/GSAT (ISRO) | ZIP → TXT + XML + PNG | Validation only, not used in training |

**Verified CDAWeb dataset IDs** (confirmed live, not from docs — CDAWeb's own naming
conventions have drifted from what's commonly referenced):
- Wind MFI: `WI_H2_MFI`, variable `BGSE`
- Wind SWE: `WI_K0_SWE`, variables `Proton_V_nonlin`, `Proton_Np_nonlin`
- GOES-13 >2 MeV electrons: `GOES13_EPEAD-SCIENCE-ELECTRONS-E13EW_1MIN`, variable `E2W_COR_FLUX`
  (2010-05 to 2017-12)
- GOES-15 >2 MeV electrons: `GOES15_EPEAD-SCIENCE-ELECTRONS-E13EW_1MIN`, variable `E2W_COR_FLUX`
  (2010-03 to 2020-03)

Note: the GOES EPEAD >2 MeV corrected channel has a naturally high NaN rate — solar proton
contamination frequently invalidates it (`E2W_DQF` flag). That's expected data behavior, not
a pipeline bug.

---

## Project Structure

```
seasaw/
├── requirements.txt
├── TRAINING_GUIDE.md                 ← full walkthrough: fetch → train → GPU options → dashboard
├── run_phase1_ingestion.py
├── run_phase2_preprocessing.py
├── run_phase3_features.py
├── run_phase4_dataset_builder.py
├── run_phase5_training.py
├── data/
│   ├── raw/{goes,wind_mfi,wind_swe,grasp}/    ← place downloaded files here
│   ├── processed/                              ← pipeline outputs (gitignored, regenerated)
│   └── validation/
├── src/
│   ├── ingestion/          Phase 1 — cdf_inspector, goes_reader, wind_reader, grasp_reader,
│   │                       data_pipeline, auto_fetcher (automated NASA/NOAA download)
│   ├── preprocessing/      Phase 2 — preprocessor.py
│   ├── features/           Phase 3 — feature_engineer.py
│   └── models/             Phase 5 — xgb_trainer.py, lstm_trainer.py
└── models/                 saved model artifacts (.pkl, .h5) — gitignored, regenerated
```

---

## Known Gotchas (found while building this)

These were live, verified bugs fixed during development — worth knowing if you touch the
ingestion or model-saving code:

- **`cdflib` >= 1.0 returns `cdf_info()` as a `CDFInfo` dataclass, not a dict.** Code written
  against the older dict-style API (`info.get("rVariables", [])`) crashes with
  `AttributeError` on every real CDF file. Fixed in `cdf_inspector.py`, `goes_reader.py`,
  `wind_reader.py` to use attribute access (`info.rVariables`).
- **`cdasws`'s `get_data()` returns in-memory data, not file URLs.** `auto_fetcher.py` now
  uses `get_data_file()`, which returns `(status_code, {"FileDescription": [...]})`.
  `get_datasets()`/`get_variables()` also return lists directly, not dict-wrapped results.
- **Keras 3 can't reliably reload a legacy `.h5` model compiled with a string loss.** Loading
  raises `Could not deserialize 'keras.metrics.mse'`. `LSTMTrainer.load()` works around this
  with `compile=False` (fine for `predict()`; pass `recompile=True` if you need `.evaluate()`
  or further training).
- Several `→ ✓ —` unicode characters in log/print statements crashed or garbled on Windows'
  cp1252 console encoding — replaced with ASCII equivalents throughout.
