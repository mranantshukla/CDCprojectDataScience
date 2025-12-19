## Multimodal House Price Valuation

This repository implements a **production-style multimodal regression system** to predict real estate prices using:

- **Tabular features** (King County housing data: size, quality, age, location, etc.)
- **Satellite imagery** (Sentinel Hub; top-down environmental and spatial context)

The core scientific goal is to **test whether visual context adds genuine economic signal beyond standard tabular attributes**, not just to chase marginal metric gains.

---

### 1. Repository Structure

- `data_fetcher.py` – Robust Sentinel Hub image fetching pipeline (lat/long → bounding box → saved tile + metadata).
- `predict_test_prices.py` – Complete pipeline for generating price predictions on test.xlsx (downloads images, extracts embeddings, applies trained model).
- `notebooks/preprocessing.ipynb` – Problem framing, causal assumptions, tabular + geospatial EDA, and data splits.
- `notebooks/model_training.ipynb` – Baselines, CNN feature extraction, multimodal fusion, and evaluation.
- `notebooks/explainability.ipynb` – Grad-CAM (or similar) visualizations and economic interpretation of attention.
- `data/` – Expected data directory (not tracked):
  - `data/raw/kc_house_data.csv` – King County housing dataset (user-provided).
  - `data/raw/` – Any additional raw assets.
  - `data/processed/` – Clean tables, train/val/test splits.
  - `data/satellite/` – Downloaded Sentinel imagery tiles.
  - `data/embeddings/` – Cached CNN embeddings and indices.
- `reports/` – Final plots, Grad-CAM images, and a report markdown or notebook to be exported as PDF.
- `requirements.txt` – Python dependencies.

---

### 2. Conceptual Overview

- **Economic intuition**: Tabular data misses neighborhood texture (greenery, density, water, road network, block shape). Satellite images encode these **latent spatial and environmental factors**, which may drive systematic price differences even after conditioning on standard attributes.
- **Causal stance**: We are **not** claiming causal effects of pixels on price. Instead, we treat imagery as a proxy for hard-to-measure neighborhood quality variables. We explicitly check for:
  - **Spatial leakage** (train/test contamination via geography).
  - **Overfitting to imagery noise**.
  - **Stability of performance gains** across validation splits.
- **Modeling strategy**:
  - Use a **pretrained ResNet** as a frozen feature extractor (transfer learning, CPU-friendly).
  - Compare:
    - **Tabular-only baselines** (linear + tree ensembles).
    - **Late fusion** (separate tabular and image models, combine predictions/embeddings).
    - **Feature-level fusion** (concatenate tabular features + CNN embeddings, single regression head).

---

### 3. Data Requirements and Setup

1. **Create a virtual environment** (recommended, Python ≥ 3.9):

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Place the King County dataset**:

- Download the King County housing CSV (commonly named `kc_house_data.csv`) and save it to:
  - `data/raw/kc_house_data.csv`

4. **Configure Sentinel Hub credentials**:

- Create a `.env` file in the project root or set environment variables (recommended):
  - `SENTINELHUB_CLIENT_ID`
  - `SENTINELHUB_CLIENT_SECRET`
  - Optionally, additional config such as:
    - `SENTINELHUB_INSTANCE_ID` (for older setups)
    - Default collection / resolution if needed.

The `data_fetcher.py` module will read these environment variables and handle authentication.

---

### 4. Sentinel Hub Image Fetching (Engineering Considerations)

**Why zoom level, resolution, and tile size matter economically:**

- **Zoom / ground sampling distance (GSD)**:
  - Too coarse (e.g., 60–100 m/pixel): individual parcels blur together; neighborhood structure collapses.
  - Too fine (e.g., sub-meter imagery): redundant detail, heavier downloads, more noise and overfitting risk.
  - We target a **parcel-scale context window** (e.g., 10 m/pixel, ~256×256 px) that captures:
    - Local greenery / tree canopy.
    - Road layout and accessibility.
    - Proximity to water bodies.
    - Density of surrounding buildings.
- **Bounding box size**:
  - Economically, buyers care about **nearby** amenities and disamenities within a few hundred meters (walkable context).
  - We define a fixed-radius bounding box (e.g., 250–500 m) centered on the property coordinates. This encodes **neighborhood quality** rather than just the parcel.
- **Determinism and reproducibility**:
  - Image filenames are deterministic functions of `id` and coordinates.
  - Metadata is stored in a CSV/Parquet file linking each property ID to its image path and bounding box.

**Robustness engineering**:

- Automatic **retry logic** with exponential backoff for transient Sentinel Hub errors.
- Graceful handling of:
  - Missing tiles or cloud cover (configurable filters).
  - API rate limits (throttling and logging).
  - Partial coverage (e.g., edges of acquisition area).

Details are implemented in `data_fetcher.py`.

---

### 5. Workflow

1. **Run `data_fetcher.py`** (or import and call its functions) to download Sentinel images for the King County properties.
2. Open `notebooks/preprocessing.ipynb`:
   - Read tabular data.
   - Perform tabular EDA (distributions, correlations, multicollinearity, monotonicity checks).
   - Perform geospatial EDA (price heatmaps, clustering, spatial leakage checks).
   - Create robust train/validation/test splits that account for geography.
3. Open `notebooks/model_training.ipynb`:
   - Extract **CNN embeddings** from Sentinel tiles using a frozen ResNet.
   - Cache embeddings to `data/embeddings/` for reproducibility.
   - Train baseline tabular models and multimodal fusion models.
   - Evaluate with **RMSE**, **R²**, and **% improvement over tabular-only baseline**.
4. Open `notebooks/explainability.ipynb`:
   - Run Grad-CAM (or similar) on the image model.
   - Inspect where the model focuses for **high vs. low priced** houses.
   - Reject models whose attention is spatially random or economically uninterpretable.
5. **Generate test predictions**:
   - Place `test.xlsx` in `Provided/` directory (or `data/raw/` or project root).
   - Run the prediction script:
     ```bash
     python predict_test_prices.py
     ```
   - This script will:
     - Load test data from `test.xlsx`
     - Download satellite images for test properties (saved to `data/satellite_test/`)
     - Extract image embeddings using ResNet18
     - Train/load the multimodal model (late fusion)
     - Generate predictions for all test properties
     - Save predictions to `reports/predictions_test_final.csv` in format: `id, actual_price, predicted_price`
6. **Reporting and final PDF**:
   - Save key plots, tables, and Grad-CAM figures into `reports/`.
   - In a final summary notebook or markdown file, assemble:
     - EDA visuals and key descriptive statistics.
     - Model comparison table (tabular vs image vs fusion).
     - Representative Grad-CAM overlays and discussion.
     - Honest failure analysis and limitations.
   - Use Jupyter or `nbconvert` to export that notebook/markdown to a **PDF report** suitable for stakeholders.

---

### 6. Interpreting Results as an Economist

Throughout the notebooks, we explicitly translate model behavior into **financial and business insights**, such as:

- How much **green coverage** or **water proximity** appears to raise prices, conditional on size and quality.
- Whether **visual density** proxies for lot size in urban cores.
- Whether satellite-derived signals **truly add information** beyond tabular features, as evidenced by:
  - Stable performance gains across spatial cross-validation folds.
  - Grad-CAM maps focusing on economically meaningful regions (e.g., waterfront, parks, arterial roads).

If imagery only yields marginal or unstable improvements, the conclusion will explicitly reflect that—**no black-box optimism**.

---

### 7. Reproducibility Notes

- All random processes (splits, model seeds, bootstrapping) should be seeded in the notebooks.
- Embeddings and splits are cached to disk to avoid recomputation and ensure reproducible experiments.
- The code is written assuming **CPU-only** environments; models and batch sizes are tuned accordingly.

---

**Submitted by:**
- Anant Shukla
- 23113024
- IIT Roorkee
