from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load processed splits and CNN embeddings
    # ------------------------------------------------------------------
    train_path = PROCESSED_DIR / "train.parquet"
    val_path = PROCESSED_DIR / "val.parquet"

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(
            f"Expected processed splits at {train_path} and {val_path}. "
            "Run notebooks/preprocessing.ipynb first."
        )

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    print(f"Train split shape: {train_df.shape}")
    print(f"Val split shape:   {val_df.shape}")

    cols = train_df.columns
    target_col = "log_price" if "log_price" in cols else "price"
    price_col = "price" if "price" in cols else target_col
    id_col = "id" if "id" in cols else "Id" if "Id" in cols else None

    if id_col is None:
        raise ValueError("Could not infer ID column (expected 'id' or 'Id').")

    print(f"TARGET_COL = {target_col}")
    print(f"PRICE_COL  = {price_col}")
    print(f"ID_COL     = {id_col}")

    # Embeddings saved by model_training.ipynb
    train_emb_path = EMBEDDINGS_DIR / "resnet18_train_embeddings.parquet"
    val_emb_path = EMBEDDINGS_DIR / "resnet18_val_embeddings.parquet"
    if not train_emb_path.exists() or not val_emb_path.exists():
        raise FileNotFoundError(
            f"Expected ResNet embeddings at {train_emb_path} and {val_emb_path}. "
            "Run notebooks/model_training.ipynb embedding cells first."
        )

    train_img_emb = pd.read_parquet(train_emb_path)
    val_img_emb = pd.read_parquet(val_emb_path)

    print(f"Train image embeddings: {train_img_emb.shape}")
    print(f"Val image embeddings:   {val_img_emb.shape}")

    emb_cols = [c for c in train_img_emb.columns if c.startswith("img_emb_")]
    if not emb_cols:
        raise RuntimeError("No embedding columns found (expected columns starting with 'img_emb_').")

    # ------------------------------------------------------------------
    # 2. Build tabular preprocessing and baselines
    # ------------------------------------------------------------------
    y_train = train_df[target_col].values
    X_train = train_df.drop(columns=[target_col])

    y_val = val_df[target_col].values
    X_val = val_df.drop(columns=[target_col])

    # Avoid target leakage: drop raw price if modeling log_price
    if target_col == "log_price" and price_col in X_train.columns:
        X_train = X_train.drop(columns=[price_col])
        X_val = X_val.drop(columns=[price_col])

    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    if id_col in numeric_features:
        numeric_features.remove(id_col)

    categorical_features = [
        c for c in X_train.columns if c not in numeric_features and c != id_col
    ]

    print(f"Numeric features:     {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Ridge baseline
    ridge_model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )
    ridge_model.fit(X_train, y_train)
    y_val_pred_ridge = ridge_model.predict(X_val)

    if target_col == "log_price" and price_col in val_df.columns:
        y_val_price = val_df[price_col].values
        y_val_price_pred_ridge = np.expm1(y_val_pred_ridge)
    else:
        y_val_price = y_val
        y_val_price_pred_ridge = y_val_pred_ridge

    ridge_rmse = rmse(y_val_price, y_val_price_pred_ridge)
    ridge_r2 = r2_score(y_val_price, y_val_price_pred_ridge)

    print(f"[Ridge] RMSE (price, full val): {ridge_rmse:,.2f} | R^2: {ridge_r2:.3f}")

    # Random Forest baseline
    rf_model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )
    rf_model.fit(X_train, y_train)
    y_val_pred_rf = rf_model.predict(X_val)

    if target_col == "log_price" and price_col in val_df.columns:
        y_val_price_pred_rf = np.expm1(y_val_pred_rf)
    else:
        y_val_price_pred_rf = y_val_pred_rf

    rf_rmse = rmse(y_val_price, y_val_price_pred_rf)
    rf_r2 = r2_score(y_val_price, y_val_price_pred_rf)

    print(f"[RF]    RMSE (price, full val): {rf_rmse:,.2f} | R^2: {rf_r2:.3f}")

    base_rmse_full = rf_rmse
    base_r2_full = rf_r2

    # ------------------------------------------------------------------
    # 3. Restrict to subset with images (for fair multimodal comparison)
    # ------------------------------------------------------------------
    train_with_img = train_df.merge(train_img_emb, on=id_col, how="inner")
    val_with_img = val_df.merge(val_img_emb, on=id_col, how="inner")

    print(f"Train with images: {train_with_img.shape}")
    print(f"Val with images:   {val_with_img.shape}")

    # Subset tabular features for the same IDs
    val_ids_with_img = val_with_img[id_col].unique()
    val_tab_subset = val_df[val_df[id_col].isin(val_ids_with_img)].copy()

    # Recompute baseline RF predictions on this subset (price scale)
    y_val_tab_subset_pred = rf_model.predict(val_tab_subset.drop(columns=[target_col]))
    if target_col == "log_price" and price_col in val_tab_subset.columns:
        y_val_tab_subset_price = val_tab_subset[price_col].values
        y_val_tab_subset_price_pred = np.expm1(y_val_tab_subset_pred)
    else:
        y_val_tab_subset_price = val_tab_subset[target_col].values
        y_val_tab_subset_price_pred = y_val_tab_subset_pred

    base_rmse_subset = rmse(y_val_tab_subset_price, y_val_tab_subset_price_pred)
    base_r2_subset = r2_score(y_val_tab_subset_price, y_val_tab_subset_price_pred)

    print(
        f"[RF]    RMSE (price, val subset with images): {base_rmse_subset:,.2f} | "
        f"R^2: {base_r2_subset:.3f}"
    )

    # ------------------------------------------------------------------
    # 4. Image-only model (embeddings -> Random Forest)
    # ------------------------------------------------------------------
    X_train_img = train_with_img[emb_cols].values
    X_val_img = val_with_img[emb_cols].values

    y_train_img = train_with_img[target_col].values
    y_val_img = val_with_img[target_col].values

    img_rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
    )
    img_rf.fit(X_train_img, y_train_img)
    y_val_pred_img = img_rf.predict(X_val_img)

    if target_col == "log_price" and price_col in val_with_img.columns:
        y_val_price_img = val_with_img[price_col].values
        y_val_price_pred_img = np.expm1(y_val_pred_img)
    else:
        y_val_price_img = y_val_img
        y_val_price_pred_img = y_val_pred_img

    img_rmse = rmse(y_val_price_img, y_val_price_pred_img)
    img_r2 = r2_score(y_val_price_img, y_val_price_pred_img)

    print(f"[IMG RF] RMSE (price, val with images): {img_rmse:,.2f} | R^2: {img_r2:.3f}")

    # ------------------------------------------------------------------
    # 5. Strategy A — Late fusion (tabular RF + image RF)
    # ------------------------------------------------------------------
    # Fit linear combiner on TRAIN subset with images, evaluate on VAL subset with images.
    # This avoids overfitting the combiner directly on the validation set.

    # Training subset with images
    train_tab_with_img = train_df[train_df[id_col].isin(train_with_img[id_col])].copy()
    tab_train_pred = rf_model.predict(train_tab_with_img.drop(columns=[target_col]))
    if target_col == "log_price" and price_col in train_tab_with_img.columns:
        tab_train_price_true = train_tab_with_img[price_col].values
        tab_train_price_pred = np.expm1(tab_train_pred)
    else:
        tab_train_price_true = train_tab_with_img[target_col].values
        tab_train_price_pred = tab_train_pred

    img_train_pred_target = img_rf.predict(
        train_with_img[emb_cols].values
    )  # same row order as train_with_img
    if target_col == "log_price" and price_col in train_with_img.columns:
        img_train_price_pred = np.expm1(img_train_pred_target)
    else:
        img_train_price_pred = img_train_pred_target

    # Align lengths (defensive; should already match)
    n_train_fusion = min(len(tab_train_price_pred), len(img_train_price_pred))
    fusion_train_X = np.column_stack(
        [tab_train_price_pred[:n_train_fusion], img_train_price_pred[:n_train_fusion]]
    )
    fusion_train_y = tab_train_price_true[:n_train_fusion]

    # Validation subset with images
    tab_val_with_img = val_df[val_df[id_col].isin(val_with_img[id_col])].copy()
    tab_val_pred = rf_model.predict(tab_val_with_img.drop(columns=[target_col]))
    if target_col == "log_price" and price_col in tab_val_with_img.columns:
        tab_val_price_true = tab_val_with_img[price_col].values
        tab_val_price_pred = np.expm1(tab_val_pred)
    else:
        tab_val_price_true = tab_val_with_img[target_col].values
        tab_val_price_pred = tab_val_pred

    img_val_pred_target = img_rf.predict(
        val_with_img[emb_cols].values
    )  # same row order as val_with_img
    if target_col == "log_price" and price_col in val_with_img.columns:
        img_val_price_pred = np.expm1(img_val_pred_target)
    else:
        img_val_price_pred = img_val_pred_target

    n_val_fusion = min(len(tab_val_price_pred), len(img_val_price_pred))
    fusion_val_X = np.column_stack(
        [tab_val_price_pred[:n_val_fusion], img_val_price_pred[:n_val_fusion]]
    )
    fusion_val_y = tab_val_price_true[:n_val_fusion]

    fusion_reg = LinearRegression()
    fusion_reg.fit(fusion_train_X, fusion_train_y)

    fusion_val_pred = fusion_reg.predict(fusion_val_X)
    fusion_rmse = rmse(fusion_val_y, fusion_val_pred)
    fusion_r2 = r2_score(fusion_val_y, fusion_val_pred)

    print(
        f"[Late fusion] RMSE (price, val with images): {fusion_rmse:,.2f} | "
        f"R^2: {fusion_r2:.3f}"
    )

    improvement_late = 100.0 * (base_rmse_subset - fusion_rmse) / base_rmse_subset
    print(f"[Late fusion] % RMSE improvement vs tabular RF (subset): {improvement_late:.2f}%")

    # ------------------------------------------------------------------
    # 6. Strategy B — Feature-level fusion (tabular + embeddings)
    # ------------------------------------------------------------------
    # Precompute tabular design matrices
    X_train_tab = preprocessor.fit_transform(X_train)
    X_val_tab = preprocessor.transform(X_val)

    train_idx_map = {idx: i for i, idx in enumerate(train_df[id_col].values)}
    val_idx_map = {idx: i for i, idx in enumerate(val_df[id_col].values)}

    train_ids_with_img = np.intersect1d(train_df[id_col].values, train_img_emb[id_col].values)
    val_ids_with_img = np.intersect1d(val_df[id_col].values, val_img_emb[id_col].values)

    def build_fusion_matrices(ids, df, idx_map, X_tab, emb_df):
        tab_rows = []
        img_rows = []
        targets = []
        prices = []
        for pid in ids:
            tab_idx = idx_map[pid]
            tab_rows.append(X_tab[tab_idx])
            img_rows.append(emb_df.loc[emb_df[id_col] == pid, emb_cols].values[0])
            targets.append(df.loc[df[id_col] == pid, target_col].values[0])
            prices.append(df.loc[df[id_col] == pid, price_col].values[0])
        return (
            np.vstack(tab_rows),
            np.vstack(img_rows),
            np.array(targets),
            np.array(prices),
        )

    X_train_tab_f, X_train_img_f, y_train_f, y_train_price_f = build_fusion_matrices(
        train_ids_with_img, train_df, train_idx_map, X_train_tab, train_img_emb
    )
    X_val_tab_f, X_val_img_f, y_val_f, y_val_price_f = build_fusion_matrices(
        val_ids_with_img, val_df, val_idx_map, X_val_tab, val_img_emb
    )

    X_train_fusion = np.hstack([X_train_tab_f, X_train_img_f])
    X_val_fusion = np.hstack([X_val_tab_f, X_val_img_f])

    print(f"Feature-level fusion matrices: {X_train_fusion.shape}, {X_val_fusion.shape}")

    fusion_rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=123,
    )
    fusion_rf.fit(X_train_fusion, y_train_f)

    fusion_val_pred_target = fusion_rf.predict(X_val_fusion)
    if target_col == "log_price":
        fusion_val_price_pred = np.expm1(fusion_val_pred_target)
    else:
        fusion_val_price_pred = fusion_val_pred_target

    fusion_rmse_feat = rmse(y_val_price_f, fusion_val_price_pred)
    fusion_r2_feat = r2_score(y_val_price_f, fusion_val_price_pred)

    print(
        f"[Feat fusion RF] RMSE (price, val with images): {fusion_rmse_feat:,.2f} | "
        f"R^2: {fusion_r2_feat:.3f}"
    )

    improvement_feat = 100.0 * (base_rmse_subset - fusion_rmse_feat) / base_rmse_subset
    print(
        f"[Feat fusion RF] % RMSE improvement vs tabular RF (subset): "
        f"{improvement_feat:.2f}%"
    )


if __name__ == "__main__":
    main()


