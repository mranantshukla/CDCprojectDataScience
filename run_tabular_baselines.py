from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
PROVIDED_DIR = PROJECT_ROOT / "Provided"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load leakage-aware splits created by preprocessing.ipynb
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

    # Infer key columns
    cols = train_df.columns
    target_col = "log_price" if "log_price" in cols else "price"
    price_col = "price" if "price" in cols else target_col
    id_col = "id" if "id" in cols else "Id" if "Id" in cols else None

    if id_col is None:
        raise ValueError("Could not infer ID column (expected 'id' or 'Id').")

    print(f"TARGET_COL = {target_col}")
    print(f"PRICE_COL  = {price_col}")
    print(f"ID_COL     = {id_col}")

    # ------------------------------------------------------------------
    # 2. Build tabular feature matrices
    # ------------------------------------------------------------------
    y_train = train_df[target_col].values
    X_train = train_df.drop(columns=[target_col])

    y_val = val_df[target_col].values
    X_val = val_df.drop(columns=[target_col])

    # Avoid target leakage: if we model log_price but raw price is present,
    # drop raw price from the feature matrix.
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

    # ------------------------------------------------------------------
    # 3. Train tabular baselines (Ridge and Random Forest)
    # ------------------------------------------------------------------
    models: dict[str, Pipeline] = {}
    metrics: dict[str, dict[str, float]] = {}

    # Ridge regression on (log-)price
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
    metrics["ridge"] = {"rmse": ridge_rmse, "r2": ridge_r2}
    models["ridge"] = ridge_model

    print(f"[Ridge] RMSE (price): {ridge_rmse:,.2f} | R^2: {ridge_r2:.3f}")

    # Random Forest
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
    metrics["rf"] = {"rmse": rf_rmse, "r2": rf_r2}
    models["rf"] = rf_model

    print(f"[RF]    RMSE (price): {rf_rmse:,.2f} | R^2: {rf_r2:.3f}")

    # ------------------------------------------------------------------
    # 4. Pick best tabular model and generate test predictions
    # ------------------------------------------------------------------
    best_name = min(metrics.keys(), key=lambda k: metrics[k]["rmse"])
    best_model = models[best_name]
    print(f"Best tabular model: {best_name} with RMSE={metrics[best_name]['rmse']:,.2f}")

    # Load test data
    test_path = PROVIDED_DIR / "test.xlsx"
    if not test_path.exists():
        raise FileNotFoundError(f"Expected test.xlsx at {test_path}")

    test_df = pd.read_excel(test_path)
    if id_col not in test_df.columns:
        raise ValueError(f"Expected ID column '{id_col}' in test data.")

    # Drop any target columns if present by accident
    drop_cols = [c for c in [target_col, "log_price", "price"] if c in test_df.columns]
    X_test = test_df.drop(columns=drop_cols)

    # Align columns with training features
    for c in numeric_features:
        if c not in X_test.columns:
            X_test[c] = np.nan
    for c in categorical_features:
        if c not in X_test.columns:
            X_test[c] = "missing"

    X_test = X_test[numeric_features + categorical_features]

    # Predict
    test_pred_target = best_model.predict(X_test)
    if target_col == "log_price":
        test_pred_price = np.expm1(test_pred_target)
    else:
        test_pred_price = test_pred_target

    pred_df = pd.DataFrame(
        {
            id_col: test_df[id_col].values,
            "actual_price": np.nan,  # blind test
            "predicted_price": test_pred_price,
        }
    )

    out_csv = REPORTS_DIR / "predictions_test_tabular.csv"
    pred_df.to_csv(out_csv, index=False)
    print(f"Saved tabular test predictions to {out_csv}")

    # ------------------------------------------------------------------
    # 5. Print a compact summary for reporting
    # ------------------------------------------------------------------
    print("\n=== Tabular Baseline Summary ===")
    for name, m in metrics.items():
        print(
            f"{name:>6}: RMSE={m['rmse']:,.2f} | R^2={m['r2']:.3f}",
        )


if __name__ == "__main__":
    main()


