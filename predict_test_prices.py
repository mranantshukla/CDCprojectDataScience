"""
Complete pipeline for generating price predictions on test.xlsx.

This script:
1. Loads test.xlsx
2. Downloads satellite images for test properties (if not already downloaded)
3. Extracts image embeddings using ResNet18
4. Trains/loads the multimodal model (late fusion)
5. Generates predictions for all test properties
6. Saves predictions to CSV in format: id, actual_price, predicted_price
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_fetcher import SatelliteImageFetcher

# Directory setup
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
REPORTS_DIR = PROJECT_ROOT / "reports"
PROVIDED_DIR = PROJECT_ROOT / "Provided"
SATELLITE_TEST_DIR = PROJECT_ROOT / "data" / "satellite_test"

# Create directories if they don't exist
for d in [EMBEDDINGS_DIR, REPORTS_DIR, SATELLITE_TEST_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("TEST SET PRICE PREDICTION PIPELINE")
print("=" * 80)

# ============================================================================
# Step 1: Load test data
# ============================================================================
print("\n[Step 1] Loading test data...")
test_candidates = [
    PROJECT_ROOT / "data" / "raw" / "test.xlsx",
    PROJECT_ROOT / "test.xlsx",
    PROVIDED_DIR / "test.xlsx",
]

test_path = None
for p in test_candidates:
    if p.exists():
        test_path = p
        break

if test_path is None:
    raise FileNotFoundError(f"Could not find test.xlsx in {test_candidates}")

test_df = pd.read_excel(test_path)
print(f"Loaded {len(test_df)} test properties from {test_path}")

# Infer column names
id_col = "id" if "id" in test_df.columns else "Id"
lat_col = "lat" if "lat" in test_df.columns else "latitude"
lon_col = "long" if "long" in test_df.columns else "lon" if "lon" in test_df.columns else "longitude"

if id_col not in test_df.columns:
    raise ValueError(f"ID column '{id_col}' not found in test data")
if lat_col not in test_df.columns or lon_col not in test_df.columns:
    raise ValueError(
        f"Latitude/longitude columns not found. Found columns: {test_df.columns.tolist()}"
    )

print(f"Using columns: id={id_col}, lat={lat_col}, lon={lon_col}")

# ============================================================================
# Step 2: Download satellite images for test set
# ============================================================================
print("\n[Step 2] Downloading satellite images for test set...")
print(f"Images will be saved to: {SATELLITE_TEST_DIR}")

fetcher = SatelliteImageFetcher(
    output_dir=SATELLITE_TEST_DIR,
    resolution=10,
    context_size_m=400,
    max_cloud_fraction=0.3,
)

# Fetch images (will skip if already downloaded)
test_meta = fetcher.fetch_for_dataframe(
    df=test_df,
    id_col=id_col,
    lat_col=lat_col,
    lon_col=lon_col,
    overwrite=False,  # Don't re-download existing images
    limit=None,  # Fetch all
)

# Save test metadata
test_meta_path = SATELLITE_TEST_DIR / "image_metadata.csv"
test_meta.to_csv(test_meta_path, index=False)

successful_downloads = len(test_meta[test_meta["status"].isin(["ok", "cached"])])
print(f"Successfully fetched/cached {successful_downloads} images out of {len(test_df)} properties")
print(f"Test metadata saved to: {test_meta_path}")

# ============================================================================
# Step 3: Load training data and prepare models
# ============================================================================
print("\n[Step 3] Loading training data and preparing models...")

# Load processed splits
train_df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
val_df = pd.read_parquet(PROCESSED_DIR / "val.parquet")

target_col = "log_price" if "log_price" in train_df.columns else "price"
price_col = "price" if "price" in train_df.columns else target_col

print(f"Target column: {target_col}")
print(f"Price column: {price_col}")

# Prepare tabular features
y_train = train_df[target_col].values
X_train = train_df.drop(columns=[target_col])
if target_col == "log_price" and price_col in X_train.columns:
    X_train = X_train.drop(columns=[price_col])

numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
if id_col in numeric_features:
    numeric_features.remove(id_col)

categorical_features = [
    c for c in X_train.columns if c not in numeric_features and c != id_col
]

print(f"Numeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[("scaler", StandardScaler())]), numeric_features),
        (
            "cat",
            Pipeline(
                steps=[
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    )
                ]
            ),
            categorical_features,
        ),
    ]
)

# Fit preprocessor on training data
X_train_tab = preprocessor.fit_transform(X_train)

# Train tabular Random Forest
print("Training tabular Random Forest model...")
tab_rf = RandomForestRegressor(
    n_estimators=200, max_depth=None, min_samples_leaf=2, n_jobs=-1, random_state=42
)
tab_rf.fit(X_train_tab, y_train)

# Load image embeddings for training
train_img_emb = pd.read_parquet(EMBEDDINGS_DIR / "resnet18_train_embeddings.parquet")
val_img_emb = pd.read_parquet(EMBEDDINGS_DIR / "resnet18_val_embeddings.parquet")

emb_cols = [c for c in train_img_emb.columns if c.startswith("img_emb_")]
print(f"Image embedding dimension: {len(emb_cols)}")

# Train image-only Random Forest
print("Training image-only Random Forest model...")
train_with_img = train_df.merge(train_img_emb, on=id_col, how="inner")
X_train_img = train_with_img[emb_cols].values
y_train_img = train_with_img[target_col].values

img_rf = RandomForestRegressor(
    n_estimators=300, max_depth=None, min_samples_leaf=2, n_jobs=-1, random_state=42
)
img_rf.fit(X_train_img, y_train_img)

# Train late fusion combiner
print("Training late fusion model...")
tab_train_pred = tab_rf.predict(
    preprocessor.transform(
        train_with_img.drop(columns=[target_col, id_col] + emb_cols)
    )
)
img_train_pred = img_rf.predict(X_train_img)

# Convert to price scale if needed
if target_col == "log_price":
    tab_train_pred_price = np.expm1(tab_train_pred)
    img_train_pred_price = np.expm1(img_train_pred)
    y_train_price = train_with_img[price_col].values
else:
    tab_train_pred_price = tab_train_pred
    img_train_pred_price = img_train_pred
    y_train_price = y_train_img

# Train fusion regressor
fusion_X_train = np.column_stack([tab_train_pred_price, img_train_pred_price])
fusion_reg = LinearRegression()
fusion_reg.fit(fusion_X_train, y_train_price)

print("All models trained and ready!")

# ============================================================================
# Step 4: Extract ResNet embeddings for test images
# ============================================================================
print("\n[Step 4] Extracting ResNet embeddings for test images...")

IMG_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Image transforms
img_transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class SatelliteDataset(Dataset):
    """Dataset for loading satellite images."""

    def __init__(self, df, id_col, transform=None):
        self.df = df.reset_index(drop=True)
        self.id_col = id_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        sample_id = row[self.id_col]
        return img, sample_id


# Load ResNet18
try:
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
except AttributeError:
    resnet = models.resnet18(pretrained=True)

for param in resnet.parameters():
    param.requires_grad = False

feature_dim = resnet.fc.in_features
resnet.fc = nn.Identity()
resnet.eval()
resnet.to(device)

# Filter test metadata to only properties with images
test_meta_filtered = test_meta[test_meta["status"].isin(["ok", "cached"])]
test_meta_filtered = test_meta_filtered[
    test_meta_filtered["image_path"].apply(lambda p: Path(p).exists())
]

print(f"Extracting embeddings for {len(test_meta_filtered)} test properties with images...")

if len(test_meta_filtered) > 0:
    test_dataset = SatelliteDataset(
        test_meta_filtered[[id_col, "image_path"]],
        id_col=id_col,
        transform=img_transform,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=0
    )

    all_embeddings = []
    all_ids = []

    with torch.no_grad():
        for imgs, ids in tqdm(test_loader, desc="Extracting embeddings"):
            imgs = imgs.to(device)
            feats = resnet(imgs)
            all_embeddings.append(feats.cpu().numpy())
            all_ids.extend([int(i) for i in ids])

    test_emb_array = np.concatenate(all_embeddings, axis=0)
    test_emb_df = pd.DataFrame(test_emb_array, columns=emb_cols)
    test_emb_df[id_col] = all_ids

    # Save test embeddings
    test_emb_path = EMBEDDINGS_DIR / "resnet18_test_embeddings.parquet"
    test_emb_df.to_parquet(test_emb_path, index=False)
    print(f"Saved test embeddings to {test_emb_path}")
else:
    print("Warning: No test images available. Will use tabular-only predictions.")
    test_emb_df = None

# ============================================================================
# Step 5: Generate predictions
# ============================================================================
print("\n[Step 5] Generating predictions...")

# Prepare test tabular features
X_test = test_df.drop(
    columns=[c for c in [target_col, "log_price", "price"] if c in test_df.columns]
)

# Align columns with training features
missing_numeric = [c for c in numeric_features if c not in X_test.columns]
for c in missing_numeric:
    X_test[c] = np.nan

missing_cats = [c for c in categorical_features if c not in X_test.columns]
for c in missing_cats:
    X_test[c] = "missing"

X_test = X_test[numeric_features + categorical_features]
X_test_tab = preprocessor.transform(X_test)

# Get tabular predictions for all test properties
test_pred_tab = tab_rf.predict(X_test_tab)

# Initialize prediction dataframe
pred_df = pd.DataFrame(
    {
        id_col: test_df[id_col].values,
        "actual_price": np.nan,  # Test set has no labels
        "predicted_price": np.nan,
    }
)

# For properties with images, use late fusion; for others, use tabular-only
if test_emb_df is not None and len(test_emb_df) > 0:
    # Merge test data with embeddings
    test_with_img = test_df.merge(test_emb_df, on=id_col, how="inner")

    # Get indices of properties with images
    test_with_img_ids = set(test_with_img[id_col].values)

    # Tabular predictions for properties with images
    X_test_with_img = test_df[test_df[id_col].isin(test_with_img_ids)].copy()
    X_test_with_img_tab = preprocessor.transform(
        X_test_with_img.drop(
            columns=[
                c
                for c in [target_col, "log_price", "price"]
                if c in X_test_with_img.columns
            ]
        )
    )
    tab_pred_with_img = tab_rf.predict(X_test_with_img_tab)

    # Image predictions
    img_pred_test = img_rf.predict(test_with_img[emb_cols].values)

    # Late fusion predictions
    if target_col == "log_price":
        tab_pred_price = np.expm1(tab_pred_with_img)
        img_pred_price = np.expm1(img_pred_test)
    else:
        tab_pred_price = tab_pred_with_img
        img_pred_price = img_pred_test

    fusion_X_test = np.column_stack([tab_pred_price, img_pred_price])
    fusion_pred_price = fusion_reg.predict(fusion_X_test)

    # Update predictions for properties with images
    for idx, pid in enumerate(test_with_img[id_col].values):
        pred_df.loc[pred_df[id_col] == pid, "predicted_price"] = fusion_pred_price[idx]

    print(f"Used late fusion for {len(test_with_img)} properties with images")

    # For properties without images, use tabular-only
    test_without_img_ids = set(test_df[id_col].values) - test_with_img_ids
    if len(test_without_img_ids) > 0:
        test_without_img = test_df[test_df[id_col].isin(test_without_img_ids)]
        X_test_without_img_tab = preprocessor.transform(
            test_without_img.drop(
                columns=[
                    c
                    for c in [target_col, "log_price", "price"]
                    if c in test_without_img.columns
                ]
            )
        )
        tab_pred_without_img = tab_rf.predict(X_test_without_img_tab)

        if target_col == "log_price":
            tab_pred_without_img_price = np.expm1(tab_pred_without_img)
        else:
            tab_pred_without_img_price = tab_pred_without_img

        for idx, pid in enumerate(test_without_img[id_col].values):
            pred_df.loc[
                pred_df[id_col] == pid, "predicted_price"
            ] = tab_pred_without_img_price[idx]

        print(f"Used tabular-only for {len(test_without_img_ids)} properties without images")
else:
    # No images available, use tabular-only for all
    if target_col == "log_price":
        test_pred_price = np.expm1(test_pred_tab)
    else:
        test_pred_price = test_pred_tab

    pred_df["predicted_price"] = test_pred_price
    print("No test images available. Used tabular-only predictions for all properties.")

# ============================================================================
# Step 6: Save final predictions
# ============================================================================
print("\n[Step 6] Saving final predictions...")

out_csv = REPORTS_DIR / "predictions_test_final.csv"
pred_df.to_csv(out_csv, index=False)

print(f"\n{'='*80}")
print("PREDICTION PIPELINE COMPLETE")
print(f"{'='*80}")
print(f"\nSaved predictions to: {out_csv}")
print(f"Total predictions: {len(pred_df)}")
print(f"Properties with images: {len(test_emb_df) if test_emb_df is not None else 0}")
print(f"Price range: ${pred_df['predicted_price'].min():,.0f} - ${pred_df['predicted_price'].max():,.0f}")
print(f"Mean predicted price: ${pred_df['predicted_price'].mean():,.0f}")
print(f"Median predicted price: ${pred_df['predicted_price'].median():,.0f}")
print(f"\nOutput format: id, actual_price, predicted_price")
print(f"First few predictions:")
print(pred_df.head(10).to_string(index=False))

