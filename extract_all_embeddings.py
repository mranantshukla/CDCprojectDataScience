"""Extract ResNet18 embeddings for all properties with satellite images."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SATELLITE_DIR = PROJECT_ROOT / "data" / "satellite"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"

EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# Load processed splits
train_df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
val_df = pd.read_parquet(PROCESSED_DIR / "val.parquet")

target_col = "log_price" if "log_price" in train_df.columns else "price"
id_col = "id" if "id" in train_df.columns else "Id"

# Load satellite metadata
meta_df = pd.read_csv(SATELLITE_DIR / "image_metadata.csv")
meta_df = meta_df[meta_df["status"].isin(["ok", "cached"])]
if id_col != "id" and "id" in meta_df.columns:
    meta_df = meta_df.rename(columns={"id": id_col})
meta_df = meta_df[meta_df["image_path"].apply(lambda p: Path(p).exists())]

print(f"Total properties with images: {len(meta_df)}")
print(f"Train properties: {len(train_df)}")
print(f"Val properties: {len(val_df)}")

IMG_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

img_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class SatelliteDataset(Dataset):
    def __init__(self, df, id_col, target_col, transform=None):
        self.df = df.reset_index(drop=True)
        self.id_col = id_col
        self.target_col = target_col
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        target = float(row[self.target_col])
        sample_id = row[self.id_col]
        return img, target, sample_id


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


def extract_embeddings(split_df, split_name, batch_size=64):
    """Extract embeddings for a split."""
    out_path = EMBEDDINGS_DIR / f"resnet18_{split_name}_embeddings.parquet"
    
    # Merge with image metadata
    merged = split_df[[id_col, target_col]].merge(
        meta_df[[id_col, "image_path"]], on=id_col, how="inner"
    )
    
    if merged.empty:
        print(f"Warning: No images for {split_name} split")
        return None
    
    print(f"\nExtracting embeddings for {split_name}: {len(merged)} properties")
    
    dataset = SatelliteDataset(
        merged[[id_col, "image_path", target_col]],
        id_col=id_col,
        target_col=target_col,
        transform=img_transform,
    )
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    all_embeddings = []
    all_ids = []
    
    with torch.no_grad():
        for imgs, _, ids in loader:
            imgs = imgs.to(device)
            feats = resnet(imgs)
            all_embeddings.append(feats.cpu().numpy())
            all_ids.extend([int(i) for i in ids])
    
    emb_array = np.concatenate(all_embeddings, axis=0)
    emb_cols = [f"img_emb_{i}" for i in range(emb_array.shape[1])]
    emb_df = pd.DataFrame(emb_array, columns=emb_cols)
    emb_df[id_col] = all_ids
    
    emb_df.to_parquet(out_path, index=False)
    print(f"Saved {split_name} embeddings: {emb_df.shape} to {out_path}")
    
    return emb_df


if __name__ == "__main__":
    print("=" * 80)
    print("EXTRACTING RESNET EMBEDDINGS FOR FULL DATASET")
    print("=" * 80)
    
    train_emb = extract_embeddings(train_df, "train", batch_size=64)
    val_emb = extract_embeddings(val_df, "val", batch_size=64)
    
    print("\n" + "=" * 80)
    print("Embedding extraction complete!")
    print("=" * 80)

