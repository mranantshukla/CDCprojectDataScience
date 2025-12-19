
import os
import time
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

from sentinelhub import (
    SHConfig,
    BBox,
    CRS,
    MimeType,
    SentinelHubRequest,
    DataCollection,
    bbox_to_dimensions,
)


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
SATELLITE_DIR = DATA_DIR / "satellite"
SATELLITE_DIR.mkdir(parents=True, exist_ok=True)
METADATA_PATH = SATELLITE_DIR / "image_metadata.csv"


def load_sentinelhub_config() -> SHConfig:
    """
    Load Sentinel Hub configuration from environment variables or .env.

    Required:
        SENTINELHUB_CLIENT_ID
        SENTINELHUB_CLIENT_SECRET

    Optionally:
        SENTINELHUB_INSTANCE_ID  (for legacy setups)

    As a fallback (for this project only), if env vars are missing we also
    try to parse `Provided/APICredentials.txt`, which is expected to contain
    lines like:
        client_id=\"...\"
        client_secret=\"...\"
    """
    load_dotenv()

    config = SHConfig()

    client_id ="ff2009e3-12dd-485b-90dc-ec6b917cb5f2"
    client_secret="mXGt5KPDZyEsCqW2PEGVBN1iiHZDhEU0"
    instance_id = os.getenv("SENTINELHUB_INSTANCE_ID")

    # Fallback: parse Provided/APICredentials.txt if env vars unset
    if not client_id or not client_secret:
        creds_path = PROJECT_ROOT / "Provided" / "APICredentials.txt"
        if creds_path.exists():
            try:
                with creds_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        low = line.lower()
                        if "client_id" in low and "=" in line:
                            value = line.split("=", 1)[1].strip().strip("\"' ")
                            if value:
                                client_id = client_id or value
                        if "client_secret" in low and "=" in line:
                            value = line.split("=", 1)[1].strip().strip("\"' ")
                            if value:
                                client_secret = client_secret or value
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to parse APICredentials.txt: %s", repr(exc))

    if client_id:
        config.sh_client_id = client_id
    if client_secret:
        config.sh_client_secret = client_secret
    if instance_id:
        config.instance_id = instance_id

    if not (config.sh_client_id and config.sh_client_secret):
        raise RuntimeError(
            "Sentinel Hub credentials are not configured. "
            "Set SENTINELHUB_CLIENT_ID and SENTINELHUB_CLIENT_SECRET "
            "as environment variables or in a .env file."
        )

    return config


def latlon_to_bbox(lat: float, lon: float, half_size_m: float) -> BBox:
    """
    Convert a latitude/longitude point and a half-size in meters
    into a WGS84 bounding box.

    We approximate the Earth as a sphere which is sufficient at the
    neighborhood scale relevant for economic context.
    """
    # Earth radius in meters
    earth_radius = 6_378_137.0

    d_lat = (half_size_m / earth_radius) * (180.0 / np.pi)
    d_lon = (half_size_m / (earth_radius * np.cos(np.deg2rad(lat)))) * (180.0 / np.pi)

    min_lat = lat - d_lat
    max_lat = lat + d_lat
    min_lon = lon - d_lon
    max_lon = lon + d_lon

    return BBox(bbox=[min_lon, min_lat, max_lon, max_lat], crs=CRS.WGS84)


DEFAULT_EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B04", "B03", "B02"],
      units: "REFLECTANCE"
    }],
    output: {
      bands: 3,
      sampleType: "AUTO"
    }
  };
}

function evaluatePixel(sample) {
  // Simple natural color RGB
  return [sample.B04, sample.B03, sample.B02];
}
"""


@dataclass
class ImageRecord:
    """Metadata record linking a house to its satellite image."""

    id: int
    lat: float
    lon: float
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float
    image_path: str
    status: str
    error: Optional[str] = None


class SatelliteImageFetcher:
    """
    Robust Sentinel Hub pipeline for fetching neighborhood-scale tiles
    around properties in the King County dataset.

    Economically, we care about a walkable neighborhood context, not just
    the parcel footprint. The `context_size_m` parameter controls the
    radius of the bounding box in meters.
    """

    def __init__(
        self,
        config: Optional[SHConfig] = None,
        collection: DataCollection = DataCollection.SENTINEL2_L2A,
        resolution: int = 10,
        context_size_m: int = 400,
        max_cloud_fraction: float = 0.2,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        output_dir: Path = SATELLITE_DIR,
        evalscript: str = DEFAULT_EVALSCRIPT,
    ) -> None:
        """
        Args:
            config: Sentinel Hub configuration. If None, load from env.
            collection: Sentinel data collection to use.
            resolution: Ground sampling distance in meters per pixel.
                10 m is a good compromise: parcels and neighborhood
                pattern are visible without being too heavy.
            context_size_m: Half-size of the context window in meters.
                Total window is roughly 2 * context_size_m across.
            max_cloud_fraction: Maximum allowed cloud fraction (0–1).
            max_retries: Number of retries for transient errors.
            backoff_factor: Exponential backoff base.
            output_dir: Directory to store image tiles.
            evalscript: Sentinel Hub evalscript string for band selection.
        """
        self.config = config or load_sentinelhub_config()
        self.collection = collection
        self.resolution = resolution
        self.context_size_m = context_size_m
        self.max_cloud_fraction = max_cloud_fraction
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.evalscript = evalscript

    def _build_request(
        self,
        bbox: BBox,
        time_interval: Tuple[str, str],
    ) -> SentinelHubRequest:
        """Internal helper to construct a Sentinel Hub request."""
        size = bbox_to_dimensions(bbox, self.resolution)

        request = SentinelHubRequest(
            evalscript=self.evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=self.collection,
                    time_interval=time_interval,
                    mosaicking_order="mostRecent",
                )
            ],
            responses=[
                SentinelHubRequest.output_response("default", MimeType.PNG)
            ],
            bbox=bbox,
            size=size,
            config=self.config,
        )
        return request

    def fetch_image_for_property(
        self,
        property_id: int,
        lat: float,
        lon: float,
        time_interval: Tuple[str, str] = ("2018-01-01", "2021-12-31"),
        overwrite: bool = False,
    ) -> ImageRecord:
        """
        Fetch a satellite tile for a single property and save it to disk.

        Returns an ImageRecord describing the result (success, cached, or failure).
        """
        if np.isnan(lat) or np.isnan(lon):
            return ImageRecord(
                id=int(property_id),
                lat=float(lat),
                lon=float(lon),
                min_lat=np.nan,
                min_lon=np.nan,
                max_lat=np.nan,
                max_lon=np.nan,
                image_path="",
                status="missing_coordinates",
                error="Latitude/longitude is NaN.",
            )

        bbox = latlon_to_bbox(lat, lon, half_size_m=self.context_size_m)
        min_lon, min_lat, max_lon, max_lat = bbox

        # Deterministic filename: stable across runs
        filename = f"{property_id}_s2_r{self.resolution}_c{self.context_size_m}.png"
        image_path = self.output_dir / filename

        if image_path.exists() and not overwrite:
            return ImageRecord(
                id=int(property_id),
                lat=float(lat),
                lon=float(lon),
                min_lat=float(min_lat),
                min_lon=float(min_lon),
                max_lat=float(max_lat),
                max_lon=float(max_lon),
                image_path=str(image_path),
                status="cached",
                error=None,
            )

        request = self._build_request(bbox=bbox, time_interval=time_interval)

        last_error: Optional[str] = None
        status = "failed"

        for attempt in range(1, self.max_retries + 1):
            try:
                data = request.get_data()[0]  # H x W x C

                # Sentinel Hub may already scale to 0–255 uint8.
                if data.dtype != np.uint8:
                    data = np.clip(data, 0, 1)
                    data = (255 * data).astype(np.uint8)

                img = Image.fromarray(data)
                img.save(image_path)

                status = "ok"
                last_error = None
                break
            except Exception as exc:  # noqa: BLE001
                last_error = repr(exc)
                logger.warning(
                    "Attempt %d/%d failed for id=%s: %s",
                    attempt,
                    self.max_retries,
                    property_id,
                    last_error,
                )
                if attempt < self.max_retries:
                    sleep_seconds = self.backoff_factor ** (attempt - 1)
                    time.sleep(sleep_seconds)

        if status != "ok":
            logger.error(
                "Failed to fetch image for id=%s after %d attempts",
                property_id,
                self.max_retries,
            )

        return ImageRecord(
            id=int(property_id),
            lat=float(lat),
            lon=float(lon),
            min_lat=float(min_lat),
            min_lon=float(min_lon),
            max_lat=float(max_lat),
            max_lon=float(max_lon),
            image_path=str(image_path if status == "ok" else ""),
            status=status,
            error=last_error,
        )

    def fetch_for_dataframe(
        self,
        df: pd.DataFrame,
        id_col: str = "id",
        lat_col: str = "lat",
        lon_col: str = "long",
        time_interval: Tuple[str, str] = ("2018-01-01", "2021-12-31"),
        overwrite: bool = False,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch images for all properties in a DataFrame.

        Args:
            df: DataFrame containing at least id, lat, lon columns.
            id_col: Column name for unique property identifier.
            lat_col: Column name for latitude.
            lon_col: Column name for longitude.
            time_interval: Time window for imagery.
            overwrite: If True, re-download even if files exist.
            limit: Optional cap on the number of rows for a dry run.

        Returns:
            DataFrame of image metadata, also written to METADATA_PATH.
        """
        if id_col not in df.columns:
            raise ValueError(f"DataFrame must contain '{id_col}' column")
        if lat_col not in df.columns or lon_col not in df.columns:
            raise ValueError(
                f"DataFrame must contain '{lat_col}' and '{lon_col}' columns"
            )

        if limit is not None:
            df_iter = df.head(limit)
        else:
            df_iter = df

        records: List[ImageRecord] = []

        for row in tqdm(df_iter.itertuples(index=False), total=len(df_iter)):
            row_dict = row._asdict() if hasattr(row, "_asdict") else dict(row._asdict())
            prop_id = row_dict[id_col]
            lat = float(row_dict[lat_col])
            lon = float(row_dict[lon_col])

            rec = self.fetch_image_for_property(
                property_id=prop_id,
                lat=lat,
                lon=lon,
                time_interval=time_interval,
                overwrite=overwrite,
            )
            records.append(rec)

        new_meta = pd.DataFrame([asdict(r) for r in records])

        if METADATA_PATH.exists():
            existing = pd.read_csv(METADATA_PATH)
            combined = (
                pd.concat([existing, new_meta], ignore_index=True)
                .drop_duplicates(subset=["id"], keep="last")
                .sort_values("id")
            )
        else:
            combined = new_meta.sort_values("id")

        combined.to_csv(METADATA_PATH, index=False)
        logger.info("Saved image metadata to %s", METADATA_PATH)

        return combined


def _parse_args() -> Tuple[Optional[Path], str, str, str, bool, Optional[int]]:
    """Simple CLI interface for batch fetching."""
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Fetch Sentinel satellite tiles for the King County housing dataset. "
            "This is designed as an engineering-quality data acquisition step, "
            "with retries and deterministic naming for reproducibility."
        )
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/raw/kc_house_data.csv",
        help="Path to the CSV file with property data.",
    )
    parser.add_argument(
        "--id-col",
        type=str,
        default="id",
        help="Column name for property ID.",
    )
    parser.add_argument(
        "--lat-col",
        type=str,
        default="lat",
        help="Column name for latitude.",
    )
    parser.add_argument(
        "--lon-col",
        type=str,
        default="long",
        help="Column name for longitude.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download images even if files exist.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of rows for a quick dry run.",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else None
    return (
        csv_path,
        args.id_col,
        args.lat_col,
        args.lon_col,
        args.overwrite,
        args.limit,
    )


def main() -> None:
    """Entry point for CLI usage."""
    csv_path, id_col, lat_col, lon_col, overwrite, limit = _parse_args()

    if csv_path is None or not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found at '{csv_path}'. "
            "Place the King County dataset at data/raw/kc_house_data.csv "
            "or provide --csv explicitly."
        )

    logger.info("Reading property data from %s", csv_path)
    df = pd.read_csv(csv_path)

    fetcher = SatelliteImageFetcher()

    logger.info(
        "Starting satellite image fetch for %d properties (limit=%s)",
        len(df),
        limit,
    )
    meta = fetcher.fetch_for_dataframe(
        df=df,
        id_col=id_col,
        lat_col=lat_col,
        lon_col=lon_col,
        overwrite=overwrite,
        limit=limit,
    )

    # Basic summary: how many successes vs failures
    summary = meta["status"].value_counts(dropna=False)
    logger.info("Fetch summary:\n%s", summary.to_string())


if __name__ == "__main__":
    main()


