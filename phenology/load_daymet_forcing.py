"""Load and align daily Daymet forcing rasters by year.

This script reads daily Daymet rasters for each requested year, window-reads
source data, reprojects it to a corresponding imagery grid, and computes:
1) Daily average temperature (tavg) from tmax/tmin,
2) Daily day length (dayl),
3) Cumulative chilling units (CU) across the year.

Band mapping in the Daymet daily raster (0-based, as requested):
- tmax: band 0
- tmin: band 1
- dayl: band 3

Because rasterio uses 1-based indexing for band reads, this script reads
bands [1, 2, 4].
"""

from __future__ import annotations

import glob
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import reproject
from rasterio.windows import Window, from_bounds


def create_window_grid(
    image_path: str,
    window_size: int = 1000,
) -> List[Window]:
    """Create a grid of windows that fully covers an image.

    The returned windows are laid out left-to-right, top-to-bottom.
    Edge windows are clipped so every pixel in the image is covered.

    Args:
      image_path: Path to a raster image.
      window_size: Desired square window size in pixels.

    Returns:
      A list of rasterio Window objects.
    """
    if window_size <= 0:
        raise ValueError("window_size must be a positive integer")

    windows: List[Window] = []
    with rio.open(image_path) as src:
        for row_off in range(0, src.height, window_size):
            height = min(window_size, src.height - row_off)
            for col_off in range(0, src.width, window_size):
                width = min(window_size, src.width - col_off)
                windows.append(Window(col_off, row_off, width, height))

    return windows


def list_daily_daymet_files(daymet_dir: str, year: int) -> List[str]:
    """List daily Daymet files for a year using YYYY*.tif naming."""
    # Load December of previous year to September of current year
    pattern1 = os.path.join(daymet_dir, f"{year-1}12*.tif")
    pattern2 = os.path.join(daymet_dir, f"{year}0*.tif")
    files = sorted(glob.glob(pattern1) + glob.glob(pattern2))
    if not files:
        raise FileNotFoundError(f"No Daymet files found for year {year} in {daymet_dir}")
    return files


def get_target_grid(
    imagery_path: str,
    window: Optional[Window],
) -> dict:
    """Get destination profile, shape, and transform from imagery."""
    with rio.open(imagery_path) as img:
        if window is None:
            height = img.height
            width = img.width
            transform = img.transform
        else:
            height = int(window.height)
            width = int(window.width)
            transform = img.window_transform(window)

        profile = {
            "crs": img.crs,
            "transform": transform,
            "height": height,
            "width": width,
        }
    return profile


def read_and_reproject_daymet(
    daymet_path: str,
    dst_profile: dict
) -> np.ndarray:
    """Read Daymet tmax/tmin/dayl with source windowing and reproject to target.

    Returns:
      A numpy array with shape (3, rows, cols) containing reprocted data from 
      the raster at `daymet_path`.
    """
    dst_crs = dst_profile['crs']
    dst_height = dst_profile['height']
    dst_width = dst_profile['width']
    dst_transform = dst_profile['transform']

    with rio.open(daymet_path) as src:
        dst_bounds = array_bounds(dst_height, dst_width, dst_transform)
        src_window = from_bounds(*dst_bounds, transform=src.transform)

        # Requested mapping is 0-based: tmax=0, tmin=1, dayl=2.
        # Rasterio band indices are 1-based: [1, 2, 3].
        src_data = src.read(
            indexes=[1, 2, 3],
            window=src_window,
            boundless=True,
        )

        dest = np.full((3, dst_height, dst_width), np.nan, dtype=np.float32)

        reproject(
            source=src_data,
            destination=dest,
            src_transform=src.window_transform(src_window),
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    return dest


def load_year_forcing(
    year: int,
    daymet_dir: str,
    imagery_path: str,
    chill_threshold: float,
    window: Optional[Window],
) -> Dict[str, np.ndarray]:
    """Load all daily forcings for a year and compute CU/tavg stacks.

    Returns dictionary with keys:
      tavg: (days, rows, cols)
      dayl: (days, rows, cols)
      cu:   (days, rows, cols)
    """
    daymet_files = list_daily_daymet_files(daymet_dir, year)
    dst_profile = get_target_grid(imagery_path, window)

    dest_days: List[np.ndarray] = []

    for daymet_path in daymet_files:
        dest_day = read_and_reproject_daymet(
            daymet_path=daymet_path,
            dst_profile=dst_profile
        )
        dest_days.append(dest_day)

    dest = np.stack(dest_days, axis=0)
    tavg = (dest[:, 0, :, :] + dest[:, 1, :, :]) / 2.0
    cu = np.cumsum(np.less(tavg, chill_threshold), axis=0).astype(np.float32)
    dayl = dest[:, 2, :, :]/86400.0

    return tavg, dayl, cu