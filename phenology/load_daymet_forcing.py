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
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds


def create_window_grid(
    image_path: str,
    window_height: int = 1000,
    window_width: int = 1000,
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
    if window_height <= 0 or window_width <= 0:
        raise ValueError("window_height and window_width must be positive integers")

    windows: List[Window] = []
    with rio.open(image_path) as src:
        for row_off in range(0, src.height, window_height):
            height = min(window_height, src.height - row_off)
            for col_off in range(0, src.width, window_width):
                width = min(window_width, src.width - col_off)
                windows.append(Window(col_off, row_off, width, height))

    return windows


def list_daily_daymet_files(daymet_dir: str, year: int) -> List[str]:
    """List daily Daymet files for a year using YYYY*.tif naming."""
    # Load December of previous year to September of current year
    # Load each month one at a time to validate enough files were downloaded
    pattern_dec = os.path.join(daymet_dir, f"{year-1}12*.tif")
    pattern_jan = os.path.join(daymet_dir, f"{year}01*.tif")
    pattern_feb = os.path.join(daymet_dir, f"{year}02*.tif")
    pattern_mar = os.path.join(daymet_dir, f"{year}03*.tif")
    pattern_apr = os.path.join(daymet_dir, f"{year}04*.tif")
    pattern_may = os.path.join(daymet_dir, f"{year}05*.tif")
    pattern_jun = os.path.join(daymet_dir, f"{year}06*.tif")
    pattern_jul = os.path.join(daymet_dir, f"{year}07*.tif")
    pattern_aug = os.path.join(daymet_dir, f"{year}08*.tif")
    pattern_sep = os.path.join(daymet_dir, f"{year}09*.tif")
    print(len(glob.glob(pattern_dec)))
    print(len(glob.glob(pattern_jan)))
    print(len(glob.glob(pattern_feb)))
    print(len(glob.glob(pattern_mar)))
    print(len(glob.glob(pattern_apr)))
    print(len(glob.glob(pattern_may)))
    print(len(glob.glob(pattern_jun)))
    print(len(glob.glob(pattern_jul)))
    print(len(glob.glob(pattern_aug)))
    print(len(glob.glob(pattern_sep)))
    files = sorted(glob.glob(pattern_dec) + glob.glob(pattern_jan) +
                   glob.glob(pattern_feb) + glob.glob(pattern_mar) +
                   glob.glob(pattern_apr) + glob.glob(pattern_may) +
                   glob.glob(pattern_jun) + glob.glob(pattern_jul) +
                   glob.glob(pattern_aug) + glob.glob(pattern_sep))
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
        src_bounds = transform_bounds(dst_crs, src.crs, *dst_bounds)
        src_window = from_bounds(*src_bounds, transform=src.transform)
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
    print('Reading using transform: ', dst_profile)
    dest_days: List[np.ndarray] = []
    for daymet_path in daymet_files:
        dest_day = read_and_reproject_daymet(
            daymet_path=daymet_path,
            dst_profile=dst_profile
        )
        dest_days.append(dest_day)
    dest = np.stack(dest_days, axis=0, dtype= np.float32)
    dest[:, [0,1], :, :] = dest[:, [0,1], :, :] * 120 / 65535 - 60 # unshrink tmax/tmin
    tavg = (dest[:, 0, :, :] + dest[:, 1, :, :]) / 2.0
    cu = np.cumsum(np.less(tavg, chill_threshold), axis=0).astype(np.float32)
    dayl = dest[:, 2, :, :]/86400.0

    # On leap years, December 31st is missing, so we only discard 30 days instead of 31
    # ensure every data frame has the same number of observations.
    print('before', tavg.shape)
    if (year-1) % 4 == 0:
        tavg = tavg[30:,:,:]
        cu = cu[30:,:,:]
        dayl = dayl[30:,:,:]
    else:
        tavg = tavg[31:,:,:]
        cu = cu[31:,:,:]
        dayl = dayl[31:,:,:]
    print('after', tavg.shape)

    return tavg, dayl, cu
