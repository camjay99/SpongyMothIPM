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

from datetime import datetime
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
    pattern_prior = os.path.join(daymet_dir, f"{year-1}1*.tif")
    files_prior = sorted(glob.glob(pattern_prior))
    def last_31_days(filename: str) -> bool:
        basename = os.path.basename(filename)
        date = datetime(int(basename[0:4]), int(basename[4:6]), int(basename[6:8]))
        return date.timetuple().tm_yday >= 335 # check if date is in last 31 days of the year
    files_prior = list(filter(last_31_days, files_prior))

    pattern_current = os.path.join(daymet_dir, f"{year}0*.tif")
    files_current = sorted(glob.glob(pattern_current))

    files = files_prior + files_current
    if not files:
        raise FileNotFoundError(f"No Daymet files found for year {year} in {daymet_dir}")
    return files

def list_daily_cmip6_files(cmip6_dir: str, year: int) -> List[str]:
    """List daily CMIP6 anomaly files for a year using *YYYY*.tif naming.

    As CMIP6 files are weekly, this function replicates each weekly file for the
    number of days it covers to create a daily file list aligned with the Daymet files.
    It assumes CMIP6 files are named like "<model>_<scenario>_<year>_<doy_start>_<doy_end>.tif"
    where doy_start and doy_end are the day of year range covered by the file.
    """

    daily_files = []
    # Get prior year December files
    pattern = os.path.join(cmip6_dir, f"*{year-1}*.tif")
    prior_year_files = sorted(glob.glob(pattern))
    for file in prior_year_files:
        filename = os.path.basename(file)
        filename = filename.split('.')[0] # remove .tif extension
        parts = filename.split('_')
        doy_end = int(parts[4])
        if doy_end < 335: # covers at least some of December
            continue
        daily_files.extend([file] * (int(parts[4]) - max(int(parts[3]), 335) + 1))

    # Get Current year files
    pattern = os.path.join(cmip6_dir, f"*{year}*.tif")
    weekly_files = sorted(glob.glob(pattern))
    if not weekly_files:
        raise FileNotFoundError(f"No CMIP6 files found for year {year} in {cmip6_dir}")
    for file in weekly_files:
        filename = os.path.basename(file)
        filename = filename.split('.')[0] # remove .tif extension
        parts = filename.split('_')
        daily_files.extend([file] * (int(parts[4]) - int(parts[3]) + 1))

    return daily_files

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
        print(transform)
        print(window)
        profile = {
            "crs": img.crs,
            "transform": transform,
            "height": height,
            "width": width,
        }
    return profile


def read_dataset(
    path: str,
    dst_profile: dict
) -> np.ndarray:
    """Read Daymet tmax/tmin/dayl with source window based on a destination
    profile.

    Returns:
      data: A numpy array with shape (3, rows, cols) containing data from the raster
        at `daymet_path`.
      src_profile: The profile of the source raster with transform updated to
        reflect the updated extent.
    """
    dst_crs = dst_profile['crs']
    dst_height = dst_profile['height']
    dst_width = dst_profile['width']
    dst_transform = dst_profile['transform']

    with rio.open(path) as src:
        src_profile = src.profile

        # Create corresponding source window and transform based on
        # destination bounds.
        dst_bounds = array_bounds(dst_height, dst_width, dst_transform)
        src_bounds = transform_bounds(dst_crs, src.crs, *dst_bounds)
        src_window = from_bounds(*src_bounds, transform=src.transform)

        data = src.read(
            window=src_window,
            boundless=True,
        )
        data = data.astype(np.float32)
        data[data == 0] = np.nan # replace nodata with nan

        src_transform = src.window_transform(src_window)
        src_profile['transform'] = src_transform # Save for future reprojection
        src_profile['height'] = data.shape[1]
        src_profile['width'] = data.shape[2]
    return data, src_profile

def reproject_raster(
    src: np.ndarray,
    src_profile: dict,
    dst_profile: dict
) -> np.ndarray:
    """Reproject a raster from source to destination profile.

    Args:
      src: Source raster data as a numpy array.
      src_profile: Profile of the source raster.
      dst_profile: Profile of the destination raster.

    Returns:
      A numpy array with shape (3, rows, cols) containing reprojected data.
    """
    dest = np.full((3, dst_profile['height'], dst_profile['width']),
                   np.nan, dtype=np.float32)
    reproject(
        source=src,
        destination=dest,
        src_transform=src_profile['transform'],
        src_crs=src_profile['crs'],
        dst_transform=dst_profile['transform'],
        dst_crs=dst_profile['crs'],
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
    cmip6_dir: str = "",
) -> Dict[str, np.ndarray]:
    """Load all daily forcings for a year and compute CU/tavg stacks.

    Returns dictionary with keys:
      tavg: (days, rows, cols)
      dayl: (days, rows, cols)
      cu:   (days, rows, cols)
    """
    daymet_files = list_daily_daymet_files(daymet_dir, year)
    if cmip6_dir != "":
        cmip6_files = list_daily_cmip6_files(cmip6_dir, year+86)
    dst_profile = get_target_grid(imagery_path, window)
    print('Reading using transform: ', dst_profile)
    dest_days: List[np.ndarray] = []
    for daymet_path in daymet_files:
        daymet_data, daymet_profile = read_dataset(
            path=daymet_path,
            dst_profile=dst_profile
        )
        # If CMIP6 data is provided reproject it to align with Daymet data,
        # and add the temperature deltas to the Daymet tmax/tmin bands.
        if cmip6_dir != "":
            cmip6_data, cmip6_profile = read_dataset(
                path=cmip6_dir,
                dst_profile=daymet_profile
            )
            cmip6_data = reproject_raster(
                src=cmip6_data,
                src_profile=cmip6_profile,
                dst_profile=daymet_profile
            )
            daymet_data[[0,1], :, :] = (daymet_data[[0,1], :, :]
                                        + cmip6_data[0, :, :])
        dest_day = reproject_raster(
            src=daymet_data,
            src_profile=daymet_profile,
            dst_profile=dst_profile
        )
        dest_days.append(dest_day)
    dest = np.stack(dest_days, axis=0)
    dest[:, [0,1], :, :] = dest[:, [0,1], :, :] * 120 / 65535 - 60 # unshrink tmax/tmin
    tavg = (dest[:, 0, :, :] + dest[:, 1, :, :]) / 2.0
    cu = np.cumsum(np.less(tavg, chill_threshold), axis=0).astype(np.float32)
    dayl = dest[:, 2, :, :]/86400.0

    tavg = tavg[31:,:,:]
    cu = cu[31:,:,:]
    dayl = dayl[31:,:,:]

    return tavg, dayl, cu
