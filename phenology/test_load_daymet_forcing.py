from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio as rio
from rasterio.transform import from_origin

from load_daymet_forcing_v2 import (
    get_target_grid,
    list_daily_cmip6_files,
    list_daily_daymet_files,
    read_dataset,
    reproject_raster,
)


def _write_raster(path: Path, data: np.ndarray, transform, crs="EPSG:4326") -> None:
    """Create pseudo-raster file for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with rio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)

def _write_rasters(
    base_path: Path, 
    filenames: list[str],
    data_list: list[np.ndarray], 
    transform,
    crs="EPSG:4326"
) -> None:
    """Create multiple pseudo-raster files for testing."""
    for filename, data in zip(filenames, data_list):
        _write_raster(base_path / filename, data, transform, crs=crs)


def test_daily_file_lists_align_temporally(tmp_path: Path) -> None:
    daymet_dir = tmp_path / "daymet"
    cmip6_dir = tmp_path / "cmip6"

    # Create 6 Daymet files and 3 CMIP6 files covering the same time period 
    # (Dec 30 - Jan 4) with appropriate naming. Files are filed with arbitray data.
    daymet_filenames = [
        "20231230.tif",
        "20231231.tif",
        "20240101.tif",
        "20240102.tif",
        "20240103.tif",
        "20240104.tif"
    ]
    _write_rasters(
        base_path=daymet_dir, 
        filenames=daymet_filenames, 
        data_list=[np.ones((3, 2, 2), dtype=np.uint16)] * len(daymet_filenames), 
        transform=from_origin(0, 2, 1, 1)
    )

    cmip6_filenames = [
        "ACCESS-CM2_ssp245_2023_200_201.tif",
        "ACCESS-CM2_ssp245_2023_364_365.tif",
        "ACCESS-CM2_ssp245_2024_1_2.tif",
        "ACCESS-CM2_ssp245_2024_3_4.tif",
    ]
    _write_rasters(
        base_path=cmip6_dir, 
        filenames=cmip6_filenames, 
        data_list=[np.ones((3, 2, 2), dtype=np.uint16)] * len(cmip6_filenames), 
        transform=from_origin(0, 2, 1, 1)
    )

    # Run loading functions to get listed files
    listed_daymet = list_daily_daymet_files(str(daymet_dir), 2024)
    listed_cmip6 = list_daily_cmip6_files(str(cmip6_dir), 2024)

    # Files should be listed in the correct order 
    # December of previous year followed by January-September of current year
    assert listed_daymet == [str(daymet_dir / path) for path in daymet_filenames]
    assert listed_cmip6 == [str(cmip6_dir / path) for path in cmip6_filenames[1:] for i in [0,1]]
    # Cmip6 anomalies should be replicated for each day they cover, 
    # so there should be 6 daily files for the 3 weekly files
    assert len(listed_daymet) == len(listed_cmip6) == 6

@pytest.mark.parametrize('imagery_size, imagery_offset, imagery_crs, daymet_size, daymet_offset, daymet_crs, cmip6_size, cmip6_offset, cmip6_crs', 
                         [((3, 4), (0, 0, 1, 1), "EPSG:4326", (3, 4), (0, 0, 1, 1), "EPSG:4326", (3, 4), (0, 0, 1, 1), "EPSG:4326"),
                          ((400, 300), (194400, 4813800, 459, 574), "EPSG:26915", (404, 515), (-97.165, 43.785, 0.01, 0.005), "EPSG:4326", (21, 25), (-99.045, 44.745, 0.23, 0.123), "EPSG:4326")])
def test_read_and_reproject_outputs_matching_shapes(
    tmp_path: Path,
    imagery_size: tuple[int, int],
    imagery_offset: tuple[int, int, int, int],
    imagery_crs: str,
    daymet_size: tuple[int, int],
    daymet_offset: tuple[int, int, int, int],
    daymet_crs: str,
    cmip6_size: tuple[int, int],
    cmip6_offset: tuple[int, int, int, int],
    cmip6_crs: str
) -> None:
    imagery_path = tmp_path / "imagery.tif"
    daymet_path = tmp_path / "daymet.tif"
    cmip6_path = tmp_path / "cmip6.tif"

    # Create test imagery, Daymet, and CMIP6 files with different sizes,  
    # offsets, and crs but overlapping spatial extents.
    imagery_data = np.zeros((3, *imagery_size), dtype=np.uint16)
    daymet_data = np.arange(3 * daymet_size[0] * daymet_size[1], dtype=np.uint16).reshape(3, *daymet_size)
    cmip6_data = np.arange(3 * cmip6_size[0] * cmip6_size[1], dtype=np.uint16).reshape(3, *cmip6_size)

    imagery_transform = from_origin(*imagery_offset)
    daymet_transform = from_origin(*daymet_offset)
    cmip6_transform = from_origin(*cmip6_offset)

    _write_raster(imagery_path, imagery_data, imagery_transform, imagery_crs)
    _write_raster(daymet_path, daymet_data, daymet_transform, daymet_crs)
    _write_raster(cmip6_path, cmip6_data, cmip6_transform, cmip6_crs)

    # Test that reading and reprojecting the Daymet and CMIP6 files to the imagery grid
    # results in arrays with the same shape and dtype as the imagery.
    dst_profile = get_target_grid(str(imagery_path), None)

    daymet_read, daymet_profile = read_dataset(str(daymet_path), dst_profile)
    cmip6_read, cmip6_profile = read_dataset(str(cmip6_path), dst_profile)

    daymet_reprojected = reproject_raster(daymet_read, daymet_profile, dst_profile)
    cmip6_reprojected = reproject_raster(cmip6_read, cmip6_profile, dst_profile)

    assert daymet_reprojected.shape == cmip6_reprojected.shape == (3, *imagery_size)
    assert daymet_reprojected.dtype == cmip6_reprojected.dtype == np.float32

@pytest.mark.parametrize('imagery_size, imagery_offset, imagery_crs, daymet_size, daymet_offset, daymet_crs, cmip6_size, cmip6_offset, cmip6_crs', 
                         [((3, 4), (0, 0, 1, 1), "EPSG:4326", (3, 4), (0, 0, 1, 1), "EPSG:4326", (3, 4), (0, 0, 1, 1), "EPSG:4326"),
                          ((400, 300), (194400, 4813800, 459, 574), "EPSG:26915", (404, 515), (-97.165, 43.785, 0.01, 0.005), "EPSG:4326", (21, 25), (-99.045, 44.745, 0.23, 0.123), "EPSG:4326")])
def test_anomaly_increment(
    tmp_path: Path,
    imagery_size: tuple[int, int],
    imagery_offset: tuple[int, int, int, int],
    imagery_crs: str,
    daymet_size: tuple[int, int],
    daymet_offset: tuple[int, int, int, int],
    daymet_crs: str,
    cmip6_size: tuple[int, int],
    cmip6_offset: tuple[int, int, int, int],
    cmip6_crs: str
) -> None:
    imagery_path = tmp_path / "imagery.tif"
    daymet_path = tmp_path / "daymet.tif"
    cmip6_path = tmp_path / "cmip6.tif"

    # Create test imagery, Daymet, and CMIP6 files with different sizes,  
    # offsets, and crs but overlapping spatial extents.
    imagery_data = np.zeros((3, *imagery_size), dtype=np.uint16)
    daymet_data = np.arange(3 * daymet_size[0] * daymet_size[1], dtype=np.uint16).reshape(3, *daymet_size)
    cmip6_data = np.arange(3 * cmip6_size[0] * cmip6_size[1], dtype=np.uint16).reshape(3, *cmip6_size)

    imagery_transform = from_origin(*imagery_offset)
    daymet_transform = from_origin(*daymet_offset)
    cmip6_transform = from_origin(*cmip6_offset)

    _write_raster(imagery_path, imagery_data, imagery_transform, imagery_crs)
    _write_raster(daymet_path, daymet_data, daymet_transform, daymet_crs)
    _write_raster(cmip6_path, cmip6_data, cmip6_transform, cmip6_crs)

    # Test that reading and reprojecting the Daymet and CMIP6 files to the imagery grid
    # results in arrays with the same shape and dtype as the imagery.
    dst_profile = get_target_grid(str(imagery_path), None)

    daymet_read, daymet_profile = read_dataset(str(daymet_path), dst_profile)
    cmip6_read, cmip6_profile = read_dataset(str(cmip6_path), dst_profile)
    
    cmip6_reprojected = reproject_raster(cmip6_read, cmip6_profile, daymet_profile)
    assert daymet_read.shape == cmip6_reprojected.shape

    daymet_read = daymet_read + cmip6_reprojected
    daymet_reprojected = reproject_raster(daymet_read, daymet_profile, dst_profile)

    assert daymet_reprojected.shape == (3, *imagery_size)