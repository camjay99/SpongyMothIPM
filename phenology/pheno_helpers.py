import json

import numpy as np
import rasterio as rio
from rasterio.plot import show
from rasterio.windows import Window
import torch
import torch.masked as tmask

import load_daymet_forcing_v2

def load_landcovers(landcover_folder_path, num_years, window=None):
    """
    Load multiple years of landcover data into a single array. File names are \
    assumed to be in the format 'YYYY_MM_DD.tif'. 

    Parameters:
    - landcover_folder_path: str, path to the folder containing landcover raster files.
    - num_years: int, number of years of data to load.
    - window: Rasterio Window object or None, specifying the region of interest.
        If None, the entire raster is loaded.

    Returns:
    - landcover_data: np.ndarray (num_years, height, width).
    """

    print("Loading landcovers")
    landcovers = []
    for year in range(2001, 2001 + num_years):
        with rio.open(f"{landcover_folder_path}/{year}_01_01.tif") as src:
            if window is not None:
                landcovers.append(src.read(1, window=window, boundless=True))
            else:
                landcovers.append(src.read(1))
    landcovers = np.stack(landcovers, axis=0)
    
    return landcovers

def get_output_dimensions(landcover_data):
    """
    Get the output dimensions (height, width) and the size of the output model 
    array from the landcover data.

    Parameters:
    - landcover_data: np.ndarray (num_years, height, width).

    Returns:
    - output_x: int, number of pixels in the x-direction.
    - output_y: int, number of pixels in the y-direction.
    - output_models: int, number of models in the output.
    """

    output_x = landcover_data.shape[1]//10  # width
    output_y = landcover_data.shape[2]//10  # height
    return output_x, output_y, output_x * output_y

def create_samples(landcovers, output_x, output_y, save_path=None):
    """
    Create random sample of pixel-years to be used for model fitting. Each sample 
    will have coordinates relative to their output pixel.

    Parameters:
    - landcovers: np.ndarray (num_years, height, width).
    - output_x: int, number of output pixels in the x-direction.
    - output_y: int, number of output pixels in the y-direction.
    - save_path: str or None, path to save the samples dictionary. If None, samples 
        are not saved.

    Returns:
    - samples: dict, a dictionary of samples.
    """

    print("Creating samples")
    samples = {}
    sample_choices = {}
    skipped = []
    for i in range(output_x):
        for j in range(output_y):
            # Get subsample that is only this output pixel
            mask = landcovers[:,
                            10*i:10*(i+1),
                            10*j:10*(j+1)]
            # Get tuple of indices along each axis of only non-zero elements
            # e.g. ([1, 1, 4, 5, 6], <- years
            #       [0, 10, 13, 15, 15], <- rows
            #       [2, 3, 3, 3, 10]) <- columns
            nonzero = np.nonzero(mask)
            # If no elements to be sampled, skip this
            if len(nonzero[0]) == 0:
                skipped.append((i,j))
                continue
            # If less than 300 elements, add all to samples
            if len(nonzero[0]) < 300:
                # Iterate through each dimension and resize to 300 by repeating 
                # elements. This ensures that all samples have the same number 
                # of pixel-years, which is required for batching in model fitting.
                sample = tuple((np.resize(dim, 300) for dim in nonzero))
                samples[f"{i}_{j}"] = sample
                sample_choices[f"{i}_{j}"] = 'all'
            else:
                # Randomly choose indices for inclusion in sample without 
                # replacement. Random choice must be along # of indices, not 
                # number of dimensions in data.
                indices = np.random.choice(np.arange(nonzero[0].shape[0]), 
                                           300, 
                                           replace=False)
                # Get sample indices
                sample = tuple(dim[indices] for dim in nonzero)
                samples[f"{i}_{j}"] = sample
                sample_choices[f"{i}_{j}"] = indices.tolist()

    # Save sample choices for reproducibility and later evaluation.
    if save_path is not None:
        with open(save_path, 'w') as f:
            f.write(json.dumps(sample_choices))

    return samples, skipped

def create_evaluation_samples(landcovers, output_x, output_y, sample_choices_path):
    """
    Create evaluation samples based on previously saved sample choices.

    Parameters:
    - landcovers: np.ndarray (num_years, height, width).
    - output_x: int, number of output pixels in the x-direction.
    - output_y: int, number of output pixels in the y-direction.
    - sample_choices_path: str, path to the JSON file containing sample choices.

    Returns:
    - eval_samples: dict, a dictionary of evaluation samples.
    - skipped: list of tuples, list of (i,j) indices with no pixel-years.
    - trained_on_eval_data: set of tuples, (i,j) indices of models whose
        evaluation samples are drawn from their training data (either because
        all pixel-years were used in training, or because no held-out
        pixel-years remained).
    """

    print("Creating evaluation samples")

    # Load sample choices from JSON file
    with open(sample_choices_path, 'r') as f:
        sample_choices = json.load(f)

    samples = {}
    skipped = []
    trained_on_eval_data = set()
    for i in range(output_x):
        for j in range(output_y):
            # Get subsample that is only this output pixel
            mask = landcovers[:,
                            10*i:10*(i+1),
                            10*j:10*(j+1)]
            # Get tuple of indices along each axis of only non-zero elements
            # e.g. ([1, 1, 4, 5, 6], <- years
            #       [0, 10, 13, 15, 15], <- rows
            #       [2, 3, 3, 3, 10]) <- columns
            nonzero = np.nonzero(mask)
            # If no elements to be sampled, skip this
            if len(nonzero[0]) == 0:
                skipped.append((i,j))
                continue
            # If less than 300 elements, all pixel-years are used for evaluation.
            if sample_choices[f'{i}_{j}'] == 'all':
                samples[f'{i}_{j}'] = nonzero
                trained_on_eval_data.add((i, j))
            else:
                # Invert the list of indices
                indices = list(set(range(nonzero[0].shape[0]))
                                    .difference(sample_choices[f'{i}_{j}']))
                # If every pixel-year was used in training (e.g. the training
                # sample count equaled the number of available pixel-years),
                # there is no held-out data left for evaluation. Fall back to
                # evaluating on the full training data for this model instead.
                if len(indices) == 0:
                    samples[f'{i}_{j}'] = nonzero
                    trained_on_eval_data.add((i, j))
                else:
                    samples[f'{i}_{j}'] = tuple(dim[indices] for dim in nonzero)

    return samples, skipped, trained_on_eval_data

def create_full_samples(landcovers, output_x, output_y):
    """
    Build samples from all available pixel-years (i.e. the full training data,
    unfiltered by any train/evaluation split) for every output pixel.

    Used as a fallback source of evaluation data for a model whose held-out
    evaluation samples turn out to be entirely invalid (e.g. all NaN after
    filtering), so that the model can still be evaluated on its training data
    instead of being dropped.

    Parameters:
    - landcovers: np.ndarray (num_years, height, width).
    - output_x: int, number of output pixels in the x-direction.
    - output_y: int, number of output pixels in the y-direction.

    Returns:
    - samples: dict, a dictionary of samples.
    - skipped: list of tuples, list of (i,j) indices with no pixel-years.
    """

    samples = {}
    skipped = []
    for i in range(output_x):
        for j in range(output_y):
            mask = landcovers[:,
                            10*i:10*(i+1),
                            10*j:10*(j+1)]
            nonzero = np.nonzero(mask)
            if len(nonzero[0]) == 0:
                skipped.append((i, j))
                continue
            samples[f"{i}_{j}"] = nonzero

    return samples, skipped

def extract_forcings_and_targets(samples,
                                 skipped,
                                 output_x, 
                                 output_y, 
                                 num_years,
                                 window,
                                 daymet_path, 
                                 imagery_path):
    """
    Extract Daymet forcings (tmax, tmin, dayl, cu) and MODIS SoS for the given 
    samples.

    Parameters:
    - samples: dict, a dictionary of samples.
    - skipped: list of tuples, list of (i,j) indices to skip.
    - output_x: int, number of output pixels in the x-direction.
    - output_y: int, number of output pixels in the y-direction.
    - num_years: int, number of years of data to include.
    - window: Rasterio Window, the window to extract data from.
    - daymet_path: str, path to the Daymet raster file.
    - imagery_path: str, path to the imagery raster file.

    Returns:
    - forcings: dict, a dictionary containing the extracted forcings.
    """

    print("Loading forcings and sampling")

    # Accumulate per-pixel forcing lists, keyed by (i,j).
    # Each entry holds one array per year that had samples for that pixel.
    pixel_data = {
        f"{i}_{j}": {'tavg': [], 'dayl': [], 'cu': [], 'sos': []}
        for i in range(output_x) for j in range(output_y)
        if (i, j) not in skipped
    }

    for year_idx, year in enumerate(range(2001, 2001 + num_years)):
        print(f"Loading year {year}")
        tavg_yr, dayl_yr, cu_yr = load_daymet_forcing_v2.load_year_forcing(
            year=year,
            daymet_dir=daymet_path,
            imagery_path=f'{imagery_path}/{year}_01_01.tif',
            chill_threshold=5.0,
            window=window
        )
        with rio.open(f'{imagery_path}/{year}_01_01.tif') as src:
            sos_yr = src.read(boundless=True, window=window)

        for i in range(output_x):
            for j in range(output_y):
                if (i, j) in skipped:
                    continue

                # Filter sample indices to those belonging to this year
                year_mask = samples[f"{i}_{j}"][0] == year_idx
                if not year_mask.any():
                    continue

                sample_rows = samples[f"{i}_{j}"][1][year_mask]
                sample_cols = samples[f"{i}_{j}"][2][year_mask]
                sample_rc = (sample_rows, sample_cols)

                # Extract forcings for the sampled pixels within the 10x10 sub-tile
                # (days, height, width) -> (days, 10, 10)
                sl = (slice(None),)
                tavg_sub = tavg_yr[:, 10*i:10*(i+1), 10*j:10*(j+1)]
                dayl_sub = dayl_yr[:, 10*i:10*(i+1), 10*j:10*(j+1)]
                cu_sub   = cu_yr[:,  10*i:10*(i+1), 10*j:10*(j+1)]
                # (1, height, width) -> (1, 10, 10)
                sos_sub  = sos_yr[:, 10*i:10*(i+1), 10*j:10*(j+1)]

                # (days, 10, 10) -> (days, n_samples_year)
                tavg_pixyear = tavg_sub[sl + sample_rc]
                dayl_pixyear = dayl_sub[sl + sample_rc]
                cu_pixyear   = cu_sub[sl + sample_rc]

                # (1, 10, 10) -> (1, n_samples_year)
                sos_vals = sos_sub[sl + sample_rc]
                # (1, n_samples_year) -> (days, n_samples_year)
                days = np.arange(tavg_pixyear.shape[0]).reshape(-1, 1)
                sos_pixyear = (days >= sos_vals.reshape(1, -1)).astype(np.float32)

                # Identify samples with NaN values anywhere in their forcing/target
                # series and drop only those samples, preserving the rest of this
                # pixel/year's data (and all other pixel/year combos).
                nan_mask = (np.isnan(tavg_pixyear).any(axis=0) |
                            np.isnan(dayl_pixyear).any(axis=0) |
                            np.isnan(cu_pixyear).any(axis=0) |
                            np.isnan(sos_pixyear).any(axis=0))
                if nan_mask.any():
                    print(f"Found NaN values in forcings for pixel ({i}, {j}) in "
                          f"year {year}; dropping {nan_mask.sum()} of "
                          f"{nan_mask.shape[0]} samples.")
                    valid = ~nan_mask
                    tavg_pixyear = tavg_pixyear[:, valid]
                    dayl_pixyear = dayl_pixyear[:, valid]
                    cu_pixyear   = cu_pixyear[:, valid]
                    sos_pixyear  = sos_pixyear[:, valid]
                    if tavg_pixyear.shape[1] == 0:
                        continue

                # Add extracted data to other years data
                pixel_data[f"{i}_{j}"]['tavg'].append(tavg_pixyear)
                pixel_data[f"{i}_{j}"]['dayl'].append(dayl_pixyear)
                pixel_data[f"{i}_{j}"]['cu'].append(cu_pixyear)
                pixel_data[f"{i}_{j}"]['sos'].append(sos_pixyear)

    # Skip years where no samples were added.
    for i in range(output_x):
        for j in range(output_y):
            if (i,j) in skipped:
                continue
            if len(pixel_data[f"{i}_{j}"]['tavg']) == 0:
                skipped.append((i, j))
                print(f"Skipping pixel ({i}, {j}) because no valid samples were found across all years.")

    return pixel_data, skipped

def make_prediction(tavgs, 
                    dayls, 
                    cus, 
                    b_tavg, 
                    b_dayl, 
                    b_cu, 
                    b_const, 
                    kappa, 
                    lam,
                    dtype=torch.float32):
    """Run the forward pass for a single model.

    tavgs, dayls, cus: (days, n_samples) tensors for one model only.
    b_tavg, b_dayl, b_cu, b_const, kappa, lam: scalar tensors for that same model.
    """
    pss = []
    # (n_samples,)
    hs = torch.zeros((tavgs.shape[1],), dtype=dtype)
    ps = torch.zeros((tavgs.shape[1],), dtype=dtype)
    # Iterate across days, updating the state of the model for each day.
    for i in range(tavgs.shape[0]):
        forcings = tavgs[i,:]*b_tavg + dayls[i,:]*b_dayl + cus[i,:]*b_cu + b_const
        forcings = torch.maximum(forcings, torch.tensor(0, dtype=dtype))
        hs = hs + forcings*(1 - hs/100.)
        ps = (1 / (torch.exp(-lam*(100*kappa + hs)) + 1)) * (1 - ps) + ps
        pss.append(ps)
    # days x (n_samples,) -> (days, n_samples)
    pred = torch.stack(pss, dim=0)

    return pred