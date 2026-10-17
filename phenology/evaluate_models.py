##################################
# Imports
##################################

import argparse
import json
import sys

import numpy as np
import rasterio as rio
from rasterio.plot import show
from rasterio.windows import Window
import torch
import torch.masked as tmask

import load_daymet_forcing_v2

########################################
# Parse arguments for running on cluster
########################################

parser = argparse.ArgumentParser(
    description='Options for fitting phenology models to MODIS data.')

# Device to use for fitting. Options are "cpu" and "cuda". 
# Note that using cuda requires a GPU with sufficient memory to 
# hold the data and model parameters.
parser.add_argument('--device', '-d', type=str, action='store')

# Dtype to use in fitting. Using float32 will reduce both memory usage and
# precision. Using float64 will increase memory usage but may improve precision.
parser.add_argument('--dtype', '-t', type=str, action='store',
                    choices=['float32', 'float64'], default='float32')

# Number of years to include in fitting. Years included will start at 2001.
parser.add_argument('--num_years', '-y', type=int, action='store', 
                    default=24)

# Width and height of windows in MODIS pixels. Should be a multiple of 10.
parser.add_argument('--height', '-H', type=int, action='store',
                    default=100)
parser.add_argument('--width', '-W', type=int, action='store',
                    default=100)

# Window number to evaluate model for. Should be less than total number of windows.
parser.add_argument('--window', '-n', type=int, action='store',
                    default=0)

# Parse arguments provided to script
args = parser.parse_args()


##################################
# Set constants
##################################

# Device to run optimizing code
device=args.device 
if device == 'cuda':
   assert torch.cuda.is_available(), "CUDA resources not available when cuda device specified."

# Dtype of data
if (args.dtype == 'float32'): 
    dtype = torch.float32
elif (args.dtype == 'float64'):
    dtype = torch.float64

# Number of years in the analysis
num_years = args.num_years 
if args.num_years <= 0:
    raise ValueError("Number of years must be greater than 0.")
elif args.num_years > 24:
    raise ValueError("Number of years cannot be greater than 24, "
                     + "as data is only available from 2001 to 2024.")

# Create list of windows to loop through for model evaluation. 
windows = load_daymet_forcing_v2.create_window_grid(
   '/lustre/scratch5/cscholl/modis/2001_01_01.tif', args.height, args.width)
print('Number of windows: ', len(windows))
window = windows[args.window]
print('Using window: ', window)


##########################################
# Retrieve pixel-year evaluation samples
##########################################

print("Loading landcovers")
# Load landcovers into single data frame for generating pixel-year samples
landcovers = []
for year in range(2001, 2001 + num_years):
  with rio.open(f'/lustre/scratch5/cscholl/landcover/{year}_01_01.tif') as src:
    landcover = src.read(1, window=window, boundless=True)
    landcovers.append(landcover)
landcovers = np.stack(landcovers, axis=0)
if np.all(landcovers == 0):
    print("No models to evaluate in this region")
    sys.exit()

output_x = landcovers.shape[1]//10
output_y = landcovers.shape[2]//10
output_models = output_x*output_y

print("Creating evaluation samples")
# Get samples used for model fitting.
with open(f'/lustre/scratch5/cscholl/samples/{args.window}.json', 'r') as f:
    sample_choices = json.load(f)

samples = {}
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
    # If less than 300 elements, all pixel-years are used for evaluation.
    if sample_choices[f'{i}_{j}'] == 'all':
      samples[f'{i}_{j}'] = nonzero
    else:
       # Invert the list of indices
       indices = list(set(range(nonzero[0].shape[0]))
                        .difference(sample_choices[f'{i}_{j}']))
       samples[f'{i}_{j}'] = tuple(dim[indices] for dim in nonzero)

total_models = output_models - len(skipped)


##########################################
# Load and rearrange data for computations
##########################################

pixel_data = {
    f"{i}_{j}": {'tavg': [], 'dayl': [], 'cu': [], 'sos': []}
    for i in range(output_x) for j in range(output_y)
    if (i, j) not in skipped
}

for year_idx, year in enumerate(range(2001, 2001 + num_years)):
    print(f"Loading year {year}")
    tavg_yr, dayl_yr, cu_yr = load_daymet_forcing_v2.load_year_forcing(
        year=year,
        daymet_dir='/lustre/scratch5/cscholl/daymet',
        imagery_path=f'/lustre/scratch5/cscholl/modis/{year}_01_01.tif',
        chill_threshold=5.0,
        window=window
    )
    with rio.open(f'/lustre/scratch5/cscholl/modis/{year}_01_01.tif') as src:
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
            tavg_pixyear = tavg_sub[sl + sample_rc]  # (days, n_samples)
            dayl_pixyear = dayl_sub[sl + sample_rc]
            cu_pixyear   = cu_sub[sl + sample_rc]

            # (1, 10, 10) -> (1, n_samples_year)
            sos_vals = sos_sub[sl + sample_rc]
            # (1, n_samples_year) -> (days, n_samples_year)
            days = np.arange(tavg_pixyear.shape[0]).reshape(-1, 1)
            sos_pixyear = (days >= sos_vals.reshape(1, -1)).astype(np.float32)

            # Verify that there are no NaN values in the extracted data
            if (np.any(np.isnan(tavg_pixyear)) or
                np.any(np.isnan(dayl_pixyear)) or
                np.any(np.isnan(cu_pixyear)) or
                np.any(np.isnan(sos_pixyear))):
                print(f"Found NaN values in forcings for pixel ({i}, {j}) in year {year}.")
                skipped.append((i, j))
                pixel_data[f"{i}_{j}"] = None
                continue

            # Add extracted data to other years data
            pixel_data[f"{i}_{j}"]['tavg'].append(tavg_pixyear)
            pixel_data[f"{i}_{j}"]['dayl'].append(dayl_pixyear)
            pixel_data[f"{i}_{j}"]['cu'].append(cu_pixyear)
            pixel_data[f"{i}_{j}"]['sos'].append(sos_pixyear)

# Assemble final arrays by concatenating year chunks per pixel, then zero-padding
# the sample axis to a fixed size and stacking across pixels. The fixed size is
# the maximum possible number of pixel-year samples for a single sub-tile
# (100 pixels per sub-tile times the number of years of data), so padding never
# has to truncate real samples.
# Shape after stack: (n_models, days, max_samples).
print("Assembling forcing arrays")
max_samples = 100 * num_years

tavg_allpixyears = []
dayl_allpixyears = []
cu_allpixyears   = []
sos_allpixyears  = []
mask_allpixyears = []
for i in range(output_x):
    for j in range(output_y):
        if (i, j) in skipped:
            continue
        d = pixel_data[f"{i}_{j}"]
        # 24 of (days, n_samples_year) -> (days, n_samples)
        tavg_ij = np.concatenate(d['tavg'], axis=1)
        dayl_ij = np.concatenate(d['dayl'], axis=1)
        cu_ij   = np.concatenate(d['cu'],   axis=1)
        sos_ij  = np.concatenate(d['sos'],  axis=1)

        # Ensure that the number of samples does not exceed the maximum allowed
        # This should never happen.
        n_samples_ij = tavg_ij.shape[1]
        assert n_samples_ij <= max_samples, (
            f"Pixel {i}_{j} has {n_samples_ij} samples, exceeding max of "
            f"{max_samples} (100 * num_years)."
        )

        # Pad the arrays to the maximum number of samples
        pad_width = max_samples - n_samples_ij
        pad_spec = ((0, 0), (0, pad_width))
        tavg_allpixyears.append(np.pad(tavg_ij, pad_spec))
        dayl_allpixyears.append(np.pad(dayl_ij, pad_spec))
        cu_allpixyears.append(np.pad(cu_ij, pad_spec))
        sos_allpixyears.append(np.pad(sos_ij, pad_spec))

        # Create a boolean mask indicating which entries are valid (not padded)
        model_mask = np.zeros((1, max_samples), dtype=bool)
        model_mask[0, :n_samples_ij] = True
        mask_allpixyears.append(model_mask)

# n_models x (days, max_samples) -> (n_models, days, max_samples)
tavg = np.stack(tavg_allpixyears, axis=0) 
dayl = np.stack(dayl_allpixyears, axis=0)
cu   = np.stack(cu_allpixyears,   axis=0)
sos  = np.stack(sos_allpixyears,  axis=0)
# n_models x (1, max_samples) -> (n_models, 1, max_samples)
mask = np.stack(mask_allpixyears, axis=0) 

# If one of these contains nan from the start, report as error
if (np.any(np.isnan(tavg)) | np.any(np.isnan(dayl)) |
    np.any(np.isnan(cu)) | np.any(np.isnan(sos))):
    print('Found nan forcings')
    sys.exit(1)

# Normalize data per model. Normalization stats must ignore the zero-padded 
# entries, or they'd be biased towards zero. 
invalid_bc = np.broadcast_to(~mask, tavg.shape)

tavg_masked = np.ma.masked_array(tavg, mask=invalid_bc)
tavg = np.ma.getdata(
    (tavg_masked - tavg_masked.mean(axis=(1,2), keepdims=True))
    / tavg_masked.std(axis=(1,2), keepdims=True))

dayl_masked = np.ma.masked_array(dayl, mask=invalid_bc)
dayl = np.ma.getdata(
    (dayl_masked - dayl_masked.mean(axis=(1,2), keepdims=True))
    / dayl_masked.std(axis=(1,2), keepdims=True))

cu_masked = np.ma.masked_array(cu, mask=invalid_bc)
cu_min = cu_masked.min(axis=(1,2), keepdims=True)
cu_max = cu_masked.max(axis=(1,2), keepdims=True)
cu = np.ma.getdata((cu_masked - cu_min) / (cu_max - cu_min))
cu = np.where(np.isnan(cu), 0, cu) # If cu doesn't change, it can be set to 0 without affecting model fitting.

print('tavg', tavg.shape)
print('dayl', dayl.shape)
print('cu', cu.shape)
print('mask', mask.shape)

##########################################
# Load model parameters
##########################################

# Save results
with rio.open(f'/lustre/scratch5/cscholl/pheno_params/pheno_params_{args.window}.tif', 
                'r') as src:
    params = src.read()
    profile = src.profile

params = params.reshape((5, output_models, 1))
valid_models = ~np.any(np.isnan(params), axis=(0, 2))
params = params[:, valid_models, :]
assert params.shape[1] == total_models, f"Number of models in parameters file ({total_models}) does not match number of models in samples ({params.shape[1]})."


##########################################
# Move data to GPU
##########################################

# (n_models, n_days, max_samples) -> (n_days, n_models, max_samples)
with torch.device(device):
  tavg = torch.tensor(tavg.transpose(1,0,2), dtype=dtype)
  dayl = torch.tensor(dayl.transpose(1,0,2), dtype=dtype)
  cu = torch.tensor(cu.transpose(1,0,2), dtype=dtype)
  sos = torch.tensor(sos.transpose(1,0,2), dtype=dtype)
  mask = torch.tensor(mask.transpose(1,0,2), dtype=torch.bool)  

# (1, n_models, 1)
with torch.device(device):
    b_tavg = torch.tensor(params[[0], :, :], dtype=dtype, requires_grad=False)
    b_dayl = torch.tensor(params[[1], :, :], dtype=dtype, requires_grad=False)
    b_cu = torch.tensor(params[[2], :, :], dtype=dtype, requires_grad=False)
    b_const = torch.tensor(params[[3], :, :], dtype=dtype, requires_grad=False)
    kappa = torch.tensor(params[[4], :, :], dtype=dtype, requires_grad=False)
    lam = torch.tensor(params[[5], :, :], dtype=dtype, requires_grad=False)


##########################################
# Define Forward Pass
##########################################

def make_prediction(tavgs, dayls, cus, b_tavg, b_dayl, b_cu, b_const, lam, kappa):
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


##########################################
# Run Model Evaluation
##########################################

with torch.device(device):
   with torch.no_grad():
        pred = make_prediction(tavg, dayl, cu, b_tavg, b_dayl, b_cu, b_const, lam, kappa)
        # We use the continuous ranked probability score (CRPS) as the loss function
        # which is a common forecasting metric. 
        ## Sum over days to get total CRPS
        ## (days, n_models, n_samples) -> (1, n_models, n_samples)
        crps = torch.sum((pred - sos)**2, dim=0, keepdim=True)
        ## Use torch.masked to take mean over pixel-years to get average CRPS 
        ## for each pixel, excluding the zero-padded (invalid) pixel-year slots 
        ## from the average.
        ## (1, n_models, n_samples) -> (1, n_models, 1)
        crps_masked = tmask.masked_tensor(crps, mask)
        crps = torch.mean(crps_masked, dim=2, keepdim=True).get_data()


##########################################
# Save Evaluation Output
##########################################

# Convert output to raster format for saving
# Create index tensor for scattering
index = []
for i in range(output_x):
  for j in range(output_y):
    # Skip if no samples
    if (i,j) in skipped:
      print(f'Skipping {i}, {j}')
      continue
    index.append(i*output_y + j)

# Scatter crps into full array. This ensures that final array
# contains all pixels, even if a model wasn't fit for all of them.
crps_save = torch.full((1, output_models, 1), np.nan, dtype=dtype, device='cpu')
crps_save.scatter_(1, torch.tensor(index).reshape(1,-1,1), crps.cpu())
crps_save = crps_save.reshape((1, output_x, output_y))

profile['count'] = 1
with rio.Env():
    with rio.open(f'/lustre/scratch5/cscholl/pheno_eval/pheno_eval_{args.window}.tif', 
                  'w', **profile) as dst:
        dst.write(crps_save)
