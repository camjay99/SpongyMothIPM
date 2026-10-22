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
import pheno_helpers

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

# Load landcovers into single data frame for generating pixel-year samples
landcovers = pheno_helpers.load_landcovers('/lustre/scratch5/cscholl/landcover', 
                              num_years, 
                              window)
if np.all(landcovers == 0):
    print("No models to evaluate in this region")
    sys.exit()

output_x, output_y, output_models = pheno_helpers.get_output_dimensions(landcovers)

samples, skipped = pheno_helpers.create_evaluation_samples(
   landcovers, 
   output_x, 
   output_y, 
   sample_choices_path=f'/lustre/scratch5/cscholl/samples/{args.window}.json')

total_models = output_models - len(skipped)


##########################################
# Load and rearrange data for computations
##########################################

pixel_data, skipped = pheno_helpers.extract_forcings_and_targets(
    samples=samples,
    skipped=skipped,
    output_x=output_x,
    output_y=output_y,
    num_years=num_years,
    window=window,
    daymet_path='/lustre/scratch5/cscholl/daymet',
    imagery_path='/lustre/scratch5/cscholl/modis'
)

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

##########################################
# Load model parameters and forcing stats
##########################################

# Save results
with rio.open(f'/lustre/scratch5/cscholl/pheno_params/pheno_params_{args.window}.tif',
                'r') as src:
    params = src.read()
    profile = src.profile

params = params.reshape((6, output_models, 1))
valid_models = np.all(np.isfinite(params), axis=(0, 2))
params = params[:, valid_models, :]
assert params.shape[1] == total_models, f"Number of models in parameters file ({total_models}) does not match number of models in samples ({params.shape[1]})."

# Forcing normalization statistics (per-model tavg mean/std, dayl mean/std,
# cu min/max) computed at training time. These must be reused here so that
# evaluation data is transformed identically to the training data.
with rio.open(f'/lustre/scratch5/cscholl/forcings/forcings_{args.window}.tif',
                'r') as src:
    forcing_stats = src.read()

forcing_stats = forcing_stats.reshape((6, output_models, 1))
forcing_stats = forcing_stats[:, valid_models, :]
assert forcing_stats.shape[1] == total_models, f"Number of models in forcing stats file ({total_models}) does not match number of models in samples ({forcing_stats.shape[1]})."

tavg_mean = forcing_stats[0].reshape(total_models, 1, 1)
tavg_std  = forcing_stats[1].reshape(total_models, 1, 1)
dayl_mean = forcing_stats[2].reshape(total_models, 1, 1)
dayl_std  = forcing_stats[3].reshape(total_models, 1, 1)
cu_min    = forcing_stats[4].reshape(total_models, 1, 1)
cu_max    = forcing_stats[5].reshape(total_models, 1, 1)

# Normalize data per model, using the training-time statistics loaded above
# rather than statistics computed from the evaluation data itself.
tavg = (tavg - tavg_mean) / tavg_std
dayl = (dayl - dayl_mean) / dayl_std
cu = (cu - cu_min) / (cu_max - cu_min)
cu = np.where(np.isnan(cu), 0, cu) # If cu doesn't change, it can be set to 0 without affecting model fitting.

print('tavg', tavg.shape)
print('dayl', dayl.shape)
print('cu', cu.shape)
print('mask', mask.shape)


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
# Run Model Evaluation
##########################################

with torch.device(device):
   with torch.no_grad():
        pred = pheno_helpers.make_prediction(
            tavg, dayl, cu, 
            b_tavg, b_dayl, b_cu, b_const, 
            kappa, lam,
            dtype
        )
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

        # We also compute the root mean squared error (RMSE) of start-of-season
        # (SoS) day predictions. The predicted SoS day is the first day on which
        # the cumulative prediction probability exceeds 50%, and the true SoS
        # day is the first day on which the (also cumulative) target exceeds 50%.
        # Since both pred and sos are non-decreasing step-like functions bounded
        # in [0, 1], argmax of the boolean crossing returns the first day index
        # where the 50% threshold is crossed.
        ## (days, n_models, n_samples) -> (1, n_models, n_samples)
        pred_day = torch.argmax((pred > 0.5), dim=0, keepdim=True).to(dtype=dtype)
        true_day = torch.argmax((sos > 0.5), dim=0, keepdim=True).to(dtype=dtype)
        sq_err = (pred_day - true_day)**2
        ## (1, n_models, n_samples) -> (1, n_models, 1)
        sq_err_masked = tmask.masked_tensor(sq_err, mask)
        rmse = torch.sqrt(torch.mean(sq_err_masked, dim=2, keepdim=True).get_data())


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

# Scatter crps and rmse into full arrays. This ensures that final arrays
# contain all pixels, even if a model wasn't fit for all of them.
crps_save = torch.full((1, output_models, 1), np.nan, dtype=dtype, device='cpu')
crps_save.scatter_(1, torch.tensor(index).reshape(1,-1,1), crps.cpu())
crps_save = crps_save.reshape((1, output_x, output_y))

rmse_save = torch.full((1, output_models, 1), np.nan, dtype=dtype, device='cpu')
rmse_save.scatter_(1, torch.tensor(index).reshape(1,-1,1), rmse.cpu())
rmse_save = rmse_save.reshape((1, output_x, output_y))

eval_save = torch.cat([crps_save, rmse_save], dim=0)

profile['count'] = 2
with rio.Env():
    with rio.open(f'/lustre/scratch5/cscholl/pheno_eval/pheno_eval_{args.window}.tif',
                  'w', **profile) as dst:
        dst.write(eval_save)
        dst.set_band_description(1, 'crps')
        dst.set_band_description(2, 'rmse')
