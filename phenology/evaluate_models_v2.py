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

samples, skipped, trained_on_training_data = pheno_helpers.create_evaluation_samples(
   landcovers,
   output_x,
   output_y,
   sample_choices_path=f'/lustre/scratch5/cscholl/samples/{args.window}.json')


##########################################
# Load and rearrange data for computations
##########################################

eval_skipped = set(skipped)

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

# Pixels newly skipped here (as opposed to already being skipped due to
# having no landcover pixel-years at all) had evaluation samples that were
# entirely dropped as NaN. Fall back to evaluating those models on their
# full training data instead of dropping them.
nan_skipped = sorted(set(skipped) - eval_skipped)
if nan_skipped:
    print(f"Falling back to training data for {len(nan_skipped)} pixels "
          "with no valid evaluation samples")
    fallback_samples, _ = pheno_helpers.create_full_samples(
        landcovers, output_x, output_y)
    fallback_skip = [(i, j) for i in range(output_x) for j in range(output_y)
                     if (i, j) not in nan_skipped]
    fallback_data, _ = pheno_helpers.extract_forcings_and_targets(
        samples=fallback_samples,
        skipped=fallback_skip,
        output_x=output_x,
        output_y=output_y,
        num_years=num_years,
        window=window,
        daymet_path='/lustre/scratch5/cscholl/daymet',
        imagery_path='/lustre/scratch5/cscholl/modis'
    )
    for (i, j) in nan_skipped:
        key = f"{i}_{j}"
        if len(fallback_data[key]['tavg']) > 0:
            pixel_data[key] = fallback_data[key]
            skipped.remove((i, j))
            trained_on_training_data.add((i, j))

# Assemble final arrays by concatenating year chunks per pixel. Each model's
# forcings/target stay as their own (days, n_samples) array -- there is no
# padding to a shared max-sample size and no stacking into one large
# (n_models, days, max_samples) array. Keeping models separate here means
# evaluation below can move one model's data onto the device at a time,
# which bounds device memory use to a single model instead of to the whole
# window's worth of models at once.
print("Assembling per-model forcing arrays")
tavg_permodel = []
dayl_permodel = []
cu_permodel   = []
sos_permodel  = []
index = []
for i in range(output_x):
    for j in range(output_y):
        if (i, j) in skipped:
            continue
        d = pixel_data[f"{i}_{j}"]
        # 24 of (days, n_samples_year) -> (days, n_samples)
        tavg_permodel.append(np.concatenate(d['tavg'], axis=1))
        dayl_permodel.append(np.concatenate(d['dayl'], axis=1))
        cu_permodel.append(np.concatenate(d['cu'],   axis=1))
        sos_permodel.append(np.concatenate(d['sos'],  axis=1))
        index.append((i, j))

n_models = len(index)

# If one of these contains nan from the start, report as error
for m in range(n_models):
    if (np.any(np.isnan(tavg_permodel[m])) | np.any(np.isnan(dayl_permodel[m])) |
        np.any(np.isnan(cu_permodel[m])) | np.any(np.isnan(sos_permodel[m]))):
        print('Found nan forcings')
        sys.exit(1)

print('n_models', n_models)


##########################################
# Load model parameters and forcing stats
##########################################

with rio.open(f'/lustre/scratch5/cscholl/pheno_params/pheno_params_{args.window}.tif',
                'r') as src:
    params = src.read()
    profile = src.profile

params = params.reshape((6, output_models, 1))
valid_models = np.all(np.isfinite(params), axis=(0, 2))
params = params[:, valid_models, :]
assert params.shape[1] == n_models, f"Number of models in parameters file ({params.shape[1]}) does not match number of models in samples ({n_models})."

# Forcing normalization statistics (per-model tavg mean/std, dayl mean/std,
# cu min/max) computed at training time. These must be reused here so that
# evaluation data is transformed identically to the training data.
with rio.open(f'/lustre/scratch5/cscholl/forcings/forcings_{args.window}.tif',
                'r') as src:
    forcing_stats = src.read()

forcing_stats = forcing_stats.reshape((6, output_models, 1))
forcing_stats = forcing_stats[:, valid_models, :]
assert forcing_stats.shape[1] == n_models, f"Number of models in forcing stats file ({forcing_stats.shape[1]}) does not match number of models in samples ({n_models})."

# Per-model scalar parameters and forcing stats, so that neither the data
# assembled above nor these need to be batched across models.
params_permodel = [params[:, m, 0] for m in range(n_models)]
forcing_stats_permodel = [forcing_stats[:, m, 0] for m in range(n_models)]


##########################################
# Run Model Evaluation (one model at a time)
##########################################

# Output accumulators live on the CPU and are filled in as each model is
# evaluated, so GPU memory only ever has to hold the forcings/targets and
# parameters for a single model.
crps_save = np.full((output_x, output_y), np.nan, dtype=np.float32)
rmse_save = np.full((output_x, output_y), np.nan, dtype=np.float32)
trained_on_training_save = np.full((output_x, output_y), np.nan, dtype=np.float32)
for (i, j) in index:
    trained_on_training_save[i, j] = (i, j) in trained_on_training_data

print("Starting model evaluation")
with torch.no_grad():
    for m in range(n_models):
        i, j = index[m]

        tavg_mean, tavg_std, dayl_mean, dayl_std, cu_min, cu_max = forcing_stats_permodel[m]
        b_tavg, b_dayl, b_cu, b_const, kappa, lam = params_permodel[m]

        # Normalize this model's forcings using the training-time statistics
        # loaded above, rather than statistics computed from the evaluation
        # data itself.
        tavg_m = (tavg_permodel[m] - tavg_mean) / tavg_std
        dayl_m = (dayl_permodel[m] - dayl_mean) / dayl_std
        cu_m = (cu_permodel[m] - cu_min) / (cu_max - cu_min)
        # If cu doesn't change, it can be set to 0 without affecting evaluation.
        cu_m = np.where(np.isnan(cu_m), 0, cu_m)

        with torch.device(device):
            tavg_t = torch.tensor(tavg_m, dtype=dtype)
            dayl_t = torch.tensor(dayl_m, dtype=dtype)
            cu_t = torch.tensor(cu_m, dtype=dtype)
            sos_t = torch.tensor(sos_permodel[m], dtype=dtype)

            b_tavg_t = torch.tensor(b_tavg, dtype=dtype)
            b_dayl_t = torch.tensor(b_dayl, dtype=dtype)
            b_cu_t = torch.tensor(b_cu, dtype=dtype)
            b_const_t = torch.tensor(b_const, dtype=dtype)
            kappa_t = torch.tensor(kappa, dtype=dtype)
            lam_t = torch.tensor(lam, dtype=dtype)

            pred = pheno_helpers.make_prediction(
                tavg_t, dayl_t, cu_t,
                b_tavg_t, b_dayl_t, b_cu_t, b_const_t,
                kappa_t, lam_t,
                dtype
            )

            # We use the continuous ranked probability score (CRPS) as the
            # loss function, which is a common forecasting metric.
            ## Sum over days, then mean over this model's pixel-year samples
            ## to get a single CRPS value for the model.
            crps = torch.sum((pred - sos_t)**2, dim=0)
            crps = torch.mean(crps)

            # We also compute the root mean squared error (RMSE) of
            # start-of-season (SoS) day predictions. The predicted SoS day is
            # the first day on which the cumulative prediction probability
            # exceeds 50%, and the true SoS day is the first day on which the
            # (also cumulative) target exceeds 50%. Since both pred and sos
            # are non-decreasing step-like functions bounded in [0, 1], argmax
            # of the boolean crossing returns the first day index where the
            # 50% threshold is crossed.
            pred_day = torch.argmax((pred > 0.5).to(dtype=torch.int32), dim=0).to(dtype=dtype)
            true_day = torch.argmax((sos_t > 0.5).to(dtype=torch.int32), dim=0).to(dtype=dtype)
            rmse = torch.sqrt(torch.mean((pred_day - true_day)**2))

        crps_save[i, j] = crps.detach().cpu().item()
        rmse_save[i, j] = rmse.detach().cpu().item()

        # Release this model's data before moving on to the next one, and
        # periodically return freed memory to the allocator so that peak GPU
        # usage stays close to a single model's footprint.
        del tavg_t, dayl_t, cu_t, sos_t
        del b_tavg_t, b_dayl_t, b_cu_t, b_const_t, kappa_t, lam_t, pred
        if device == 'cuda' and (m % 50 == 0):
            torch.cuda.empty_cache()

        if m % 50 == 0:
            print(f'Evaluated model [{m+1}/{n_models}]')

print(f'Finished evaluating {n_models} models')


##########################################
# Save Evaluation Output
##########################################

eval_save = np.stack([crps_save, rmse_save, trained_on_training_save], axis=0)

profile['count'] = 3
with rio.Env():
    with rio.open(f'/lustre/scratch5/cscholl/pheno_eval/pheno_eval_{args.window}.tif',
                  'w', **profile) as dst:
        dst.write(eval_save)
        dst.set_band_description(1, 'crps')
        dst.set_band_description(2, 'rmse')
        dst.set_band_description(3, 'trained_on_training_data')
