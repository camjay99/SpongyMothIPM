##################################
# Imports
##################################

import argparse
import os
import sys

import numpy as np
import rasterio as rio
from rasterio.windows import Window
import torch

import load_daymet_forcing_v2
import pheno_helpers

########################################
# Parse arguments for running on cluster
########################################

parser = argparse.ArgumentParser(
    description='Options for forecasting phenology under future climate conditions.')

# Device to use for prediction. Options are "cpu" and "cuda".
parser.add_argument('--device', '-d', type=str, action='store', default='cpu')

# Dtype to use in prediction.
parser.add_argument('--dtype', '-t', type=str, action='store',
                    choices=['float32', 'float64'], default='float32')

# Width and height of windows in MODIS pixels. Should be a multiple of 10, and
# must match the values used when fitting the models for this window.
parser.add_argument('--height', '-H', type=int, action='store', default=100)
parser.add_argument('--width', '-W', type=int, action='store', default=100)

# Window number to forecast. Should be less than the total number of windows.
parser.add_argument('--window', '-n', type=int, action='store', default=0)

# First future year to forecast, and number of future years to forecast.
parser.add_argument('--start_year', '-s', type=int, action='store', default=2025)
parser.add_argument('--num_future_years', '-f', type=int, action='store', default=1)

# Directory containing weekly CMIP6 anomaly rasters for the scenario of
# interest, used to adjust Daymet tmax/tmin per load_daymet_forcing_v2.
parser.add_argument('--cmip6_dir', '-c', type=str, action='store', required=True)

# Chilling threshold (degrees C) used to compute cumulative chilling units.
# Must match the threshold used when fitting the models.
parser.add_argument('--chill_threshold', type=float, action='store', default=5.0)

args = parser.parse_args()


##################################
# Set constants
##################################

device = args.device
if device == 'cuda':
    assert torch.cuda.is_available(), "CUDA resources not available when cuda device specified."

if args.dtype == 'float32':
    dtype = torch.float32
elif args.dtype == 'float64':
    dtype = torch.float64

if args.num_future_years <= 0:
    raise ValueError("Number of future years must be greater than 0.")

daymet_path = '/lustre/scratch5/cscholl/daymet'
imagery_path = '/lustre/scratch5/cscholl/modis'
params_path = f'/lustre/scratch5/cscholl/pheno_params/pheno_params_{args.window}.tif'
forcings_path = f'/lustre/scratch5/cscholl/forcings/forcings_{args.window}.tif'

# Name the output directory after the CMIP6 scenario directory so that
# forecasts for different scenarios don't collide.
scenario_name = os.path.basename(os.path.normpath(args.cmip6_dir))
output_dir = f'/lustre/scratch5/cscholl/pheno_future/{scenario_name}'


# Scale factor used to pack [0, 1] cumulative probabilities into uint16.
PROB_SCALE = 65535
NODATA = 0

windows = load_daymet_forcing_v2.create_window_grid(
    f'{imagery_path}/2001_01_01.tif', args.height, args.width)
print('Number of windows: ', len(windows))
window = windows[args.window]
print('Using window: ', window)


##########################################
# Load fitted model parameters and forcing
# normalization statistics
##########################################

with rio.open(params_path, 'r') as src:
    params = src.read()
    profile = src.profile

# Output grid dimensions come directly from the fitted-parameter raster,
# which was written at one pixel per output model (10x10 MODIS pixels).
output_x = params.shape[1]
output_y = params.shape[2]
output_models = output_x * output_y

params = params.reshape((6, output_models))
valid_models = np.all(np.isfinite(params), axis=0)
if not np.any(valid_models):
    print("No fitted models to forecast for this window")
    sys.exit()
params = params[:, valid_models]

with rio.open(forcings_path, 'r') as src:
    forcing_stats = src.read()
forcing_stats = forcing_stats.reshape((6, output_models))
forcing_stats = forcing_stats[:, valid_models]

flat_coords = [(i, j) for i in range(output_x) for j in range(output_y)]
index = [coord for coord, valid in zip(flat_coords, valid_models) if valid]
n_models = len(index)
print('n_models', n_models)

# Per-model scalar parameters and forcing stats, so that neither the
# forecast data nor these need to be batched across models.
params_permodel = [params[:, m] for m in range(n_models)]
forcing_stats_permodel = [forcing_stats[:, m] for m in range(n_models)]


##########################################
# Run forecast (one model and year at a time)
##########################################

os.makedirs(output_dir, exist_ok=True)

print("Starting forecast")
with torch.no_grad():
    for year in range(args.start_year, args.start_year + args.num_future_years):
        print(f'Forecasting year {year}')

        # Load full-resolution forcings for this future year, adjusted by the
        # CMIP6 scenario anomalies. The 2001 MODIS raster is reused purely as
        # a target grid template (shape/transform), since no future MODIS
        # imagery exists.
        tavg_yr, dayl_yr, cu_yr = load_daymet_forcing_v2.load_year_forcing(
            year=year,
            daymet_dir=daymet_path,
            imagery_path=f'{imagery_path}/2001_01_01.tif',
            chill_threshold=args.chill_threshold,
            window=window,
            cmip6_dir=args.cmip6_dir,
        )
        n_days = tavg_yr.shape[0]

        # Output accumulator lives on the CPU and is filled in as each model
        # is forecast, so GPU memory only ever has to hold one model's worth
        # of forcings and parameters at a time.
        forecast_save = np.full((n_days, output_x, output_y), NODATA, dtype=np.int16)

        for m in range(n_models):
            i, j = index[m]

            # Average full-resolution forcings over this output pixel's 10x10
            # block of MODIS pixels to get a single daily forcing series,
            # matching the spatial support the model was fit at.
            tavg_block = np.nanmean(
                tavg_yr[:, 10*i:10*(i+1), 10*j:10*(j+1)], axis=(1, 2))
            dayl_block = np.nanmean(
                dayl_yr[:, 10*i:10*(i+1), 10*j:10*(j+1)], axis=(1, 2))
            cu_block = np.nanmean(
                cu_yr[:, 10*i:10*(i+1), 10*j:10*(j+1)], axis=(1, 2))

            if np.any(np.isnan(tavg_block)) or np.any(np.isnan(dayl_block)):
                print(f'Model ({i}, {j}): no valid forcings for {year}, skipping')
                continue

            tavg_mean, tavg_std, dayl_mean, dayl_std, cu_min, cu_max = \
                forcing_stats_permodel[m]
            b_tavg, b_dayl, b_cu, b_const, kappa, lam = params_permodel[m]

            # Normalize using the training-time statistics loaded above, so
            # the forecast data is transformed identically to the training
            # and evaluation data.
            tavg_n = (tavg_block - tavg_mean) / tavg_std
            dayl_n = (dayl_block - dayl_mean) / dayl_std
            cu_n = (cu_block - cu_min) / (cu_max - cu_min)
            # If cu doesn't change, it can be set to 0 without affecting prediction.
            cu_n = np.where(np.isnan(cu_n), 0, cu_n)

            with torch.device(device):
                # Reshape to (days, 1) -- a single "sample" per model.
                tavg_t = torch.tensor(tavg_n.reshape(-1, 1), dtype=dtype)
                dayl_t = torch.tensor(dayl_n.reshape(-1, 1), dtype=dtype)
                cu_t = torch.tensor(cu_n.reshape(-1, 1), dtype=dtype)

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

            # (days, 1) -> (days,) cumulative probability of budburst.
            pred = torch.clamp(pred.squeeze(1), 0, 1)
            pred_np = pred.detach().cpu().numpy()
            forecast_save[:, i, j] = np.round(pred_np * PROB_SCALE).astype(np.uint16)

            del tavg_t, dayl_t, cu_t
            del b_tavg_t, b_dayl_t, b_cu_t, b_const_t, kappa_t, lam_t, pred
            if device == 'cuda' and (m % 50 == 0):
                torch.cuda.empty_cache()

            if m % 50 == 0:
                print(f'Forecast model [{m+1}/{n_models}]')

        #####################################
        # Save forecast output for this year
        #####################################

        year_profile = profile.copy()
        year_profile['count'] = n_days
        year_profile['dtype'] = forecast_save.dtype
        year_profile['nodata'] = NODATA

        out_path = f'{output_dir}/pheno_future_{args.window}_{year}.tif'
        with rio.Env():
            with rio.open(out_path, 'w', **year_profile) as dst:
                dst.write(forecast_save)
                for day in range(n_days):
                    dst.set_band_description(day + 1, f'day_{day}')

        print(f'Saved forecast for {year} to {out_path}')

print(f'Finished forecasting {n_models} models for {args.num_future_years} years')
