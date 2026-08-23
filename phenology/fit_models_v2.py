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


########################################
# Parse arguments for running on cluster
########################################

parser = argparse.ArgumentParser(
    description='Options for fitting phenology models to MODIS data.')

# Device to use for fitting. Options are "cpu" and "cuda".
# Note that using cuda requires a GPU with sufficient memory to
# hold the data and model parameters.
parser.add_argument('--device', '-d', type=str, default='cpu')

# Dtype to use in fitting. Using float32 will reduce both memory usage and
# precision. Using float64 will increase memory usage but may improve precision.
parser.add_argument('--dtype', '-t', type=str,
                    choices=['float32', 'float64'], default='float32')

# Number of years to include in fitting. Years included will start at 2001.
parser.add_argument('--num_years', '-y', type=int, default=24)

# Width and height of windows in MODIS pixels. Should be a multiple of 10.
parser.add_argument('--height', '-H', type=int, default=100)
parser.add_argument('--width', '-W', type=int, default=100)

# Window number to fit model for. Should be less than total number of windows.
parser.add_argument('--window', '-n', type=int, default=0)

# Number of epochs for model fitting. More epochs will increase runtime but may improve fit.
parser.add_argument('--num_epochs', '-e', type=int, default=50)

# Parse arguments provided to script
args = parser.parse_args()


##################################
# Set constants
##################################

# Device to run optimizing code
device = args.device
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
    raise ValueError("Number of years cannot be greater than 24, as data is only available from 2001 to 2024.")

# Number of epochs for model fitting
num_epochs = args.num_epochs
if args.num_epochs <= 0:
    raise ValueError("Number of epochs must be greater than 0.")

# Create list of windows to loop through for model fitting.
windows = load_daymet_forcing_v2.create_window_grid(
   '/lustre/scratch5/cscholl/modis/2001_01_01.tif', args.height, args.width)
print('Number of windows: ', len(windows))
window = windows[args.window]
print('Using window: ', args.window, window)

##################################
# Determine pixel-year samples
##################################

print("Loading landcovers")
# Load landcovers into single data frame for generating pixel-year samples
landcovers = []
for year in range(2001, 2001 + num_years):
  with rio.open(f'/lustre/scratch5/cscholl/landcover/{year}_01_01.tif') as src:
    landcover = src.read(1, window=window, boundless=True)
    landcovers.append(landcover)
landcovers = np.stack(landcovers, axis=0)
if np.all(landcovers == 0):
    print("No models to fit in this region")
    sys.exit()

output_x = landcovers.shape[1]//10
output_y = landcovers.shape[2]//10
output_models = output_x*output_y

print("Creating samples")
# Create random sample of pixel-years to be used for model fitting
# Each sample will have coordinates relative to the output pixel they are for
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
      # Iterate through each dimension and resize to 300 by repeating elements.
      # This ensures that all samples have the same number of pixel-years,
      # which is required for batching in model fitting.
      sample = tuple((np.resize(dim, 300) for dim in nonzero))
      samples[f"{i}_{j}"] = sample
      sample_choices[f"{i}_{j}"] = 'all'
    else:
      # Randomly choose indices for inclusion in sample without replacement
      # Random choice must be along # of indices, not number of dimensions in data.
      indices = np.random.choice(np.arange(nonzero[0].shape[0]), 300, replace=False)
      # Get sample indices
      sample = tuple(dim[indices] for dim in nonzero)
      samples[f"{i}_{j}"] = sample
      sample_choices[f"{i}_{j}"] = indices.tolist()

# Save sample choices for reproducibility and later evaluation.
with open(f'/lustre/scratch5/cscholl/samples/{args.window}.json', 'w') as f:
    f.write(json.dumps(sample_choices))

total_models = output_models - len(skipped)


##########################################
# Load forcings one year at a time and
# extract samples immediately into a dict
##########################################

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
            sl = (slice(None),)
            tavg_sub = tavg_yr[:, 10*i:10*(i+1), 10*j:10*(j+1)]
            dayl_sub = dayl_yr[:, 10*i:10*(i+1), 10*j:10*(j+1)]
            cu_sub   = cu_yr[:,  10*i:10*(i+1), 10*j:10*(j+1)]
            sos_sub  = sos_yr[:, 10*i:10*(i+1), 10*j:10*(j+1)]

            tavg_pixyear = tavg_sub[sl + sample_rc]  # (days, n_samples)
            dayl_pixyear = dayl_sub[sl + sample_rc]
            cu_pixyear   = cu_sub[sl + sample_rc]

            sos_vals = sos_sub[sl + sample_rc]  # (1, n_samples)
            days = np.arange(tavg_pixyear.shape[0]).reshape(-1, 1)
            sos_pixyear = (days >= sos_vals.reshape(1, -1)).astype(np.float32)

            pixel_data[f"{i}_{j}"]['tavg'].append(tavg_pixyear)
            pixel_data[f"{i}_{j}"]['dayl'].append(dayl_pixyear)
            pixel_data[f"{i}_{j}"]['cu'].append(cu_pixyear)
            pixel_data[f"{i}_{j}"]['sos'].append(sos_pixyear)

# Assemble final arrays by concatenating year chunks per pixel, then stacking
# across pixels. Shape after stack: (n_models, days, n_samples).
print("Assembling forcing arrays")
tavg_allpixyears = []
dayl_allpixyears = []
cu_allpixyears   = []
sos_allpixyears  = []
for i in range(output_x):
    for j in range(output_y):
        if (i, j) in skipped:
            continue
        d = pixel_data[f"{i}_{j}"]
        tavg_allpixyears.append(np.concatenate(d['tavg'], axis=1))
        dayl_allpixyears.append(np.concatenate(d['dayl'], axis=1))
        cu_allpixyears.append(np.concatenate(d['cu'],   axis=1))
        sos_allpixyears.append(np.concatenate(d['sos'],  axis=1))

tavg = np.stack(tavg_allpixyears, axis=0)  # (n_models, days, n_samples)
dayl = np.stack(dayl_allpixyears, axis=0)
cu   = np.stack(cu_allpixyears,   axis=0)
sos  = np.stack(sos_allpixyears,  axis=0)

# If one of these are nan from the start, report as error
if (np.any(np.isnan(tavg)) | np.any(np.isnan(dayl)) |
    np.any(np.isnan(cu)) | np.any(np.isnan(sos))):
    print('Found nan forcings')
    sys.exit(1)

tavg = (tavg - tavg.mean(axis=(0,2),keepdims=True)) / tavg.std(axis=(0,2),keepdims=True)
dayl = (dayl - dayl.mean(axis=(0,2),keepdims=True)) / dayl.std(axis=(0,2),keepdims=True)
cu = (cu - cu.min(axis=(0,2),keepdims=True)) / (cu.max(axis=(0,2),keepdims=True) - cu.min(axis=(0,2),keepdims=True))

print('tavg', tavg.shape)
print('dayl', dayl.shape)
print('cu', cu.shape)

##########################################
# Move data to GPU
##########################################

# Tensors are organized as:
# dim0 - days
# dim1 - models
# dim2 - samples (pixel-years)
with torch.device(device):
  tavg = torch.tensor(tavg.transpose(1,0,2), dtype=dtype)
  dayl = torch.tensor(dayl.transpose(1,0,2), dtype=dtype)
  cu   = torch.tensor(cu.transpose(1,0,2),   dtype=dtype)
  sos  = torch.tensor(sos.transpose(1,0,2),  dtype=dtype)

def random_init_params(total_models, device, dtype):
    with torch.device(device):
        b_tavg = (torch.rand((1,total_models,1), dtype=dtype)*0.1 + 0.95).requires_grad_()
        b_dayl = (torch.rand((1,total_models,1), dtype=dtype)*0.1 + 0.95).requires_grad_()
        b_cu   = (torch.rand((1,total_models,1), dtype=dtype)*0.01 + 0.095).requires_grad_()
        kappa  = (torch.rand((1,total_models,1), dtype=dtype)*0.1 - 8.05).requires_grad_()
        lam    = (torch.rand((1,total_models,1), dtype=dtype)*0.01 - 0.095).requires_grad_()
    return b_tavg, b_dayl, b_cu, kappa, lam
b_tavg, b_dayl, b_cu, kappa, lam = random_init_params(total_models, device, dtype)


##########################################
# Define Forward Pass
##########################################

def make_prediction(tavgs, dayls, cus, b_tavg, b_dayl, b_cu, lam, kappa):
    pss = []
    hs = torch.zeros((tavgs.shape[1], tavgs.shape[2]))
    ps = torch.zeros((tavgs.shape[1], tavgs.shape[2]))
    for i in range(tavgs.shape[0]):
        forcings = tavgs[i,:,:]*b_tavg + dayls[i,:,:]*b_dayl + cus[i,:,:]*b_cu
        forcings = torch.maximum(forcings, torch.tensor(0, dtype=dtype))
        hs = hs + forcings*(1 - hs/100.)
        ps = (1 / (torch.exp(-(kappa + lam*hs)) + 1)) * (1 - ps) + ps
        pss.append(ps)
    pred = torch.concat(pss, dim=0)

    return pred


##########################################
# Run Model Training
##########################################

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < (self.min_validation_loss - self.min_delta):
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

print("Starting model fitting")
with torch.device(device):
  report_freq = 5
  #optimizer = torch.optim.LBFGS([b_tavg, b_dayl, b_cu, kappa, lam], lr=1,
  #                              history_size=400, max_iter=20, line_search_fn='strong_wolfe')
  es = EarlyStopper(patience=10, min_delta=0.001)
  # Closure capturing forward/backward passes is required for LBFGS optimization
  def training_run(optimizer):
    # b_tavg.grad = None
    # b_dayl.grad = None
    # b_cu.grad = None
    # kappa.grad = None
    # lam.grad = None
    optimizer.zero_grad()

    pred = make_prediction(tavg, dayl, cu, b_tavg, b_dayl, b_cu, lam, kappa)

    # We use the continuous ranked probability score (CRPS) as the loss function
    # which is a common forecasting metric.
    loss = torch.sum((pred - sos)**2)
    loss.backward()

    if torch.any(torch.isnan(loss)):
        print("Encountered NaN loss")

    # Replace nan grads so fitting may continue
    b_tavg.grad[torch.isnan(b_tavg.grad)] = 0
    b_dayl.grad[torch.isnan(b_dayl.grad)] = 0
    b_cu.grad[torch.isnan(b_cu.grad)] = 0
    kappa.grad[torch.isnan(kappa.grad)] = 0
    lam.grad[torch.isnan(lam.grad)] = 0

    return loss
  retry = 3
  fit = False
  while retry > 0 and not fit:
    # Run a course pass with Adam optimizer to find a good starting point for L-BFGS optimization
    opt_adam = torch.optim.Adam([b_tavg, b_dayl, b_cu, kappa, lam], lr=0.01)
    for epoch in range(500):
        try:
            loss = training_run(opt_adam)
            opt_adam.step()
        except:
            print("Encountered error during Adam optimization, retrying with new random initialization.")
            retry -= 1
            b_tavg, b_dayl, b_cu, kappa, lam = random_init_params(total_models, device, dtype)
            break

        if epoch % report_freq == 0:
            print(f'Adam Epoch [{epoch+1}/500], Loss: {loss:.4f}')

    opt_lbgfs = torch.optim.LBFGS([b_tavg, b_dayl, b_cu, kappa, lam], lr=2,
                                  history_size=200, max_iter=20, line_search_fn='strong_wolfe')
    print(f"Initial Loss: {training_run(opt_lbgfs):.4f}")
    for epoch in range(num_epochs):
        try:
            loss = training_run(opt_lbgfs)
            opt_lbgfs.step(lambda : training_run(opt_lbgfs))
        except:
            print("Encountered error during L-BFGS optimization, retrying with new random initialization.")
            retry -= 1
            b_tavg, b_dayl, b_cu, kappa, lam = random_init_params(total_models, device, dtype)
            break

        if torch.any(torch.isnan(loss)):
            print("Encountered NaN loss, retrying with new random initialization.")
            retry -= 1
            b_tavg, b_dayl, b_cu, kappa, lam = random_init_params(total_models, device, dtype)
            break

        if es.early_stop(loss.item()):
            print('Early stopping')
            fit = True
            break
        
        if epoch % report_freq == 0:
            print(f'L-BFGS Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}')
    if epoch == num_epochs-1:
        fit = True
  if retry == 0:
    print("Failed to fit model after multiple retries. Exiting.")
    sys.exit(1)

##########################################
# Save Training Output
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

# Scatter parameters into full array. This ensures that final array
# contains all pixels, even if a model wasn't fit for all of them.
b_tavg_save = torch.full((1, output_models, 1), np.nan, dtype=dtype, device='cpu')
b_dayl_save = torch.full((1, output_models, 1), np.nan, dtype=dtype, device='cpu')
b_cu_save   = torch.full((1, output_models, 1), np.nan, dtype=dtype, device='cpu')
kappa_save  = torch.full((1, output_models, 1), np.nan, dtype=dtype, device='cpu')
lam_save    = torch.full((1, output_models, 1), np.nan, dtype=dtype, device='cpu')
b_tavg_save.scatter_(1, torch.tensor(index).reshape(1,-1,1), b_tavg.detach().cpu())
b_dayl_save.scatter_(1, torch.tensor(index).reshape(1,-1,1), b_dayl.detach().cpu())
b_cu_save.scatter_(1,   torch.tensor(index).reshape(1,-1,1), b_cu.detach().cpu())
kappa_save.scatter_(1,  torch.tensor(index).reshape(1,-1,1), kappa.detach().cpu())
lam_save.scatter_(1,    torch.tensor(index).reshape(1,-1,1), lam.detach().cpu())

# Reshape and concatenate output arrays
b_tavg_save = b_tavg_save.reshape(1, output_x, output_y)
b_dayl_save = b_dayl_save.reshape(1, output_x, output_y)
b_cu_save   = b_cu_save.reshape(1, output_x, output_y)
kappa_save  = kappa_save.reshape(1, output_x, output_y)
lam_save    = lam_save.reshape(1, output_x, output_y)
output = torch.cat((b_tavg_save, b_dayl_save, b_cu_save, kappa_save, lam_save), axis=0)
output = output.detach().numpy()

with rio.open(f'/lustre/scratch5/cscholl/modis/2001_01_01.tif',
                boundless=True,
                window=window) as src:
  profile = src.profile

# Update relevant save data
profile['transform'] = rio.Affine(profile['transform'][0]*10,
                                  profile['transform'][1],
                                  profile['transform'][2],
                                  profile['transform'][3],
                                  profile['transform'][4]*10,
                                  profile['transform'][5])
profile['count'] = output.shape[0]
profile['height'] = output.shape[1]
profile['width'] = output.shape[2]
profile['dtype'] = output.dtype

# Save results
with rio.Env():
    with rio.open(f'/lustre/scratch5/cscholl/pheno_params/pheno_params_{args.window}.tif',
                  'w', **profile) as dst:
        dst.write(output)
