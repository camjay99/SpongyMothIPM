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
from rasterio.windows import transform
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



# Assemble final arrays by concatenating year chunks per pixel. Each entry
# stays a separate per-model array (shape (days, n_samples)) -- nothing is
# ever stacked or combined across models.
print("Assembling forcing arrays")
tavg_permodel = []
dayl_permodel = []
cu_permodel   = []
sos_permodel  = []
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

total_models = output_x*output_y - len(skipped)

# If one of these are nan from the start, report as error
for m in range(total_models):
    if (np.any(np.isnan(tavg_permodel[m])) | np.any(np.isnan(dayl_permodel[m])) |
        np.any(np.isnan(cu_permodel[m])) | np.any(np.isnan(sos_permodel[m]))):
        print('Found nan forcings')
        sys.exit(1)

# Normalize each model's forcings using only that model's own statistics.
forcing_stats_permodel = []
for m in range(total_models):
    tavg_mean = tavg_permodel[m].mean()
    tavg_std = tavg_permodel[m].std()
    dayl_mean = dayl_permodel[m].mean()
    dayl_std = dayl_permodel[m].std()
    cu_min = cu_permodel[m].min()
    cu_max = cu_permodel[m].max()
    forcing_stats_permodel.append(
        np.array([tavg_mean, tavg_std, dayl_mean, dayl_std, cu_min, cu_max]))

    tavg_permodel[m] = (tavg_permodel[m] - tavg_mean) / tavg_std
    dayl_permodel[m] = (dayl_permodel[m] - dayl_mean) / dayl_std
    cu_norm = (cu_permodel[m] - cu_min) / (cu_max - cu_min)
    # If cu doesn't change, it can be set to 0 without affecting model fitting.
    cu_permodel[m] = np.where(np.isnan(cu_norm), 0, cu_norm)

print('total_models', total_models)
if total_models > 0:
    print('tavg[0]', tavg_permodel[0].shape)
    print('dayl[0]', dayl_permodel[0].shape)
    print('cu[0]', cu_permodel[0].shape)

##########################################
# Move data to device, one tensor per model
##########################################

tavg_list = []
dayl_list = []
cu_list   = []
sos_list  = []
with torch.device(device):
  for m in range(total_models):
    tavg_list.append(torch.tensor(tavg_permodel[m], dtype=dtype))
    dayl_list.append(torch.tensor(dayl_permodel[m], dtype=dtype))
    cu_list.append(torch.tensor(cu_permodel[m],   dtype=dtype))
    sos_list.append(torch.tensor(sos_permodel[m],  dtype=dtype))


def random_init_params(device, dtype):
    """Randomly initialize the 6 scalar parameters for a single model."""
    with torch.device(device):
        b_tavg = (torch.rand((), dtype=dtype)*0.1 + 0.95).requires_grad_()
        b_dayl = (torch.rand((), dtype=dtype)*0.1 + 0.95).requires_grad_()
        b_cu   = (torch.rand((), dtype=dtype)*0.01 + 0.095).requires_grad_()
        b_const = (torch.rand((), dtype=dtype) + 0.5).requires_grad_()
        kappa  = (torch.rand((), dtype=dtype)*0.1 - -0.55).requires_grad_()
        lam    = (torch.rand((), dtype=dtype)*0.01 - 0.095).requires_grad_()
    return [b_tavg, b_dayl, b_cu, b_const, kappa, lam]


# Per-model list of parameter lists: params[m] = [b_tavg, b_dayl, b_cu, b_const, kappa, lam]
params = [random_init_params(device, dtype) for _ in range(total_models)]


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
# Run Model Training
##########################################

class EarlyStopper:
    """Tracks early stopping for a single model."""
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = float('inf')
        self.stopped = False

    def update(self, loss):
        # loss: detached scalar loss value (python float) for this model
        if loss < (self.min_loss - self.min_delta):
            self.min_loss = loss
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.stopped = True
        return self.stopped


def training_run(m, optimizer):
    """Run a single forward/backward/grad-fixup pass for model m. Does not step."""
    optimizer.zero_grad()

    pred = make_prediction(tavg_list[m], dayl_list[m], cu_list[m],
                            *params[m][:4], params[m][5], params[m][4])

    # Sum-of-squared-error loss for this model only.
    loss = torch.sum((pred - sos_list[m])**2)
    loss.backward()

    if torch.isnan(loss):
        raise Exception("Encountered NaN loss")

    # Replace nan grads so fitting may continue
    for p in params[m]:
        if torch.isnan(p.grad).any():
            p.grad = torch.zeros_like(p.grad)

    return loss


print("Starting model fitting")
with torch.device(device):
  max_retries = 3
  optimizers = [torch.optim.Adam(params[m], lr=0.01) for m in range(total_models)]
  stoppers = [EarlyStopper(patience=25, min_delta=0.1) for _ in range(total_models)]
  retries_left = [max_retries for _ in range(total_models)]
  failed = [False for _ in range(total_models)]
  active = set(range(total_models))

  if total_models > 0:
      init_losses = [training_run(m, optimizers[m]).item() for m in range(total_models)]
      print(f"Initial Loss: {sum(init_losses):.4f}")

  for epoch in range(15000):
    if not active:
        break
    newly_done = []
    for m in list(active):
        try:
            loss = training_run(m, optimizers[m])
            optimizers[m].step()
            stopped = stoppers[m].update(loss.detach().item())
            if stopped:
                newly_done.append(m)
        except Exception as e:
            print(f"Model {m}: encountered error during optimization: {e}")
            retries_left[m] -= 1
            if retries_left[m] <= 0:
                print(f"Model {m}: failed to fit after multiple retries. Leaving output as NaN.")
                failed[m] = True
                newly_done.append(m)
            else:
                print(f"Model {m}: retrying with new random initialization.")
                params[m] = random_init_params(device, dtype)
                optimizers[m] = torch.optim.Adam(params[m], lr=0.01)
                stoppers[m] = EarlyStopper(patience=25, min_delta=0.1)

    for m in newly_done:
        active.discard(m)

    if epoch % 50 == 0:
        print(f'Adam Epoch [{epoch+1}/15000], '
              f'Active: {len(active)}/{total_models}, '
              f'Stopped: {total_models - len(active)}/{total_models}')

  if active:
      print(f'Reached max epochs with {len(active)} models still active.')


##########################################
# Save Training Output
##########################################

# Create index list mapping fitted model index -> (i, j) output pixel.
index = []
for i in range(output_x):
  for j in range(output_y):
    # Skip if no samples
    if (i,j) in skipped:
      print(f'Skipping {i}, {j}')
      continue
    index.append((i,j))

# Build output rasters by placing each model's own fitted (or NaN, if failed)
# params directly into its (i, j) location. This final assembly combines
# all models only for writing the output raster -- no training computation
# ever touches more than one model's data.
b_tavg_save = np.full((output_x, output_y), np.nan, dtype=np.float32)
b_dayl_save = np.full((output_x, output_y), np.nan, dtype=np.float32)
b_cu_save   = np.full((output_x, output_y), np.nan, dtype=np.float32)
b_const_save = np.full((output_x, output_y), np.nan, dtype=np.float32)
kappa_save  = np.full((output_x, output_y), np.nan, dtype=np.float32)
lam_save    = np.full((output_x, output_y), np.nan, dtype=np.float32)
fstats_save = np.full((6, output_x, output_y), np.nan, dtype=np.float32)

for m, (i, j) in enumerate(index):
    if failed[m]:
        continue
    b_tavg, b_dayl, b_cu, b_const, kappa, lam = params[m]
    b_tavg_save[i, j] = b_tavg.detach().cpu().item()
    b_dayl_save[i, j] = b_dayl.detach().cpu().item()
    b_cu_save[i, j]   = b_cu.detach().cpu().item()
    b_const_save[i, j] = b_const.detach().cpu().item()
    kappa_save[i, j]  = kappa.detach().cpu().item()
    lam_save[i, j]    = lam.detach().cpu().item()
    fstats_save[:, i, j] = forcing_stats_permodel[m]

output = np.stack((b_tavg_save, b_dayl_save, b_cu_save, b_const_save, kappa_save, lam_save), axis=0)

with rio.open(f'/lustre/scratch5/cscholl/modis/2001_01_01.tif',
                boundless=True,
                window=window) as src:
  profile = src.profile

# Update relevant save data
window_transform = transform(window, src.transform)
profile['transform'] = rio.Affine(window_transform[0]*10,
                                  window_transform[1],
                                  window_transform[2],
                                  window_transform[3],
                                  window_transform[4]*10,
                                  window_transform[5])
profile['count'] = output.shape[0]
profile['height'] = output.shape[1]
profile['width'] = output.shape[2]
profile['dtype'] = output.dtype

# Save results
with rio.Env():
    with rio.open(f'/lustre/scratch5/cscholl/pheno_params/pheno_params_{args.window}.tif',
                  'w', **profile) as dst:
        dst.write(output)

profile['count'] = fstats_save.shape[0]
profile['dtype'] = fstats_save.dtype
# Save results
with rio.Env():
    with rio.open(f'/lustre/scratch5/cscholl/forcings/forcings_{args.window}.tif',
                  'w', **profile) as dst:
        dst.write(fstats_save)
