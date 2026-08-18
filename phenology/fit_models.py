##################################
# Imports
##################################

import argparse
import json

import numpy as np
import rasterio as rio
from rasterio.plot import show
from rasterio.windows import Window
import torch

import load_daymet_forcing


########################################
# Parse arguments for running on cluster
########################################

parser = argparse.ArgumentParser(
    description='Options for fitting phenology models to MODIS data.')

# Device to use for fitting. Options are "cpu" and "cuda". 
# Note that using cuda requires a GPU with sufficient memory to 
# hold the data and model parameters.
parser.add_argument('--device', '-d', type=str, action='store_true')

# Dtype to use in fitting. Using float32 will reduce both memory usage and
# precision. Using float64 will increase memory usage but may improve precision.
parser.add_argument('--dtype', '-t', type=str, action='store_true',
                    choices=['float32', 'float64'], default='float32')

# Number of years to include in fitting. Years included will start at 2001.
parser.add_argument('--num_years', '-y', type=int, action='store_true', 
                    default=24)

# Width and height of windows in MODIS pixels. Should be a multiple of 10.
parser.add_argument('--height', '-H', type=int, action='store_true',
                    default=100)
parser.add_argument('--width', '-W', type=int, action='store_true',
                    default=100)

# Window number to fit model for. Should be less than total number of windows.
parser.add_argument('--window', '-n', type=int, action='store_true',
                    default=0)

# Number of epochs for model fitting. More epochs will increase runtime but may improve fit.
parser.add_argument('--num_epochs', '-e', type=int, action='store_true',
                    default=50)

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
    raise ValueError("Number of years cannot be greater than 24, as data is only available from 2001 to 2024.")

# Number of epochs for model fitting
num_epochs = args.num_epochs 
if args.num_epochs <= 0:
    raise ValueError("Number of epochs must be greater than 0.")

# Create list of windows to loop through for model fitting. 
windows = load_daymet_forcing.create_window_grid(
   '/lustre/scratch5/cscholl/modis/2001_01_01.tif', args.height, args.width)
print('Number of windows: ', len(windows))
window = windows[args.window]
print('Using window: ', window)

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
      samples[(i,j)] = sample # Save sample for each tile.
      sample_choices[(i,j)] = indices.tolist() 

# Save sample choices for reproducibility and later evaluation.
with open(f'../data/samples/{args.window}.json', 'w') as f:
    f.write(json.dumps(sample_choices))

total_models = output_models - len(skipped)


##########################################
# Load and rearrange data for computations
##########################################

print("Loading forcings")
# Load all of the data frames in advance
tavgs = []
dayls = []
cus = []
soss = []
for year in range(2001, 2001 + num_years):
  tavg, dayl, cu = load_daymet_forcing.load_year_forcing(
      year=year,
      daymet_dir='/lustre/scratch5/cscholl/daymet',
      imagery_path=f'/lustre/scratch5/cscholl/modis/{year}_01_01.tif',
      chill_threshold=5.0,
      window=window
  )
  tavgs.append(tavg)
  dayls.append(dayl)
  cus.append(cu)
  with rio.open(f'/lustre/scratch5/cscholl/modis/{year}_01_01.tif') as src:
    soss.append(src.read(boundless=True, window=window))

print("Sampling forcings")
# Create forcings arrays for all samples
tavg_allpixyears = []
dayl_allpixyears = []
cu_allpixyears = []
sos_allpixyears = []
print(output_x)
print(output_y)
# For each sample, need to collect drivers + phenology for each year and accumulate
for i in range(output_x):
  for j in range(output_y):
    # Skip if no samples
    if (i,j) in skipped:
      print(f'Skipping {i}, {j}')
      continue
    # Create forcings arrays for all years for this sample
    tavg_pixel = []
    dayl_pixel = []
    cu_pixel = []
    sos_pixel = []
    # # Create window for loading data
    # window = Window(j*10, i*10, 10, 10) # col/row but resulting array is row/col
    # For each year, open data files and extract appropriate data
    for k in range(num_years):
      # Get all pixels for this year
      sample_year = [s[samples[(i,j)][0] == k] for s in samples[(i,j)]]
      tavg_year = tavgs[k][:, 10*i:10*(i+1),
                         10*j:10*(j+1)]
      dayl_year = dayls[k][:, 10*i:10*(i+1),
                         10*j:10*(j+1)]
      cu_year = cus[k][:, 10*i:10*(i+1),
                     10*j:10*(j+1)]
      sos_year = soss[k][:, 10*i:10*(i+1),
                       10*j:10*(j+1)]
      # Get forcing only for pixel-years to be sampled
      tavg_pixyear = tavg_year[(slice(None),) + (sample_year[1], sample_year[2])]
      dayl_pixyear = dayl_year[(slice(None),) + (sample_year[1], sample_year[2])]
      cu_pixyear = cu_year[(slice(None),) + (sample_year[1], sample_year[2])]
      # Create sos sample arrays, which are 1 if sos has been reached and 0 otherwise. This is done by comparing the day of year to the sos day of year for each pixel.
      print(sos_year.shape)
      sos_pixyear = np.arange(tavg_pixyear.shape[0]).reshape(-1, 1).repeat(tavg_pixyear.shape[1], 1)
      sos_pixyear = sos_pixyear >= sos_year[(slice(None),) + (sample_year[1], sample_year[2])].reshape(1,-1)
      # Pad with
      # Save each pixel-year of data
      print('Tavg shape', tavg_pixyear.shape)
      tavg_pixel.append(tavg_pixyear)
      dayl_pixel.append(dayl_pixyear)
      cu_pixel.append(cu_pixyear)
      sos_pixel.append(sos_pixyear)
    # Combine all pixel-years into single data frame for the current pixel
    tavg_pixel = np.concat(tavg_pixel, axis=1)
    dayl_pixel = np.concat(dayl_pixel, axis=1)
    cu_pixel = np.concat(cu_pixel, axis=1)
    sos_pixel = np.concat(sos_pixel, axis=1)
    # Add to list of all pixel-years for all pixels.
    print('Tavg_pixel shape: ', tavg_pixel.shape)
    tavg_allpixyears.append(tavg_pixel)
    dayl_allpixyears.append(dayl_pixel)
    cu_allpixyears.append(cu_pixel)
    sos_allpixyears.append(sos_pixel)
tavg = np.stack(tavg_allpixyears, axis=0)
dayl = np.stack(dayl_allpixyears, axis=0)
cu = np.stack(cu_allpixyears, axis=0)
sos = np.stack(sos_allpixyears, axis=0)

# Rescale tavg, dayl, and cu for numerical stability.
tavg = (tavg - tavg.mean(dim=(0,2))) / tavg.std(dim=(0,2))
dayl = (dayl - dayl.mean(dim=(0,2))) / dayl.std(dim=(0,2))
cu = (cu - cu.mean(dim=(0,2))) / cu.std(dim=(0,2))

print('tavg', tavg.shape)
print('dayl', dayl.shape)
print('cu', cu.shape)
##########################################
# Move data to GPU
##########################################

# Tensors are organized as:
# dim1 - days
# dim2 - models
# dim3 - samples (pixel-years)
with torch.device(device):
  tavg = torch.tensor(tavg.transpose(1,0,2), dtype=dtype)
  dayl = torch.tensor(dayl.transpose(1,0,2), dtype=dtype)
  cu = torch.tensor(cu.transpose(1,0,2), dtype=dtype)
  sos = torch.tensor(sos.transpose(1,0,2), dtype=dtype)

with torch.device(device):
  b_tavg = torch.full((1,total_models,1), 1, requires_grad=True, dtype=dtype)
  b_dayl = torch.full((1,total_models,1), 1, requires_grad=True, dtype=dtype)
  b_cu = torch.full((1,total_models,1), 0.1, requires_grad=True, dtype=dtype)
  kappa = torch.full((1,total_models,1), -8, requires_grad=True, dtype=dtype)
  lam = torch.full((1,total_models,1), 0.1, requires_grad=True, dtype=dtype)


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
  optimizer = torch.optim.LBFGS([b_tavg, b_dayl, b_cu, kappa, lam], lr=1,
                                history_size=200, max_iter=20, line_search_fn='strong_wolfe')
  es = EarlyStopper(patience=3, min_delta=0.001)
  # Closure capturing forward/backward passes is required for LBFGS optimization
  def closure():
    b_tavg.grad = None
    b_dayl.grad = None
    b_cu.grad = None
    kappa.grad = None
    lam.grad = None

    pred = make_prediction(tavg, dayl, cu, b_tavg, b_dayl, b_cu, lam, kappa)

    # We use the continuous ranked probability score (CRPS) as the loss function
    # which is a common forecasting metric.
    loss = torch.sum((pred - sos)**2)
    loss.backward()

    # Replace nan grads so fitting may continue
    b_tavg.grad[torch.isnan(b_tavg.grad)] = 0
    b_dayl.grad[torch.isnan(b_dayl.grad)] = 0
    b_cu.grad[torch.isnan(b_cu.grad)] = 0
    kappa.grad[torch.isnan(kappa.grad)] = 0
    lam.grad[torch.isnan(lam.grad)] = 0

    return loss

  for epoch in range(num_epochs):
      loss = closure()
      if es.early_stop(loss.item()):
        print('Early stopping')
        break
      if epoch % report_freq == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}')
      optimizer.step(closure)


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
b_cu_save = torch.full((1, output_models, 1), np.nan, dtype=dtype, device='cpu')
kappa_save = torch.full((1, output_models, 1), np.nan, dtype=dtype, device='cpu')
lam_save = torch.full((1, output_models, 1), np.nan, dtype=dtype, device='cpu')
b_tavg_save.scatter_(1, torch.tensor(index).reshape(1,-1,1), b_tavg.detach().cpu())
b_dayl_save.scatter_(1, torch.tensor(index).reshape(1,-1,1), b_dayl.detach().cpu())
b_cu_save.scatter_(1, torch.tensor(index).reshape(1,-1,1), b_cu.detach().cpu())
kappa_save.scatter_(1, torch.tensor(index).reshape(1,-1,1), kappa.detach().cpu())
lam_save.scatter_(1, torch.tensor(index).reshape(1,-1,1), lam.detach().cpu())

# Reshape and concatenate output arrays
b_tavg_save = b_tavg_save.reshape(1, output_x, output_y)
b_dayl_save = b_dayl_save.reshape(1, output_x, output_y)
b_cu_save = b_cu_save.reshape(1, output_x, output_y)
kappa_save = kappa_save.reshape(1, output_x, output_y)
lam_save = lam_save.reshape(1, output_x, output_y)
output = torch.cat((b_tavg_save, b_dayl_save, b_cu_save, kappa_save, lam_save), axis=0)
output = output.detach().numpy()

with rio.open(f'/lustre/scratch5/cscholl/modis/{year}_01_01.tif',
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

x = 'test'
y = 'test'
# Save results
with rio.Env():
    with rio.open(f'/lustre/scratch5/cscholl/pheno_params/pheno_params_{args.window}.tif', 
                  'w', **profile) as dst:
        dst.write(output)
