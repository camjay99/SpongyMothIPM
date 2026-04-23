##################################
# Imports
##################################

import argparse

import numpy as np
import rasterio as rio
from rasterio.plot import show
from rasterio.windows import Window
import torch

import load_daymet_forcing

##################################
# Set constants
##################################

device='cpu' # Device to run optimizing code
dtype=torch.float32 # Dtype of data
num_years = 2 # Number of years in the analysis
num_days = 213 # Number of days since November 1st of previous year to use

# Create list of windows to loop through for model fitting. 
windows = load_daymet_forcing.create_window_grid(
   '/lustre/scratch5/cscholl/modis/2001_01_01.tif', 100)
print('Number of windows: ', len(windows))
window = windows[10]
print('Using window: ', window)

##################################
# Determine pixel-year samples
##################################

print("Loading landcovers")
# Load landcovers into single data frame for generating pixel-year samples
landcovers = []
for year in range(2001, 2003):
  with rio.open(f'/lustre/scratch5/cscholl/landcover/{year}_01_01.tif') as src:
    landcover = src.read(1, window=window, boundless=True)
    landcovers.append(landcover)
landcovers = np.stack(landcovers, axis=0)

output_x = landcovers.shape[1]//10
output_y = landcovers.shape[2]//10
output_pixels = output_x*output_y

print("Creating samples")
# Create random sample of pixel-years to be used for model fitting
# Each sample will have coordinates relative to the output pixel they are for
samples = []
skipped = []
for i in range(output_x):
  for j in range(output_y):
    # Get subsample that is only this output pixel
    mask = landcovers[:,
                      10*i:10*(i+1),
                      10*j:10*(j+1)]
    # Get indices of only non-zero elements
    nonzero = np.nonzero(mask)
    # If no elements to be sampled, skip this
    if len(nonzero[0]) == 0:
      skipped.append((i,j))
      samples.append([[-1],[-1],[-1]])
      continue
    # If less than 300 elements, add all to samples
    if len(nonzero[0]) < 300:
      nonzero = tuple((np.resize(dim, 300) for dim in nonzero))
      samples.append(nonzero)
      continue
    # Randomly choose list indices for inclusion in sample
    indices = np.random.choice(np.arange(nonzero[0].shape[0]), 300, replace=False)
    # Get sample indices
    sample = tuple(dim[indices] for dim in nonzero)
    samples.append(sample) # Save sample for each year.

total_pixels = output_pixels - len(skipped)


##########################################
# Load and rearrange data for computations
##########################################

print("Loading forcings")
# Load all of the data frames in advance
tavgs = []
dayls = []
cus = []
soss = []
for year in range(2001, 2003):
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
tavg_allsites = []
dayl_allsites = []
cu_allsites = []
sos_allsites = []
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
    tavg_samples = []
    dayl_samples = []
    cu_samples = []
    sos_samples = []
    # # Create window for loading data
    # window = Window(j*10, i*10, 10, 10) # col/row but resulting array is row/col
    # For each year, open data files and extract appropriate data
    for k in range(num_years):
      # Get all pixels for this year
      sample_year = [s[samples[i*output_y + j][0] == k] for s in samples[i*output_y + j]]
      tavg_year = tavgs[k][:, 10*i:10*(i+1),
                         10*j:10*(j+1)]
      dayl_year = dayls[k][:, 10*i:10*(i+1),
                         10*j:10*(j+1)]
      cu_year = cus[k][:, 10*i:10*(i+1),
                     10*j:10*(j+1)]
      sos_year = soss[k][:, 10*i:10*(i+1),
                       10*j:10*(j+1)]
      # Get forcing only for pixels to be sampled
      tavg_sample = tavg_year[(slice(None),) + (sample_year[1], sample_year[2])]
      dayl_sample = dayl_year[(slice(None),) + (sample_year[1], sample_year[2])]
      cu_sample = cu_year[(slice(None),) + (sample_year[1], sample_year[2])]
      # Create sos sample arrays, which are 1 if sos has been reached and 0 otherwise. This is done by comparing the day of year to the sos day of year for each pixel.
      print(sos_year.shape)
      sos_sample = np.arange(tavg_sample.shape[0]).reshape(-1, 1).repeat(tavg_sample.shape[1], 1)
      sos_sample = sos_sample >= sos_year[(slice(None),) + (sample_year[1], sample_year[2])].reshape(1,-1)
      # Pad with
      # Save each year of data
      print('Tavg shape', tavg_sample.shape)
      tavg_samples.append(tavg_sample)
      dayl_samples.append(dayl_sample)
      cu_samples.append(cu_sample)
      sos_samples.append(sos_sample)
    # Combine all years into single data frame
    tavg_samples = np.concat(tavg_samples, axis=1)
    dayl_samples = np.concat(dayl_samples, axis=1)
    cu_samples = np.concat(cu_samples, axis=1)
    sos_samples = np.concat(sos_samples, axis=1)
    # Add to outputs
    print('Tavg_samples shape: ', tavg_samples.shape)
    tavg_allsites.append(tavg_samples)
    dayl_allsites.append(dayl_samples)
    cu_allsites.append(cu_samples)
    sos_allsites.append(sos_samples)
tavg = np.stack(tavg_allsites, axis=0)
dayl = np.stack(dayl_allsites, axis=0)
cu = np.stack(cu_allsites, axis=0)
sos = np.stack(sos_allsites, axis=0)

print('tavg', tavg.shape)
print('dayl', dayl.shape)
print('cu', cu.shape)
##########################################
# Move data to GPU
##########################################

with torch.device(device):
  tavg = torch.tensor(tavg.transpose(1,0,2), dtype=dtype)
  dayl = torch.tensor(dayl.transpose(1,0,2), dtype=dtype)
  cu = torch.tensor(cu.transpose(1,0,2), dtype=dtype)
  sos = torch.tensor(sos.transpose(1,0,2), dtype=dtype)

with torch.device(device):
  b_tavg = torch.full((1,total_pixels,1), 1, requires_grad=True, dtype=dtype)
  b_dayl = torch.full((1,total_pixels,1), 1, requires_grad=True, dtype=dtype)
  b_cu = torch.full((1,total_pixels,1), 0.1, requires_grad=True, dtype=dtype)
  kappa = torch.full((1,total_pixels,1), -8, requires_grad=True, dtype=dtype)
  lam = torch.full((1,total_pixels,1), 0.1, requires_grad=True, dtype=dtype)


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
  num_epochs = 50
  report_freq = 1
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
b_tavg_save = torch.full((1, output_pixels, 1), np.nan, dtype=dtype, device='cpu')
b_dayl_save = torch.full((1, output_pixels, 1), np.nan, dtype=dtype, device='cpu')
b_cu_save = torch.full((1, output_pixels, 1), np.nan, dtype=dtype, device='cpu')
kappa_save = torch.full((1, output_pixels, 1), np.nan, dtype=dtype, device='cpu')
lam_save = torch.full((1, output_pixels, 1), np.nan, dtype=dtype, device='cpu')
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
    with rio.open(f'/lustre/scratch5/cscholl/pheno_params/pheno_params_{x}_{y}.tif', 'w', **profile) as dst:
        dst.write(output)
