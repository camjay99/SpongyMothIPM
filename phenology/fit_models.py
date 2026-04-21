##################################
# Imports
##################################

import numpy as np
import rasterio as rio
from rasterio.plot import show
from rasterio.windows import Window
import torch


##################################
# Set constants
##################################

device='cuda' # Device to run optimizing code
dtype=torch.float32 # Dtype of data
num_years = 24 # Number of years in the analysis
num_days = 213 # Number of days since November 1st of previous year to use

##################################
# Determine pixel-year samples
##################################

# Load landcovers into single data frame for generating pixel-year samples
landcovers = []
for year in range(2001, 2025):
  with rio.open(f'/tmp/modis_landcover_{year}.tif') as src:
    landcover = src.read(1)
    landcovers.append(landcover)
landcovers = np.stack(landcovers, axis=0)

output_x = landcovers.shape[1]//10
output_y = landcovers.shape[2]//10
output_pixels = output_x*output_y

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
      continue
    # If less than 300 elements, add all to samples
    if len(nonzero[0]) < 300:
      samples.append(nonzero)
      continue
    # Randomly choose list indices for inclusion in sample
    indices = np.random.choice(np.arange(nonzero[0].shape[0]), 300)
    # Get sample indices
    sample = tuple(dim[indices] for dim in nonzero)
    samples.append(sample) # Save sample for each year.

total_pixels = output_pixels - len(skipped)


##########################################
# Load and rearrange data for computations
##########################################

