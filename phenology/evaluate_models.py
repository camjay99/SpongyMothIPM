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
parser.add_argument('--height', '-h', type=int, action='store_true',
                    default=100)
parser.add_argument('--width', '-w', type=int, action='store_true',
                    default=100)

# Window number to evaluate model for. Should be less than total number of windows.
parser.add_argument('--window', '-n', type=int, action='store_true',
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
    raise ValueError("Number of years cannot be greater than 24, as data is only available from 2001 to 2024.")

# Create list of windows to loop through for model evaluation. 
windows = load_daymet_forcing.create_window_grid(
   '/lustre/scratch5/cscholl/modis/2001_01_01.tif', args.height, args.width)
print('Number of windows: ', len(windows))
window = windows[args.window]
print('Using window: ', window)


##########################################
# Retrieve pixel-year evaluation samples
##########################################