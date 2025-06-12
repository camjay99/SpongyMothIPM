import math

import torch
import matplotlib.pyplot as plt

from SpongthMothIPM.config import Config
import SpongyMothIPM.util as util

config = Config(dtype=torch.float,
                n_bins=200,
                min_x=0,
                max_x=0,
                delta_t=1)

###############
# Get Transfers
###############

def get_transfers(tensor):
    transfers = torch.sum(tensor*config.xs_for_transfer)
    tensor = tensor*~config.xs_for_transfer
    return tensor, transfers
    
def add_transfers(tensor, transfers):
    return tensor + transfers*config.input_xs

def get_transfers_diapause(tensor):
    tensor = torch.reshape(tensor, config.shape)
    transfers = torch.sum(tensor*config.grid2d_for_transfer)
    tensor = tensor*~config.grid2d_for_transfer
    tensor = torch.flatten(tensor)
    return tensor, transfers

def add_transfers_diapause(tensor, transfers):
    tensor = torch.reshape(tensor, config.shape)
    tensor = tensor + transfers*config.input_xs
    return torch.flatten(tensor)

###########
# Mortality
###########

# Mortality for all stages beyond new instars will be 
# temporarily modeling as a single, age-independent
# rate. It will be replaced if better estimates are 
# found.

mortality_prediapause = torch.tensor(0.1, dtype=config.dtype, requires_grad=True)
mortality_diapause = torch.tensor(0.1, dtype=config.dtype, requires_grad=True)
mortality_postdiapause = torch.tensor(0.1, dtype=config.dtype, requires_grad=True)
mortality_L1 = torch.tensor(0.7, dtype=config.dtype, requires_grad=True)
mortality_L2 = torch.tensor(0.7, dtype=config.dtype, requires_grad=True)
mortality_L3 = torch.tensor(0.7, dtype=config.dtype, requires_grad=True)
mortality_L4 = torch.tensor(0.7, dtype=config.dtype, requires_grad=True)
mortality_L5_male = torch.tensor(0.7, dtype=config.dtype, requires_grad=True)
mortality_L5_L6_female = torch.tensor(0.7, dtype=config.dtype, requires_grad=True)
mortality_pupae_male = torch.tensor(0.4, dtype=config.dtype, requires_grad=True)
mortality_pupae_female = torch.tensor(0.4, dtype=config.dtype, requires_grad=True)
mortality_adults = torch.tensor(0.1, dtype=config.dtype, requires_grad=True)

def apply_mortality(tensor, mortality):
    return tensor*(1-mortality)

# Starvation of L1 instars prior to finding food
preincrease = torch.tensor(7.20292573)
changepoint = torch.tensor(14.22353787)
slope = torch.tensor(1.53550927)
days = torch.tensor(3)

def calc_starvation(temp):
    return (((temp < changepoint)
              * preincrease)
            + ((temp > changepoint)
               * (slope
                  * (temp - changepoint) 
                  + preincrease)))

##############
# Reproduction
##############

# Basic reproduction function, as we are currently only focusing
# on early season synchrony. Future versions can include more
# robust reproduction.

def calc_reproduction(adult_females):
    return 2*adult_females

##############
# Model Driver
##############

pop_prediapause = util.LnormPDF(config.xs, torch.tensor(0.2), torch.tensor(1.1))
pop_diapause_I = util.LnormPDF(config.from_x, torch.tensor(0.2), torch.tensor(1.1))
pop_diapause_D = util.LnormPDF(config.to_x, torch.tensor(0.4), torch.tensor(1.1))
pop_diapause = torch.flatten(pop_diapause_I * pop_diapause_D)
pop_postdiapause = util.LnormPDF(xs, torch.tensor(0.2), torch.tensor(1.1))
pop_L1 = util.LnormPDF(config.xs, torch.tensor(0.2), torch.tensor(1.1))
pop_L2 = util.LnormPDF(config.xs, torch.tensor(0.2), torch.tensor(1.1))
pop_L3 = util.LnormPDF(config.xs, torch.tensor(0.2), torch.tensor(1.1))
pop_L4 = util.LnormPDF(config.xs, torch.tensor(0.2), torch.tensor(1.1))
pop_L5_male = util.LnormPDF(config.xs, torch.tensor(0.2), torch.tensor(1.1))
pop_L5_L6_female = util.LnormPDF(config.xs, torch.tensor(0.2), torch.tensor(1.1))
pop_pupae_male = util.LnormPDF(config.xs, torch.tensor(0.2), torch.tensor(1.1))
pop_pupae_female = util.LnormPDF(config.xs, torch.tensor(0.2), torch.tensor(1.1))
pop_adult = util.LnormPDF(config.xs, torch.tensor(0.2), torch.tensor(1.1))

def run_model_one_time_step(temp):
    # Apply mortality
    pop_prediapause = apply_mortality(pop_prediapause, mortality_prediapause)
    pop_diapause = apply_mortality(pop_diapause, mortality_diapause)
    pop_postdiapause = apply_mortality(pop_postdiapause, mortality_postdiapause)
    pop_L1 = apply_mortality(pop_L1, mortality_L1)
    pop_L2 = apply_mortality(pop_L2, mortality_L2)
    pop_L3 = apply_mortality(pop_L3, mortality_L3)
    pop_L4 = apply_mortality(pop_L4, mortality_L4)
    pop_L5_male = apply_mortality(pop_L5_male, mortality_L5_male)
    pop_L5_L6_female = apply_mortality(pop_L5_L6_female, mortality_L5_L6_female)
    pop_pupae_male = apply_mortality(pop_pupae_male, mortality_pupae_male)
    pop_pupae_female = apply_mortality(pop_pupae_female, mortality_pupae_female)
    #pop_adult = apply_mortality(pop_adult, mortality_adult)

    # Apply kernels
    pop_prediapause = kern_prediapause @ pop_prediapause
    pop_diapause = kern_diapause_2D @ pop_diapause
    pop_postdiapause = kern_postdiapause @ pop_postdiapause
    pop_L1 = kern_L1 @ pop_L1
    pop_L2 = kern_L2 @ pop_L2
    pop_L3 = kern_L3 @ pop_L3
    pop_L4 = kern_L4 @ pop_L4
    pop_L5_male = kern_L5_male @ pop_L5_male
    pop_L5_L6_female = kern_L5_L6_female @ pop_L5_L6_female
    pop_pupae_female = kern_pupae_female @ pop_pupae_female
    pop_pupae_male = kern_pupae_male @ pop_pupae_male
    pop_adult = kern_adult @ pop_adult

    # Collect transfers
    pop_prediapause, transfer_prediapause = get_transfers(pop_prediapause)
    pop_diapause, transfer_diapause = get_transfers_diapause(pop_diapause)
    pop_postdiapause, transfer_postdiapause = get_transfers(pop_postdiapause)
    pop_L1, transfer_L1 = get_transfers(pop_L1)
    pop_L2, transfer_L2 = get_transfers(pop_L2)
    pop_L3, transfer_L3 = get_transfers(pop_L3)
    pop_L4, transfer_L4 = get_transfers(pop_L4)
    pop_L5_male, transfer_L5_male = get_transfers(pop_L5_male)
    pop_L5_L6_female, transfer_L5_L6_female = get_transfers(pop_L5_L6_female)
    pop_pupae_male, transfer_pupae_male = get_transfers(pop_pupae_male)
    pop_pupae_female, transfer_pupae_female = get_transfers(pop_pupae_female)
    pop_adult, transfer_adult = get_transfers(pop_adult)

    # Calculate reproduction
    transfer_adult = calc_reproduction(transfer_adult)

    # Add transfers
    pop_prediapause = add_transfers(pop_prediapause, transfer_adult)
    pop_diapause = add_transfers_diapause(pop_diapause, transfer_prediapause)
    pop_postdiapause = add_transfers(pop_postdiapause, transfer_diapause)
    pop_L1 = add_transfers(pop_L1, transfer_postdiapause)
    pop_L2 = add_transfers(pop_L2, transfer_L1)
    pop_L3 = add_transfers(pop_L3, transfer_L2)
    pop_L4 = add_transfers(pop_L4, transfer_L3)
    pop_L5_male = add_transfers(pop_L5_male, transfer_L4/2)
    pop_L5_L6_female = add_transfers(pop_L5_L6_female, transfer_L4/2)
    pop_pupae_male = add_transfers(pop_pupae_male, transfer_L5_male)
    pop_pupae_female = add_transfers(pop_pupae_female, transfer_L5_L6_female)
    pop_adult = add_transfers(pop_adult, transfer_L5_male + transfer_L5_L6_female)