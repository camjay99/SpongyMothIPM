import math

import torch
import matplotlib.pyplot as plt

############################################
# Helper functions (move to separate module)
############################################
def LnormPDF(x, mu, sigma):
    """Compute the log-normal pdf based on x and a list of mulog/sigmalog."""
    # When x = 0, lognormal pdf will return nan when it should be 0.
    # Therefore, we slightly nudge values first (this helps ensure
    # differentiability, which zeroing the first entry would not do).
    x = x + 0.000000001
    return (1
            / (x
               * torch.log(sigma)
               * (math.sqrt(2.0*math.pi)))
            * torch.exp(
                -((torch.log(x)-torch.log(mu))**2)
                / (2*torch.log(sigma)**2)))


def Logan_TM1(temp, psi, rho, t_max, crit_temp_width, tbase=0):
    tau = (t_max - temp + tbase) / crit_temp_width
    return (psi
            * (torch.exp(rho*(temp - tbase)) 
                - torch.exp(rho*t_max - tau)))

def Logan_TM2(temp, alpha, kappa, rho, t_max, crit_temp_width, tbase=0):
    tau = (t_max - temp + tbase) / crit_temp_width
    return (alpha
            * ((1 + kappa
                * torch.exp(
                   -rho * (temp-tbase)))**-1
                - torch.exp(-tau)))


#############
# Torch Setup
#############
dtype = torch.float

################
# IPM Parameters
################
n_bins = 100 # Resolution of physiological age for each stage
min_x = 0
max_x = 1.5
delta_t = 1
temp = torch.tensor(15)

##################
# Helper Variables
##################
shape = (n_bins, n_bins)
xs = torch.linspace(min_x, max_x, n_bins)
xs_for_transfer = xs > 1
input_xs = torch.zeros_like(xs)
input_xs[0] = 1
from_x = torch.reshape(xs, (1, n_bins))
to_x = torch.reshape(xs, (n_bins, 1))
x_dif = torch.maximum(torch.tensor(0), to_x - from_x)

# These are used for computing diapause kernel
ys = torch.linspace(0.0001, 1, n_bins)
from_I = torch.reshape(xs, (n_bins, 1, 1, 1))
to_I = torch.reshape(xs, (1, 1, n_bins, 1))
from_D = torch.reshape(ys, (1, n_bins, 1, 1))
to_D = torch.reshape(ys, (1, 1, 1, n_bins))
I_dif = torch.maximum(torch.tensor(0), to_I - from_I)
D_dif = torch.maximum(torch.tensor(0), to_D - from_D)
grid2d = torch.squeeze(torch.ones_like(from_I)*from_D)
grid2d_for_transfer = grid2d > 1
input_grid2d = torch.zeros_like(grid2d)
input_grid2d[-1, 0] = 1

#####################
# Pre-diapause kernel
#####################
## Assumed Parameters
rho = torch.tensor(0.1455)
t_max = torch.tensor(33.993) # Maximum temperature before development failure
crit_temp_width = torch.tensor(6.350) # Width of interval below t_max where development decreases rapidly.
psi = torch.tensor(0.0191)

## Optimized Parameters
sigma_prediapause = torch.tensor(3, dtype=dtype, requires_grad=True)

## Compute kernel
mu_prediapause = (
    delta_t
    * Logan_TM1(temp, psi, rho, t_max, crit_temp_width))

kern_prediapause = LnormPDF(x_dif, mu_prediapause, sigma_prediapause)

#################
# Diapause kernel
#################
## Assumed Parameters
c = torch.tensor(-5.627108200)
pdr_t = torch.tensor(0.059969414)
pdr_t_2 = torch.tensor(0.010390411)
pdr_t_4 = torch.tensor(-0.000007987)
rp_c = torch.tensor(0.00042178)
rs_c = torch.tensor(0.7633152)
rs_rp = torch.tensor(-0.6404470)
I_0 = torch.tensor(1.1880)
A_1 = torch.tensor(1.56441438)
A_2 = torch.tensor(0.46354992)
t_min = torch.tensor(-5)
t_max = torch.tensor(25)
alpha = torch.tensor(2.00000)
beta = torch.tensor(0.62062)
gamma = torch.tensor(0.56000)

## Optimized Parameters
sigma_I_diapause = torch.tensor(3, dtype=dtype, requires_grad=True)
sigma_D_diapause = torch.tensor(3, dtype=dtype, requires_grad=True)

## Compute kernel
# Current strategy is to compute as a 4-D tensor to take advantage of broadcasting, then to 
# reshape into a 2D matrix to take advantage of matrix multiplication.
# To simplify calculations, we keep track of 1-I rather than I, so that
# all traits are always increasing.
Z = (t_max - temp) / (t_max - t_min)
rp = 1 + rp_c*(torch.exp(Z)**6)
rs = rs_c + rs_rp*rp
# Here we calculate dI/dt from I* = 1 - I
mu_I_diapause = (
    (delta_t
    * (torch.maximum(-1 + from_I,
                     (torch.log(rp)
                      * ((1 - from_I) 
                         - I_0 
                         - rs))))))
mu_I_diapause = -1*mu_I_diapause # dI*/dt = -dI/dt
# Change is expressed over entire input space, since
# inhibitor depletion does not depend on development rate
mu_I_diapause = torch.tile(mu_I_diapause, (1, n_bins, 1, 1)) 

A = 0.3 + 0.7*(1-Z)**(A_1 * (Z**A_2))
pdr = torch.exp(c + pdr_t*temp + pdr_t_2*(temp**2) + pdr_t_4*(temp**4))
mu_D_diapause = (
    (delta_t
     * (torch.maximum(torch.tensor(0),
                      (pdr
                       * (1 - (1 - from_I)*A))))))

kern_diapause_4D = (
    LnormPDF(I_dif, mu_I_diapause, sigma_I_diapause)
    * LnormPDF(D_dif, mu_D_diapause, sigma_D_diapause))

# Need to reshape kernel so that it can be 
# used in matrix-vector multiplication.
kern_diapause_2D = torch.reshape(kern_diapause_4D, 
                                 (n_bins, n_bins, n_bins*n_bins))
kern_diapause_2D = torch.permute(kern_diapause_2D, (2, 0, 1))
kern_diapause_2D = torch.reshape(kern_diapause_2D, 
                                 (n_bins*n_bins, n_bins*n_bins))


######################
# Post-diapause kernel
######################
## Assumed Parameters
tau = torch.tensor(-0.0127)
delta = torch.tensor(0.00297)
omega = torch.tensor(-0.08323)
kappa = torch.tensor(0.01298)
psi = torch.tensor(0.00099)
zeta = torch.tensor(-0.00004)

## Optimized Parameters
sigma_postdiapause = torch.tensor(1.1, dtype=dtype, requires_grad=True)

## Compute kernel
mu_postdiapause = (
    (delta_t
     * (tau + delta*temp # R_T(0)
        + (from_x
           * (omega 
              + kappa*temp 
              + psi*temp**2 
              + zeta*temp**3))))) # a_T * A

kern_postdiapause = LnormPDF(x_dif, mu_postdiapause, sigma_postdiapause)


###########
# L1 kernel
###########
## Assumed parameters
alpha = torch.tensor(0.9643)
kappa = torch.tensor(7.700)
rho = torch.tensor(0.1427)
t_max = torch.tensor(30.87)
crit_temp_width = torch.tensor(12.65)

## Optimized Parameters
sigma_L1 = torch.tensor(3, dtype=dtype, requires_grad=True)

## Compute kernel
mu_L1 = (
    (delta_t
     * Logan_TM2(temp, alpha, kappa, rho, t_max, crit_temp_width, 10)))

kern_L1 = LnormPDF(x_dif, mu_L1, sigma_L1)


###########
# L2 kernel
###########
## Assumed parameters
psi = torch.tensor(0.1454)
rho = torch.tensor(0.1720)
t_max = torch.tensor(21.09)
crit_temp_width = torch.tensor(4.688)

## Optimized Parameters
sigma_L2 = torch.tensor(3, dtype=dtype, requires_grad=True)

## Compute kernel
mu_L2 = (
    (delta_t
     * Logan_TM1(temp, psi, rho, t_max, crit_temp_width, 13.3)))

kern_L2 = LnormPDF(x_dif, mu_L2, sigma_L2)


###########
# L3 kernel
###########
## Assumed Parameters
alpha = torch.tensor(1.2039)
kappa = torch.tensor(8.062)
rho = torch.tensor(0.1737)
t_max = torch.tensor(24.12)
crit_temp_width = torch.tensor(8.494)

## Optimized Parameters
sigma_L3 = torch.tensor(3, dtype=dtype, requires_grad=True)

## Compute kernel
mu_L3 = (
    (delta_t
     * Logan_TM2(temp, alpha, kappa, rho, t_max, crit_temp_width, 13.3)))

kern_L3 = LnormPDF(x_dif, mu_L3, sigma_L3)


###########
# L4 kernel
###########
## Assumed Parameters
psi = torch.tensor(0.1120)
rho = torch.tensor(0.1422)
t_max = torch.tensor(22.29)
crit_temp_width = torch.tensor(5.358)

## Optimized Parameters
sigma_L4 = torch.tensor(3, dtype=dtype, requires_grad=True)

## Compute kernel
mu_L4 = (
    (delta_t 
     * Logan_TM1(temp, psi, rho, t_max, crit_temp_width, 13.3)))

kern_L4 = LnormPDF(x_dif, mu_L4, sigma_L4)


#######################
# L5/L6 kernel (Female)
#######################
## Assumed Parameters
b = torch.tensor(-0.0132)
m = torch.tensor(0.00162)
# m/b from average model over all larval stages, 
# here we rescale be the approximate proportion 
# of time spent in 5th/6th instars
scaling = torch.tensor(650/326)

## Optimized Parameters
sigma_L5_L6_female = torch.tensor(3, dtype=dtype, requires_grad=True)

## Compute kernel
mu_L5_L6_female = (
    (delta_t 
     * (b + m*scaling*temp)))

kern_L5_L6_female = LnormPDF(x_dif, mu_L5_L6_female, sigma_L5_L6_female)


##################
# L5 kernel (Male)
##################
## Assumed Parameters
b = torch.tensor(-0.0127)
m = torch.tensor(0.00177)
# m/b from average model over all larval stages, 
# here we rescale be the approximate proportion 
# of time spent in 5th instars
scaling = torch.tensor(583/240)

## Optimized Parameters
sigma_L5_male = torch.tensor(3, dtype=dtype, requires_grad=True)

## Compute kernel
mu_L5_male = (
    (delta_t
     * (b + m*scaling*temp)))

kern_L5_male = LnormPDF(x_dif, mu_L5_male, sigma_L5_male)


#######################
# Pupae kernel (Female)
#######################
## Assumed Parameters
b = torch.tensor(-0.0217)
m = torch.tensor(0.00427)

## Optimized Parameters
sigma_pupae_female = torch.tensor(3, dtype=dtype, requires_grad=True)

## Calculate kernel
mu_pupae_female = (
    (delta_t 
     * (b + m*temp)))

kern_pupae_female = LnormPDF(x_dif, mu_pupae_female, sigma_pupae_female)


#####################
# Pupae kernel (male)
#####################
## Assumed Parameters
b = torch.tensor(-0.0238)
m = torch.tensor(0.00362)

## Optimized Parameters
sigma_pupae_male = torch.tensor(3, dtype=dtype, requires_grad=True)

## Calculate kernel
mu_pupae_male = (
    (delta_t 
     * (b + m*temp)))

kern_pupae_male = LnormPDF(x_dif, mu_pupae_male, sigma_pupae_male)


##############
# Adult kernel
##############
## Assumed Parameters
b = torch.tensor(0.062)
m = torch.tensor(0.04)
## Optimized Parameters
sigma_adult = torch.tensor(3, dtype=dtype, requires_grad=True)

## Calculate kernel
mu_adult = (
    (delta_t 
     * (b + m*(temp-10))))

kern_adult = LnormPDF(x_dif, mu_adult, sigma_adult)


###############
# Get Transfers
###############

def get_transfers(tensor):
    transfers = torch.sum(tensor*xs_for_transfer)
    tensor = tensor*~xs_for_transfer
    return tensor, transfers
    
def add_transfers(tensor, transfers):
    return tensor + transfers*input_xs

def get_transfers_diapause(tensor):
    tensor = torch.reshape(tensor, shape)
    transfers = torch.sum(tensor*grid2d_for_transfer)
    tensor = tensor*~grid2d_for_transfer
    tensor = torch.flatten(tensor)
    return tensor, transfers

def add_transfers_diapause(tensor, transfers):
    tensor = torch.reshape(tensor, shape)
    tensor = tensor + transfers*input_xs
    return torch.flatten(tensor)

###########
# Mortality
###########

# Mortality for all stages beyond new instars will be 
# temporarily modeling as a single, age-independent
# rate. It will be replaced if better estimates are 
# found.

mortality_prediapause = torch.tensor(0.1, dtype=dtype, requires_grad=True)
mortality_diapause = torch.tensor(0.1, dtype=dtype, requires_grad=True)
mortality_postdiapause = torch.tensor(0.1, dtype=dtype, requires_grad=True)
mortality_L1 = torch.tensor(0.7, dtype=dtype, requires_grad=True)
mortality_L2 = torch.tensor(0.7, dtype=dtype, requires_grad=True)
mortality_L3 = torch.tensor(0.7, dtype=dtype, requires_grad=True)
mortality_L4 = torch.tensor(0.7, dtype=dtype, requires_grad=True)
mortality_L5_male = torch.tensor(0.7, dtype=dtype, requires_grad=True)
mortality_L5_L6_female = torch.tensor(0.7, dtype=dtype, requires_grad=True)
mortality_pupae_male = torch.tensor(0.4, dtype=dtype, requires_grad=True)
mortality_pupae_female = torch.tensor(0.4, dtype=dtype, requires_grad=True)
mortality_adults = torch.tensor(0.1, dtype=dtype, requires_grad=True)

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

pop_prediapause = LnormPDF(xs, torch.tensor(0.2), torch.tensor(1.1))
pop_diapause_I = LnormPDF(from_x, torch.tensor(0.2), torch.tensor(1.1))
pop_diapause_D = LnormPDF(to_x, torch.tensor(0.4), torch.tensor(1.1))
pop_diapause = torch.flatten(pop_diapause_I * pop_diapause_D)
pop_postdiapause = LnormPDF(xs, torch.tensor(0.2), torch.tensor(1.1))
pop_L1 = LnormPDF(xs, torch.tensor(0.2), torch.tensor(1.1))
pop_L2 = LnormPDF(xs, torch.tensor(0.2), torch.tensor(1.1))
pop_L3 = LnormPDF(xs, torch.tensor(0.2), torch.tensor(1.1))
pop_L4 = LnormPDF(xs, torch.tensor(0.2), torch.tensor(1.1))
pop_L5_male = LnormPDF(xs, torch.tensor(0.2), torch.tensor(1.1))
pop_L5_L6_female = LnormPDF(xs, torch.tensor(0.2), torch.tensor(1.1))
pop_pupae_male = LnormPDF(xs, torch.tensor(0.2), torch.tensor(1.1))
pop_pupae_female = LnormPDF(xs, torch.tensor(0.2), torch.tensor(1.1))
pop_adult = LnormPDF(xs, torch.tensor(0.2), torch.tensor(1.1))

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