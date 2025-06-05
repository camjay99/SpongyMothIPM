import math

import torch
import matplotlib.pyplot as plt

############################################
# Helper functions (move to separate module)
############################################
def LnormPDF(x, mu, sigma):
    """Compute the log-normal pdf based on x and a list of mulog/sigmalog."""
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
shape = (n_bins, n_bins)
xs = torch.linspace(0.0001, 1, n_bins)
from_x = torch.reshape(xs, (1, n_bins))
to_x = torch.reshape(xs, (n_bins, 1))
delta_t = 1
temp = torch.tensor(15)

#####################
# Pre-diapause kernel
#####################
## Assumed Parameters
rho = torch.tensor(0.1455)
t_max = torch.tensor(33.993) # Maximum temperature before development failure
crit_temp_width = torch.tensor(6.350) # Width of interval below t_max where development decreases rapidly.
psi = torch.tensor(0.0191)

## Optimized Parameters
sigma_prediapause = torch.tensor(1.1, dtype=dtype, requires_grad=True)

## Compute kernel
mu_prediapause = (
    from_x 
    + (delta_t
       * Logan_TM1(temp, psi, rho, t_max, crit_temp_width)))
kern_prediapause = LnormPDF(to_x, mu_prediapause, sigma_prediapause)
kern_prediapause = kern_prediapause / torch.sum(kern_prediapause, dim=0, keepdim=True)

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
sigma_I_diapause = torch.tensor(1.1, dtype=dtype, requires_grad=True)
sigma_D_diapause = torch.tensor(1.1, dtype=dtype, requires_grad=True)

## Compute kernel
# Current strategy is to compute as a 4-D tensor to take advantage of broadcasting, then to 
# reshape into a 2D matrix to take advantage of matrix multiplication
ys = torch.linspace(0.0001, 1, n_bins)
from_I = torch.reshape(xs, (n_bins, 1, 1, 1))
to_I = torch.reshape(xs, (1, 1, n_bins, 1))
from_D = torch.reshape(ys, (1, n_bins, 1, 1))
to_D = torch.reshape(ys, (1, 1, 1, n_bins))

Z = (t_max - temp) / (t_max - t_min)
rp = 1 + rp_c*(torch.log(Z)**6)
rs = rs_c + rs_rp*rp
mu_I_diapause = (
    from_I 
    + (delta_t
       * (torch.maximum(-1*from_I,
                        (torch.log(rp)
                         * (from_I 
                            - I_0 
                            - rs))))))
# Change is expressed over entire input space, since
# inhibitor depletion does not depend on development rate
mu_I_diapause = torch.tile(mu_I_diapause, (1, n_bins, 1, 1)) 

A = 0.3 + 0.7*(1-Z)**(A_1 * (Z**A_2))
pdr = torch.exp(c + pdr_t*temp + pdr_t_2*(temp**2) + pdr_t_4*(temp**4))
mu_D_diapause = (
    from_D
    + (delta_t
       * (torch.maximum(torch.tensor(0),
                        (pdr
                         * (1 - from_I*A))))))

# Using these we calculate lognormal distribution across
# each axis. Multiplying together gets the final probability 
# distribution.
kern_I_diapause = LnormPDF(to_I, mu_I_diapause, sigma_I_diapause)
kern_D_diapause = LnormPDF(to_D, mu_D_diapause, sigma_D_diapause)
kern_diapause = kern_I_diapause * kern_D_diapause
kern_diapause = kern_diapause / torch.sum(kern_diapause, (2, 3), keepdim=True)

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
    from_x
    + (delta_t
       * (tau + delta*temp # R_T(0)
          + (from_x
             * (omega 
                + kappa*temp 
                + psi*temp**2 
                + zeta*temp**3))))) # a_T * A
kern_postdiapause = LnormPDF(to_x, mu_postdiapause, sigma_postdiapause)
kern_postdiapause = kern_postdiapause / torch.sum(kern_postdiapause, dim=0, keepdim=True)

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
sigma_L1 = torch.tensor(1.1, dtype=dtype, requires_grad=True)

## Compute kernel
mu_L1 = (
    from_x
    + (delta_t
       * Logan_TM2(temp, alpha, kappa, rho, t_max, crit_temp_width, 10)))
kern_L1 = LnormPDF(to_x, mu_L1, sigma_L1)
kern_L1 = kern_L1 / torch.sum(kern_L1, dim=0, keepdim=True)

###########
# L2 kernel
###########
## Assumed parameters
psi = torch.tensor(0.1454)
rho = torch.tensor(0.1720)
t_max = torch.tensor(21.09)
crit_temp_width = torch.tensor(4.688)

## Optimized Parameters
sigma_L2 = torch.tensor(1.1, dtype=dtype, requires_grad=True)

## Compute kernel
mu_L2 = (
    from_x 
    + (delta_t
       * Logan_TM1(temp, psi, rho, t_max, crit_temp_width, 13.3)))
kern_L2 = LnormPDF(to_x, mu_L2, sigma_L2)
kern_L2 = kern_L2 / torch.sum(kern_L2, dim=0, keepdim=True)

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
sigma_L3 = torch.tensor(1.1, dtype=dtype, requires_grad=True)

## Compute kernel
mu_L3 = (
    from_x
    + (delta_t
       * Logan_TM2(temp, alpha, kappa, rho, t_max, crit_temp_width, 13.3)))
kern_L3 = LnormPDF(to_x, mu_L3, sigma_L3)
kern_L3 = kern_L3 / torch.sum(kern_L3, dim=0, keepdim=True)

###########
# L4 kernel
###########
## Assumed Parameters
psi = torch.tensor(0.1120)
rho = torch.tensor(0.1422)
t_max = torch.tensor(22.29)
crit_temp_width = torch.tensor(5.358)

## Optimized Parameters
sigma_L4 = torch.tensor(1.1, dtype=dtype, requires_grad=True)

## Compute kernel
mu_L4 = (
    from_x
    + (delta_t 
       * Logan_TM1(temp, psi, rho, t_max, crit_temp_width, 13.3)))
kern_L4 = LnormPDF(to_x, mu_L4, sigma_L4)
kern_L4 = kern_L4 / torch.sum(kern_L4, dim=0, keepdim=True)

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
sigma_L5_L6_female = torch.tensor(1.1, dtype=dtype, requires_grad=True)

## Compute kernel
mu_L5_L6_female = (
    from_x
    + (delta_t 
       * (b + m*scaling*temp)))
kern_L5_L6_female = LnormPDF(to_x, mu_L5_L6_female, sigma_L5_L6_female)
kern_L5_L6_female = kern_L5_L6_female / torch.sum(kern_L5_L6_female, dim=0, keepdim=True)

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
sigma_L5_male = torch.tensor(1.1, dtype=dtype, requires_grad=True)

## Compute kernel
mu_L5_male = (
    from_x
    + (delta_t
       * (b + m*scaling*temp)))
kern_L5_male = LnormPDF(to_x, mu_L5_male, sigma_L5_male)
kern_L5_male = kern_L5_male / torch.sum(kern_L5_male, dim=0, keepdim=True)

#######################
# Pupae kernel (Female)
#######################
## Assumed Parameters
b = torch.tensor(-0.0217)
m = torch.tensor(0.00427)

## Optimized Parameters
sigma_pupae_female = torch.tensor(1.1, dtype=dtype, requires_grad=True)

## Calculate kernel
mu_pupae_female = (
    from_x
    + (delta_t 
       * (b + m*temp)))
kern_pupae_female = LnormPDF(to_x, mu_pupae_female, sigma_pupae_female)
kern_pupae_female = kern_pupae_female / torch.sum(kern_pupae_female, dim=0, keepdim=True)

#####################
# Pupae kernel (male)
#####################
## Assumed Parameters
b = torch.tensor(-0.0238)
m = torch.tensor(0.00362)

## Optimized Parameters
sigma_pupae_male = torch.tensor(1.1, dtype=dtype, requires_grad=True)

## Calculate kernel
mu_pupae_male = (
    from_x
    + (delta_t 
       * (b + m*temp)))
kern_pupae_male = LnormPDF(to_x, mu_pupae_male, sigma_pupae_male)
kern_pupae_male = kern_pupae_male / torch.sum(kern_pupae_male, dim=0, keepdim=True)

##############
# Adult kernel
##############
## Assumed Parameters
b = torch.tensor(0.062)
m = torch.tensor(0.04)
## Optimized Parameters
sigma_adult = torch.tensor(1.1, dtype=dtype, requires_grad=True)

## Calculate kernel
mu_adult = (
    from_x
    + (delta_t 
       * (b + m*(temp-10))))
kern_adult = LnormPDF(to_x, mu_adult, sigma_adult)
kern_adult = kern_adult / torch.sum(kern_adult, dim=0, keepdim=True)

# test = LnormPDF(xs, torch.tensor(0.2), torch.tensor(1.1))
# test = test / test.sum()

# for i in range(4):
#     plt.plot(test.detach(), label=i)
#     test = kern_postdiapause @ test
# plt.show()

