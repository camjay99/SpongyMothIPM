import math

import torch

def LnormPDF(x, mu, sigma):
    """Compute the log-normal pdf based on x and a list of mulog/sigmalog."""
    # When x = 0, lognormal pdf will return nan when it should be 0.
    # Therefore, we slightly nudge values first (this helps ensure
    # differentiability, which zeroing the first entry would not do).
    # When mu = 0, set 1 along the diagonal and rest of value in column
    # to 0.
    x = x + 0.000000001
    dist = (1
            / (x
            * torch.log(sigma)
            * (math.sqrt(2.0*math.pi)))
            * torch.exp(
                - ((torch.log(x)-torch.log(mu))**2)
                / (2*torch.log(sigma)**2)))
    return dist

def validate(tensor):
    return torch.where(tensor.sum(dim=0, keepdim=True) == torch.tensor(0),
                       torch.eye(tensor.shape[0]),
                       tensor)

def Logan_TM1(temp, psi, rho, t_max, crit_temp_width, tbase=0):
    tau = (t_max - temp + tbase) / crit_temp_width
    return torch.maximum(torch.tensor(0),
                         (psi
                          * (torch.exp(rho*(temp - tbase)) 
                             - torch.exp(rho*t_max - tau))))

def Logan_TM2(temp, alpha, kappa, rho, t_max, crit_temp_width, tbase=0):
    tau = (t_max - temp + tbase) / crit_temp_width
    return torch.maximum(torch.tensor(0),
                         (alpha
                          * ((1 + kappa
                              * torch.exp(
                                -rho * (temp-tbase)))**-1
                                - torch.exp(-tau))))