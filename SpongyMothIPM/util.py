import math

import torch

# Automatic differentiaion of the log normal cdf is not numerically stable.
# Here we provide our own backward step that simplifies computations
# to ensure the gradient can be computed.
class LogNormalCDF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, mu, sigma):
        """Compute the log-normal cdf based on x and a tensor of mu/sigma"""
        ctx.save_for_backward(input, mu, sigma)
        result = torch.where(
            input > 0,
            (0.5
            * (1
                + torch.erf(
                    (torch.log(input)
                    - torch.log(mu)) 
                    / (sigma
                    * math.sqrt(2.0))))),
            0)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        x, mu, sigma = ctx.saved_tensors
        dx = torch.where(
            x > 0,
            (1
            / (x
                * sigma
                * (math.sqrt(2.0*math.pi)))
                * torch.exp(
                    - ((torch.log(x)-torch.log(mu))**2)
                    / (2*(sigma**2)))),
            0)
        grad_x = grad_output * dx
        grad_mu = torch.where(
            (x > 0) & (mu > 0),
            (1 / (mu*sigma*math.sqrt(2*math.pi))
             * torch.exp(
                 - (((torch.log(x) - torch.log(mu))
                    / (math.sqrt(2)*sigma))**2))),
            0)
        grad_mu = grad_mu * grad_output
        grad_sigma = torch.where(
            (x > 0) & (mu > 0),
            (1 / math.sqrt(2*math.pi)
             * torch.exp(
                 - (((torch.log(x) - torch.log(mu))
                    / (math.sqrt(2)*sigma))**2))
             * ((torch.log(mu) - torch.log(x))
                / (sigma**2))),
            0)
        grad_sigma = grad_sigma * grad_output
        if torch.any(torch.isinf(grad_sigma)) or torch.any(torch.isnan(grad_sigma)):
            print(grad_sigma)
            print(x)
            print(mu)
            print(sigma)
            raise Exception()
        return grad_x, grad_mu, grad_sigma

def LnormCDF(x, mu, sigma):
    result = LogNormalCDF.apply(x, mu, sigma)
    return result

def LnormPDF(x, mu, sigma):
    result = torch.where(
        x > 0,
        (1
        / (x
            * torch.log(sigma)
            * (math.sqrt(2.0*math.pi)))
            * torch.exp(
                - ((torch.log(x)-torch.log(mu))**2)
                / (2*torch.log(sigma)**2))),
        0)
    return result


def validate(tensor, mu):
    return torch.where(
        torch.isclose(mu, 
                      torch.tensor(0.0)),
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