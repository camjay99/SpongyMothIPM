import torch

import SpongyMothIPM.util as util

class _LifeStage():
    def init_pop(self, position, scale):
        self.pop = util.LnormPDF(self.config.xs, 
                                 torch.tensor(position), 
                                 torch.tensor(scale))
        
    def grow_pop(self, temp):
        kernel = self.build_kernel(temp)
        self.pop = kernel @ self.pop

    def apply_mortality(self):
        self.pop *= 1 - self.mortality

    def get_transfers(self):
        transfers = torch.sum(self.pop*self.config.xs_for_transfer)
        self.pop *= ~self.config.xs_for_transfer
        return transfers
        
    def add_transfers(self, transfers):
        self.pop += transfers*self.config.input_xs

    def run_one_step(self, temp, incoming=0):
        self.apply_mortality()
        self.grow_pop(temp)
        outgoing = self.get_transfers
        self.add_transfers(incoming)


class Prediapause(_LifeStage):
    def __init__(self, config):
        self.config = config

        ## Assumed Parameters
        self.rho = torch.tensor(0.1455)
        self.t_max = torch.tensor(33.993)
        self.crit_temp_width = torch.tensor(6.350)
        self.psi = torch.tensor(0.0191)

        ## Optimized Parameters
        self.sigma = torch.tensor(3, 
                                  dtype=self.config.dtype, 
                                  requires_grad=True)
        self.mortality = torch.tensor(0.1, 
                                      dtype=self.config.dtype,
                                      requires_grad=True)

    def build_kernel(self, temp):
        mu = (
            self.config.delta_t
            * util.Logan_TM1(temp, 
                             self.psi, 
                             self.rho, 
                             self.t_max, 
                             self.crit_temp_width))
        kernel = (
            util.LnormPDF(
                self.config.x_dif, 
                mu, 
                self.sigmasigma_prediapause))
        return kernel
    

class Diapause(_LifeStage):
    def __init__(self, config):
        self.config = config

        ## Assumed Parameters
        self.c = torch.tensor(-5.627108200)
        self.pdr_t = torch.tensor(0.059969414)
        self.pdr_t_2 = torch.tensor(0.010390411)
        self.pdr_t_4 = torch.tensor(-0.000007987)
        self.rp_c = torch.tensor(0.00042178)
        self.rs_c = torch.tensor(0.7633152)
        self.rs_rp = torch.tensor(-0.6404470)
        self.I_0 = torch.tensor(1.1880)
        self.A_1 = torch.tensor(1.56441438)
        self.A_2 = torch.tensor(0.46354992)
        self.t_min = torch.tensor(-5)
        self.t_max = torch.tensor(25)
        self.alpha = torch.tensor(2.00000)
        self.beta = torch.tensor(0.62062)
        self.gamma = torch.tensor(0.56000)

        ## Optimized Parameters
        self.sigma_I = torch.tensor(3, 
                                    dtype=self.config.dtype, 
                                    requires_grad=True)
        self.sigma_D = torch.tensor(3, 
                                    dtype=self.config.dtype, 
                                    requires_grad=True)
        self.mortality = torch.tensor(0.1, 
                                      dtype=config.dtype, 
                                      requires_grad=True)

    def build_kernel(self, temp, twoD=True):
        # Current strategy is to compute as a 4-D tensor to take advantage of broadcasting, then to 
        # reshape into a 2D matrix to take advantage of matrix multiplication.
        # To simplify calculations, we keep track of 1-I rather than I, so that
        # all traits are always increasing.
        Z = (self.t_max - temp) / (self.t_max - self.t_min)
        rp = 1 + self.rp_c*(torch.exp(Z)**6)
        rs = self.rs_c + self.rs_rp*rp
        # Here we calculate dI/dt from I* = 1 - I
        mu_I = (
            self.config.delta_t
            * (torch.maximum(-1 + self.config.from_I,
                            (torch.log(rp)
                            * ((1 - self.config.from_I) 
                                - self.I_0 
                                - rs))))))
        mu_I = -1*mu_I # dI*/dt = -dI/dt
        # Change is expressed over entire input space, since
        # inhibitor depletion does not depend on development rate
        mu_I = torch.tile(mu_I, (1, self.config.n_bins, 1, 1)) 

        A = 0.3 + 0.7*(1-Z)**(self.A_1 * (Z**self.A_2))
        pdr = torch.exp(self.c 
                        + self.pdr_t*temp 
                        + self.pdr_t_2*(temp**2) 
                        + self.pdr_t_4*(temp**4))
        mu_D = (
            self.config.delta_t
            * (torch.maximum(torch.tensor(0),
                            (pdr
                            * (1 - (1 - self.config.from_I)*A))))))

        kernel_4D = (
            util.LnormPDF(self.config.I_dif, mu_I, self.sigma_I)
            * util.LnormPDF(self.config.D_dif, mu_D, self.sigma_D))

        # Need to reshape kernel so that it can be 
        # used in matrix-vector multiplication.
        n_bins = self.config.n_bins
        kernel_2D = torch.reshape(kernel_4D, 
                                  (n_bins, n_bins, n_bins*n_bins))
        kernel_2D = torch.permute(kernel_2D, (2, 0, 1))
        kernel_2D = torch.reshape(kernel_2D, 
                                  (n_bins*n_bins, n_bins*n_bins))
        
        if twoD:
            return kernel_2D
        else:
            return kernel_4D
    
    def init_pop(self, position_I, scale_I, position_D, scale_D):
        pop_I = util.LnormPDF(self.config.from_x, 
                              torch.tensor(position_I), 
                              torch.tensor(scale_I))
        pop_D = util.LnormPDF(self.config.to_x, 
                              torch.tensor(position_D), 
                              torch.tensor(scale_D))
        self.pop = torch.flatten(pop_I * pop_D)

    def get_transfers_diapause(self):
        pop_2D = torch.reshape(self.pop, config.shape)
        transfers = torch.sum(pop_2D*self.config.grid2d_for_transfer)
        pop_2D *= ~self.config.grid2d_for_transfer
        self.pop = torch.flatten(pop_2D)
        return transfers

    def add_transfers_diapause(self, transfers):
        pop_2D = torch.reshape(self.pop, self.config.shape)
        pop_2D += transfers*self.config.input_xs
        self.pop = torch.flatten(pop_2D)


class Postdiapause(_LifeStage):
    def __init__(self, config):
        self.config = config

        ## Assumed Parameters
        self.tau = torch.tensor(-0.0127)
        self.delta = torch.tensor(0.00297)
        self.omega = torch.tensor(-0.08323)
        self.kappa = torch.tensor(0.01298)
        self.psi = torch.tensor(0.00099)
        self.zeta = torch.tensor(-0.00004)
        # Starvation of L1 instars prior to finding food
        # Based on Hunter 1993
        self.preincrease = torch.tensor(7.20292573)
        self.changepoint = torch.tensor(14.22353787)
        self.slope = torch.tensor(1.53550927)

        ## Optimized Parameters
        self.sigma = torch.tensor(1.1, 
                                  dtype=self.config.dtype, 
                                  requires_grad=True)
        self.mortality = torch.tensor(0.1, 
                                      dtype=self.config.dtype, 
                                      requires_grad=True)
        
    def build_kernel(self, temp):
        mu = (
            self.config.delta_t
            * (self.tau + self.delta*temp # R_T(0)
                + (self.configfrom_x
                   * (self.omega 
                      + self.kappa*temp 
                      + self.psi*temp**2 
                      + self.zeta*temp**3)))) # a_T * A
        kernel = util.LnormPDF(self.config.x_dif, mu, self.sigma)
        return kernel
    
    def calc_starvation(self, temp):
        return (((temp < self.changepoint)
                * self.preincrease)
                + ((temp > self.changepoint)
                * (self.slope
                    * (temp - self.changepoint) 
                    + self.preincrease)))
        

class FirstInstar(_LifeStage):
    def __init__(self, config):
        self.config = config

        ## Assumed parameters
        self.alpha = torch.tensor(0.9643)
        self.kappa = torch.tensor(7.700)
        self.rho = torch.tensor(0.1427)
        self.t_max = torch.tensor(30.87)
        self.crit_temp_width = torch.tensor(12.65)

        ## Optimized Parameters
        self.sigma = torch.tensor(3, 
                                  dtype=self.config.dtype, 
                                  requires_grad=True)
        self.mortality = torch.tensor(0.7, 
                                      dtype=self.config.dtype, 
                                      requires_grad=True)
        
    def build_kernel(self, temp):
        mu = (
            self.config.delta_t
            * util.Logan_TM2(temp, 
                             self.alpha, 
                             self.kappa, 
                             self.rho, 
                             self.t_max, 
                             self.crit_temp_width, 
                             10))
        kernel = util.LnormPDF(self.config.x_dif, mu, self.sigma)
        return kernel
    

class SecondInstar(_LifeStage):
    def __init__(self, config):
        self.config = config

        ## Assumed parameters
        self.psi = torch.tensor(0.1454)
        self.rho = torch.tensor(0.1720)
        self.t_max = torch.tensor(21.09)
        self.crit_temp_width = torch.tensor(4.688)

        ## Optimized Parameters
        self.sigma = torch.tensor(3, 
                                  dtype=self.config.dtype, 
                                  requires_grad=True)
        self.mortality = torch.tensor(0.7, 
                                      dtype=self.config.dtype, 
                                      requires_grad=True)
        
    def build_kernel(self, temp):
        mu = (
            self.config.delta_t
            * util.Logan_TM1(temp, 
                             self.psi, 
                             self.rho, 
                             self.t_max, 
                             self.crit_temp_width, 
                             13.3))
        kernel = util.LnormPDF(self.config.x_dif, mu, self.sigma)
        return kernel
    
    
class ThirdInstar(_LifeStage):
    def __init__(self, config):
        self.config = config

        ## Assumed Parameters
        self.alpha = torch.tensor(1.2039)
        self.kappa = torch.tensor(8.062)
        self.rho = torch.tensor(0.1737)
        self.t_max = torch.tensor(24.12)
        self.crit_temp_width = torch.tensor(8.494)

        ## Optimized Parameters
        self.sigma = torch.tensor(3, 
                                  dtype=self.config.dtype, 
                                  requires_grad=True)
        self.mortality = torch.tensor(0.7, 
                                      dtype=self.config.dtype, 
                                      requires_grad=True)
        
    def build_kernel(self, temp):
        mu = (
            self.config.delta_t
            * util.Logan_TM2(temp, 
                             self.alpha, 
                             self.kappa, 
                             self.rho, 
                             self.t_max, 
                             self.crit_temp_width, 
                             13.3))
        kernel = util.LnormPDF(self.config.x_dif, mu, self.sigma)
        return kernel


class FourthInstar(_LifeStage):
    def __init__(self, config):
        self.config = config

        ## Assumed Parameters
        self.psi = torch.tensor(0.1120)
        self.rho = torch.tensor(0.1422)
        self.t_max = torch.tensor(22.29)
        self.crit_temp_width = torch.tensor(5.358)

        ## Optimized Parameters
        self.sigma = torch.tensor(3, 
                                  dtype=self.config.dtype, 
                                  requires_grad=True)
        self.mortality = torch.tensor(0.7, 
                                      dtype=self.config.dtype, 
                                      requires_grad=True)
        
    def build_kernel(self, temp):
        mu = (
            self.config.delta_t 
            * util.Logan_TM1(temp, 
                             self.psi, 
                             self.rho, 
                             self.t_max, 
                             self.crit_temp_width, 
                             13.3))
        kernel = self.LnormPDF(self.config.x_dif, mu, self.sigma)
        return kernel


class FemaleFifthSixthInstar(_LifeStage):
    def __init__(self, config):
        self.config = config

        ## Assumed Parameters
        self.psi = torch.tensor(0.18496921)
        self.rho = torch.tensor(0.14727929)
        self.t_max = torch.tensor(36.50535344)
        self.crit_temp_width = torch.tensor(6.76039768)

        ## Optimized Parameters
        self.sigma = torch.tensor(3, 
                                  dtype=self.config.dtype, 
                                  requires_grad=True)
        self.mortality = torch.tensor(0.7, 
                                      dtype=self.config.dtype, 
                                      requires_grad=True)
        
    def build_kernel(self, temp): 
        mu = (
            self.config.delta_t 
            * util.Logan_TM1(temp, 
                             self.psi, 
                             self.rho, 
                             self.t_max, 
                             self.crit_temp_width, 
                             0))
        kernel = util.LnormPDF(self.config.x_dif, mu, self.sigma)
        return kernel

    
class MaleFifthInstar(_LifeStage):
    def __init__(self, config):
        self.config = config

        ## Assumed Parameters
        self.psi = torch.tensor(0.1701305)
        self.rho = torch.tensor(0.14787517)
        self.t_max = torch.tensor(36.24067684)
        self.crit_temp_width = torch.tensor(6.71654206)

        ## Optimized Parameters
        self.sigma = torch.tensor(3, 
                                  dtype=self.config.dtype, 
                                  requires_grad=True)
        self.mortality = torch.tensor(0.7, 
                                      dtype=self.config.dtype, 
                                      requires_grad=True)
        
    def build_kernel(self, temp):
        mu = (
            self.config.delta_t 
            * util.Logan_TM1(temp, 
                             self.psi, 
                             self.rho, 
                             self.t_max, 
                             self.crit_temp_width, 
                             0))
        kernel = util.LnormPDF(self.config.x_dif, mu, self.sigma)
        return kernel


class FemalePupae(_LifeStage):
    def __init__(self, config):
        self.config = config

        ## Assumed Parameters
        self.psi = torch.tensor(2.00490155e-02)
        self.rho = torch.tensor(5.70991497e-02)
        self.t_max = torch.tensor(3.29603231e+01)
        self.crit_temp_width = torch.tensor(6.24241402e-01)

        ## Optimized Parameters
        self.sigma = torch.tensor(3, 
                                  dtype=self.config.dtype, 
                                  requires_grad=True)
        self.mortality = torch.tensor(0.4, 
                                      dtype=self.config.dtype, 
                                      requires_grad=True)

    def build_kernel(self, temp):
        mu = (
            self.config.delta_t 
            * util.Logan_TM1(temp, 
                             self.psi, 
                             self.rho, 
                             self.t_max, 
                             self.crit_temp_width, 
                             0))
        kernel = util.LnormPDF(self.config.x_dif, mu, self.sigma)
        return kernel
    
    
class MalePupae(_LifeStage):
    def __init__(self, config):
        self.config = config

        ## Assumed Parameters
        self.psi = torch.tensor(1.43475792e-02)
        self.rho = torch.tensor(6.15004658e-02)
        self.t_max = torch.tensor(3.34993288e+01)
        self.crit_temp_width = torch.tensor(9.75671208e-01)


        self.b = torch.tensor(-0.0238)
        self.m = torch.tensor(0.00362)

        ## Optimized Parameters
        self.sigma = torch.tensor(3, 
                                  dtype=self.config.dtype, 
                                  requires_grad=True)
        self.mortality = torch.tensor(0.1, 
                                      dtype=self.config.dtype, 
                                      requires_grad=True)

    def build_kernel(self, temp):
        ## Calculate kernel
        mu = (
            self.config.delta_t 
            * (self.b 
               + (self.m
                  * self.temp)))
        kernel = util.LnormPDF(self.config.x_dif, mu, self.sigma)
        return kernel
    
    
class Adult(_LifeStage):
    def __init__(self, config):
        self.config = config

        ## Assumed Parameters
        self.b = torch.tensor(0.062)
        self.m = torch.tensor(0.04)

        ## Optimized Parameters
        self.sigma = torch.tensor(3, 
                                  dtype=self.config.dtype, 
                                  requires_grad=True)
        
    def build_method(self, temp):
        mu = (
            self.config.delta_t 
            * (self.b 
               + (self.m
                  *(temp-10))))
        kernel = util.LnormPDF(self.config.x_dif, mu, self.sigma)
        return kernel

    def get_transfers(adult_females):
        # Basic reproduction function, as we are currently only focusing
        # on early season synchrony. Future versions can include more
        # robust reproduction.
        return 2*_LifeStage.get_transfers()