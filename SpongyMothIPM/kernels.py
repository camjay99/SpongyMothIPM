import torch

import SpongyMothIPM.util as util

class _LifeStage():
    def init_pop(self, total, position, scale):
        self.pop = util.LnormPDF(self.config.xs, 
                                 torch.tensor(position), 
                                 torch.tensor(scale))
        self.pop = self.pop*total/self.pop.sum()
        
    def grow_pop(self, temps):
        kernel = self.build_kernel(temps)
        self.pop = kernel @ self.pop

    def apply_mortality(self):
        self.pop *= 1 - self.mortality

    def get_transfers(self):
        transfers = torch.sum(self.pop*self.config.xs_for_transfer)
        self.pop *= ~self.config.xs_for_transfer
        return transfers
        
    def add_transfers(self, transfers=0):
        self.pop += transfers*self.config.input_xs

    def run_one_step(self, temps, incoming=0):
        if self.save:
            self.save_pop()
        self.apply_mortality()
        self.grow_pop(temps)
        outgoing = self.get_transfers()
        self.add_transfers(incoming)
        return outgoing

    def save_pop(self):
        self.abundances.append(torch.sum(self.pop))
        self.hist_pops.append(self.pop.detach().clone())

    def build_kernel(self, temps):
        if len(temps) == 0:
            raise Exception("Must provide non-empty temps array to build kernel.")
        mu = torch.tensor(0, dtype=self.config.dtype)
        for temp in temps:
            mu = mu + self.calc_mu(temp)
        kernel = util.LnormCDF(self.config.x_dif - 1/(2*(self.config.n_bins-1)), 
                               mu, self.sigma)
        kernel = torch.diff(kernel, 
                            dim=0, 
                            append=torch.ones((1,
                                               self.config.n_bins)))
        kernel = util.validate(kernel, mu)
        return kernel


class Prediapause(_LifeStage):
    def __init__(self, config, save=False, sigma=1.1, mortality=0.1):
        self.config = config
        self.save = save
        if save:
            self.abundances = []
            self.hist_pops = []

        ## Assumed Parameters
        self.rho = torch.tensor(0.1455)
        self.t_max = torch.tensor(33.993)
        self.crit_temp_width = torch.tensor(6.350)
        self.psi = torch.tensor(0.0191)

        ## Optimized Parameters
        self.sigma = torch.tensor(sigma, 
                                  dtype=self.config.dtype, 
                                  requires_grad=True)
        self.mortality = torch.tensor(mortality, 
                                      dtype=self.config.dtype,
                                      requires_grad=True)

    def calc_mu(self, temp):
        mu = (
            self.config.delta_t
            * util.Logan_TM1(temp, 
                             self.psi, 
                             self.rho, 
                             self.t_max, 
                             self.crit_temp_width))
        return mu


    

class Diapause(_LifeStage):
    def __init__(self, config, save=False, sigma_I=1.1, sigma_D=1.1, mortality=0.1):
        self.config = config
        self.save = save
        if save:
            self.abundances = []
            self.hist_pops = []

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
        self.A_min = torch.tensor(0.3)
        self.A_max = torch.tensor(1)
        self.t_min = torch.tensor(-5)
        self.t_max = torch.tensor(25)
        self.alpha = torch.tensor(2.00000)
        self.beta = torch.tensor(0.62062)
        self.gamma = torch.tensor(0.56000)

        ## Optimized Parameters
        self.sigma_I = torch.tensor(sigma_I, 
                                    dtype=self.config.dtype, 
                                    requires_grad=True)
        self.sigma_D = torch.tensor(sigma_D, 
                                    dtype=self.config.dtype, 
                                    requires_grad=True)
        self.mortality = torch.tensor(mortality, 
                                      dtype=config.dtype, 
                                      requires_grad=True)

    def calc_mu_I(self, temp):
        Z = (self.t_max - temp) / (self.t_max - self.t_min)
        rp = 1 + self.rp_c*(torch.exp(Z)**6)
        rs = self.rs_c + self.rs_rp*rp
        # Here we calculate dI/dt from I* = 1 - I
        mu_I = (
            self.config.delta_t
            * (torch.maximum(
                torch.tensor(0.0),
                -1 * (torch.maximum(
                    -1 + self.config.from_I,
                    (torch.log(rp)
                    * ((1 - self.config.from_I) 
                        - self.I_0 
                        - rs)))))))
        # Change is expressed over entire input space, since
        # inhibitor depletion does not depend on development rate
        mu_I = torch.tile(mu_I, (1, self.config.n_bins, 1, 1)) 
        return mu_I
    
    def calc_mu_D(self, temp):
        Z = (self.t_max - temp) / (self.t_max - self.t_min)
        if temp <= self.t_min:
            A = self.A_min
        elif temp >= self.t_max:
            A = self.A_max
        else:
            A = 0.3 + 0.7*(1-Z)**(self.A_1 * (Z**self.A_2))
        pdr = torch.exp(self.c 
                        + self.pdr_t*temp 
                        + self.pdr_t_2*(temp**2) 
                        + self.pdr_t_4*(temp**4))
        mu_D = (
            self.config.delta_t
            * (torch.maximum(torch.tensor(0),
                            (pdr
                            * (1 - (1 - self.config.from_I)*A)))))
        return mu_D

    def build_kernel(self, temps, twoD=True):
        if len(temps) == 0:
            raise Exception("Must provide non-empty temps array to build kernel.")
        # Current strategy is to compute as a 4-D tensor to take advantage of broadcasting, then to 
        # reshape into a 2D matrix to take advantage of matrix multiplication.
        # To simplify calculations, we keep track of 1-I rather than I, so that
        # all traits are always increasing.
        mu_I = torch.tensor(0, dtype=self.config.dtype)
        mu_D = torch.tensor(0, dtype=self.config.dtype)
        for temp in temps:
            mu_I = mu_I + self.calc_mu_I(temp)
            mu_D = mu_D + self.calc_mu_D(temp)
        
        kernel_I_4D = util.LnormCDF(self.config.I_dif, mu_I, self.sigma_I)
        kernel_I_4D = torch.diff(kernel_I_4D, 
                            dim=2, 
                            append=torch.ones((self.config.n_bins,
                                               self.config.n_bins,
                                               1,
                                               1)))

        kernel_D_4D = util.LnormCDF(self.config.D_dif, mu_D, self.sigma_D)
        kernel_D_4D = torch.diff(kernel_D_4D, 
                            dim=3, 
                            append=torch.ones((self.config.n_bins,
                                               self.config.n_bins,
                                               1,
                                               1)))

        kernel_4D = kernel_I_4D * kernel_D_4D
        
        # Nans can be generated as some of the "state space" currently
        # contains unreachable states.
        #kernel_4D = torch.nan_to_num(kernel_4D)
        if twoD:
            # Need to reshape kernel so that it can be 
            # used in matrix-vector multiplication.
            n_bins = self.config.n_bins
            kernel_2D = torch.reshape(kernel_4D, 
                                    (n_bins, n_bins, n_bins*n_bins))
            kernel_2D = torch.permute(kernel_2D, (2, 0, 1))
            kernel_2D = torch.reshape(kernel_2D, 
                                    (n_bins*n_bins, n_bins*n_bins))
            
            # Also reshape to create means
            mu = torch.reshape(mu_I*mu_D, (1, n_bins*n_bins))
            kernel_2D = util.validate(kernel_2D, mu)
            return kernel_2D
        else:
            return kernel_4D
    
    def init_pop(self, total, position_I, scale_I, position_D=None, scale_D=None):
        if (position_D == None) and (scale_D == None):
            position_D = position_I
            scale_D = scale_I
        pop_I = util.LnormPDF(self.config.to_x, 
                              torch.tensor(position_I), 
                              torch.tensor(scale_I))
        pop_D = util.LnormPDF(self.config.from_x, 
                              torch.tensor(position_D), 
                              torch.tensor(scale_D))
        self.pop = torch.flatten(pop_I * pop_D)
        self.pop = self.pop*total/self.pop.sum()

    def get_transfers(self):
        pop_2D = torch.reshape(self.pop, self.config.shape)
        transfers = torch.sum(pop_2D*self.config.grid2d_for_transfer)
        pop_2D *= ~self.config.grid2d_for_transfer
        self.pop = torch.flatten(pop_2D)
        return transfers

    def add_transfers(self, transfers=0):
        pop_2D = torch.reshape(self.pop, self.config.shape)
        pop_2D += transfers*self.config.input_grid2d
        self.pop = torch.flatten(pop_2D)


class Postdiapause(_LifeStage):
    def __init__(self, config, save=False, sigma=1.1, mortality=0.1):
        self.config = config
        self.save = save
        if save:
            self.abundances = []
            self.hist_pops = []

        ## Assumed Parameters
        self.tau = torch.tensor(3.338182*1e-7)
        self.delta = torch.tensor(0.390727)
        self.omega = torch.tensor(-1.821620)
        self.kappa = torch.tensor(0.373854)
        self.psi = torch.tensor(-0.0148244286)
        self.zeta = torch.tensor(0.00001561466667)
        # Starvation of L1 instars prior to finding food
        # Based on Hunter 1993
        self.preincrease = torch.tensor(7.20292573)
        self.changepoint = torch.tensor(14.22353787)
        self.slope = torch.tensor(1.53550927)

        ## Optimized Parameters
        self.sigma = torch.tensor(sigma, 
                                  dtype=self.config.dtype, 
                                  requires_grad=True)
        self.mortality = torch.tensor(mortality, 
                                      dtype=self.config.dtype, 
                                      requires_grad=True)
    
    def calc_mu(self, temp):
        mu = (
            self.config.delta_t
            * (torch.maximum(
                (self.tau + torch.exp(self.delta*temp) # R_T(0)
                 + (self.config.from_x
                    * (self.omega 
                       + self.kappa*temp 
                       + self.psi*temp**2 
                       + self.zeta*temp**3))),
                torch.tensor(0)))) # a_T * A
        return mu
    
    def calc_starvation(self, temp):
        return (((temp < self.changepoint)
                * self.preincrease)
                + ((temp > self.changepoint)
                * (self.slope
                    * (temp - self.changepoint) 
                    + self.preincrease)))
        

class FirstInstar(_LifeStage):
    def __init__(self, config, save=False, sigma=1.1, mortality=0.1):
        self.config = config
        self.save = save
        if save:
            self.abundances = []
            self.hist_pops = []

        ## Assumed parameters
        self.alpha = torch.tensor(0.9643)
        self.kappa = torch.tensor(7.700)
        self.rho = torch.tensor(0.1427)
        self.t_max = torch.tensor(30.87)
        self.crit_temp_width = torch.tensor(12.65)

        ## Optimized Parameters
        self.sigma = torch.tensor(sigma, 
                                  dtype=self.config.dtype, 
                                  requires_grad=True)
        self.mortality = torch.tensor(mortality, 
                                      dtype=self.config.dtype, 
                                      requires_grad=True)
        
    def calc_mu(self, temp):
        mu = (
            self.config.delta_t
            * util.Logan_TM2(temp, 
                             self.alpha, 
                             self.kappa, 
                             self.rho, 
                             self.t_max, 
                             self.crit_temp_width, 
                             10))
        return mu

class SecondInstar(_LifeStage):
    def __init__(self, config, save=False, sigma=1.1, mortality=0.1):
        self.config = config
        self.save = save
        if save:
            self.abundances = []
            self.hist_pops = []

        ## Assumed parameters
        self.psi = torch.tensor(0.1454)
        self.rho = torch.tensor(0.1720)
        self.t_max = torch.tensor(21.09)
        self.crit_temp_width = torch.tensor(4.688)

        ## Optimized Parameters
        self.sigma = torch.tensor(sigma, 
                                  dtype=self.config.dtype, 
                                  requires_grad=True)
        self.mortality = torch.tensor(mortality, 
                                      dtype=self.config.dtype, 
                                      requires_grad=True)
    
    def calc_mu(self, temp):
        mu = (
            self.config.delta_t
            * util.Logan_TM1(temp, 
                             self.psi, 
                             self.rho, 
                             self.t_max, 
                             self.crit_temp_width, 
                             13.3))
        return mu
    
    
class ThirdInstar(_LifeStage):
    def __init__(self, config, save=False, sigma=1.1, mortality=0.1):
        self.config = config
        self.save = save
        if save:
            self.abundances = []
            self.hist_pops = []

        ## Assumed Parameters
        self.alpha = torch.tensor(1.2039)
        self.kappa = torch.tensor(8.062)
        self.rho = torch.tensor(0.1737)
        self.t_max = torch.tensor(24.12)
        self.crit_temp_width = torch.tensor(8.494)

        ## Optimized Parameters
        self.sigma = torch.tensor(sigma, 
                                  dtype=self.config.dtype, 
                                  requires_grad=True)
        self.mortality = torch.tensor(mortality, 
                                      dtype=self.config.dtype, 
                                      requires_grad=True)
    
    def calc_mu(self, temp):
        mu = (
            self.config.delta_t
            * util.Logan_TM2(temp, 
                             self.alpha, 
                             self.kappa, 
                             self.rho, 
                             self.t_max, 
                             self.crit_temp_width, 
                             13.3))
        return mu


class FourthInstar(_LifeStage):
    def __init__(self, config, save=False, sigma=1.1, mortality=0.1):
        self.config = config
        self.save = save
        if save:
            self.abundances = []
            self.hist_pops = []

        ## Assumed Parameters
        self.psi = torch.tensor(0.1120)
        self.rho = torch.tensor(0.1422)
        self.t_max = torch.tensor(22.29)
        self.crit_temp_width = torch.tensor(5.358)

        ## Optimized Parameters
        self.sigma = torch.tensor(sigma, 
                                  dtype=self.config.dtype, 
                                  requires_grad=True)
        self.mortality = torch.tensor(mortality, 
                                      dtype=self.config.dtype, 
                                      requires_grad=True)
    
    def calc_mu(self, temp):
        mu = (
            self.config.delta_t 
            * util.Logan_TM1(temp, 
                             self.psi, 
                             self.rho, 
                             self.t_max, 
                             self.crit_temp_width, 
                             13.3))
        return mu


class FemaleFifthSixthInstar(_LifeStage):
    def __init__(self, config, save=False, sigma=1.1, mortality=0.1):
        self.config = config
        self.save = save
        if save:
            self.abundances = []
            self.hist_pops = []

        ## Assumed Parameters
        self.psi = torch.tensor(0.18496921)
        self.rho = torch.tensor(0.14727929)
        self.t_max = torch.tensor(36.50535344)
        self.crit_temp_width = torch.tensor(6.76039768)

        ## Optimized Parameters
        self.sigma = torch.tensor(sigma, 
                                  dtype=self.config.dtype, 
                                  requires_grad=True)
        self.mortality = torch.tensor(mortality, 
                                      dtype=self.config.dtype, 
                                      requires_grad=True)
    
    def calc_mu(self, temp):
        mu = (
            self.config.delta_t 
            * util.Logan_TM1(temp, 
                             self.psi, 
                             self.rho, 
                             self.t_max, 
                             self.crit_temp_width, 
                             0))
        return mu

    
class MaleFifthInstar(_LifeStage):
    def __init__(self, config, save=False, sigma=1.1, mortality=0.1):
        self.config = config
        self.save = save
        if save:
            self.abundances = []
            self.hist_pops = []

        ## Assumed Parameters
        self.psi = torch.tensor(0.1701305)
        self.rho = torch.tensor(0.14787517)
        self.t_max = torch.tensor(36.24067684)
        self.crit_temp_width = torch.tensor(6.71654206)

        ## Optimized Parameters
        self.sigma = torch.tensor(sigma, 
                                  dtype=self.config.dtype, 
                                  requires_grad=True)
        self.mortality = torch.tensor(mortality, 
                                      dtype=self.config.dtype, 
                                      requires_grad=True)
        
    def calc_mu(self, temp):
        mu = (
            self.config.delta_t 
            * util.Logan_TM1(temp, 
                             self.psi, 
                             self.rho, 
                             self.t_max, 
                             self.crit_temp_width, 
                             0))
        return mu


class FemalePupae(_LifeStage):
    def __init__(self, config, save=False, sigma=1.1, mortality=0.1):
        self.config = config
        self.save = save
        if save:
            self.abundances = []
            self.hist_pops = []

        ## Assumed Parameters
        self.psi = torch.tensor(2.00490155e-02)
        self.rho = torch.tensor(5.70991497e-02)
        self.t_max = torch.tensor(3.29603231e+01)
        self.crit_temp_width = torch.tensor(6.24241402e-01)

        ## Optimized Parameters
        self.sigma = torch.tensor(sigma, 
                                  dtype=self.config.dtype, 
                                  requires_grad=True)
        self.mortality = torch.tensor(mortality, 
                                      dtype=self.config.dtype, 
                                      requires_grad=True)

    def calc_mu(self, temp):
        mu = (
            self.config.delta_t 
            * util.Logan_TM1(temp, 
                             self.psi, 
                             self.rho, 
                             self.t_max, 
                             self.crit_temp_width, 
                             0))
        return mu
    
    
class MalePupae(_LifeStage):
    def __init__(self, config, save=False, sigma=1.1, mortality=0.1):
        self.config = config
        self.save = save
        if save:
            self.abundances = []
            self.hist_pops = []

        ## Assumed Parameters
        self.psi = torch.tensor(1.43475792e-02)
        self.rho = torch.tensor(6.15004658e-02)
        self.t_max = torch.tensor(3.34993288e+01)
        self.crit_temp_width = torch.tensor(9.75671208e-01)

        ## Optimized Parameters
        self.sigma = torch.tensor(sigma, 
                                  dtype=self.config.dtype, 
                                  requires_grad=True)
        self.mortality = torch.tensor(mortality, 
                                      dtype=self.config.dtype, 
                                      requires_grad=True)

    def calc_mu(self, temp):
        mu = (
            self.config.delta_t 
            * util.Logan_TM1(temp, 
                             self.psi, 
                             self.rho, 
                             self.t_max, 
                             self.crit_temp_width, 
                             0))
        return mu
    
    
class Adult(_LifeStage):
    def __init__(self, config, save=False, sigma=1.1, mortality=0.1):
        self.config = config
        self.save = save
        if save:
            self.abundances = []
            self.hist_pops = []

        ## Assumed Parameters
        self.b = torch.tensor(0.062)
        self.m = torch.tensor(0.04)

        ## Optimized Parameters
        self.sigma = torch.tensor(sigma, 
                                  dtype=self.config.dtype, 
                                  requires_grad=True)
        self.mortality = torch.tensor(mortality, 
                                     dtype=self.config.dtype, 
                                     requires_grad=True)
    
    def calc_mu(self, temp):
        mu = (
            self.config.delta_t 
            * (torch.maximum(
                (self.b 
                 + (self.m
                    *(temp-10))),
                torch.tensor(0))))
        return mu

    def get_transfers(adult_females):
        # Basic reproduction function, as we are currently only focusing
        # on early season synchrony. Future versions can include more
        # robust reproduction.
        return 2*super().get_transfers()