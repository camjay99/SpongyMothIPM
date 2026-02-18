"""Classes for generating kernels for various life stages based on temperature.

This module contains all of the functionality required for generating projection
kernels and maintaining an associated population. The _LifeStage helper class 
implements basic functionality common to all kernel generators such as 
initialization, saving, and moving individuals between populations; while other
classes implement specific kernel generating functions for those life stages. 

Classes that can be imported from this module:
- Prediapause
- Diapause
- Postdiapause
- FirstInstar
- SecondInstar
- ThirdInstar
- FourthInstar
- FemaleFifthSixthInstar
- MaleFifthInstar
- FemalePupae
- MalePupae
- Adult
"""


import os

import numpy as np
import pandas as pd
import torch

import SpongyMothIPM.util as util

class _LifeStage():
    def __init__(self, 
                 config,
                 save=False, 
                 file_path='', 
                 save_rate=5, 
                 write_rate=10, 
                 precision=4):
        """Initializes the instance based on saving preferences.

        Args:
          config:  A Config instance describing global settings.
          save:  A boolean indicating whether to save population information.
          file_path: A string indicating where population info will be saved,
            generally into a .csv file. Passing "memory" will save into an
            in-memory list that can be accessed through the `saved_pops` field.
          save_rate:  Number of iterations before saving population information.
          write_rate:  Number of saves befores writing buffer out to disk.
          precision:  Number of digits after decimal point to use while saving
            floating points.
        """

        # Config provides global settings and utilities
        self.config = config
        # Parameters for saving results
        self.save = save 
        self.file_path = file_path 
        self.save_rate = save_rate
        self.write_rate = write_rate
        self.precision = precision
        # Initialize buffers for saved information
        if self.file_path == 'memory':
            self.saved_pops = []
        self.years = [] 
        self.ydays = []
        self.hist_pops = []
        # Counters for when to save and write
        self.num_iters = 0
        self.num_saves = 0
    
    def init_kernel_helpers(self, n_bins, min_x, max_x):
        """Initializes helper data needed for computing kernels.

        Args:
          n_bins:  An integer representing the number of bins to separate 
            developmental stages into.
          min_x:  A float representing the minimum developmental age to track. 
            The first bin will be centered on this value.
          max_x:  A float representing the maximum developmental age to track. 
            The last bin will be centered on this value.
        """

        # Save settings for kernel
        self.n_bins = n_bins
        self.min_x = min_x
        self.max_x = max_x
        # Create helper tensors for computing kernels
        self.shape = (self.n_bins, self.n_bins)
        ## Variables values of bin centers
        self.xs = torch.linspace(self.min_x, self.max_x, self.n_bins)
        ## Reshape bin center tensors for broadcasting
        self.from_x = torch.reshape(self.xs, (1, self.n_bins))
        self.to_x = torch.reshape(self.xs, (self.n_bins, 1))
        ## Create tensors representing growth increments
        ## These are used to calculate probabilities of moving between bins
        self.x_dif = torch.maximum(torch.tensor(0), self.to_x - self.from_x)
        ## Create tensors for adding/removing individuals from this life stage.
        self.xs_for_transfer = self.xs >= 1
        self.input_xs = torch.zeros_like(self.xs)
        self.input_xs[0] = 1

    def init_pop(self, total, location, scale):
        """Initializes a sample population using a lognormal distribution.

        Initializes a tensor attribute `pop` representing the sample population 
        using a lognormal distribution. This tensor will have size matching the
        attribute `xs`, so init_kernel_helpers must be called prior to calling
        this method.

        Args:
          total:  A float indicating the total population density.
          location: A float representing the location of the lognormal 
            distribution used to initialize the population.
          scale: A float representing the scale of the lognormal distribution
            used to initialize the population.
        """

        self.pop = util.LnormPDF(self.xs, 
                                 torch.tensor(location), 
                                 torch.tensor(scale))
        self.pop = self.pop*total/self.pop.sum()
        
    def project_pop(self, temps):
        """Projects the population forward one day.
        
        Args:
          temps: A list of floats representing sub-daily temperatures. The 
            length of the list should match the number of sub-daily time 
            periods (Currently not enforced).
        """
        kernel = self.build_kernel(temps)
        self.pop = kernel @ self.pop

    def apply_mortality(self):
        """Decreases population total density based on mortality.
        
        Decreases population total density by (1-mortality). Mortality is 
        applied in a development age-independent fashion.
        """

        self.pop = self.pop*(1 - self.mortality)

    def get_transfers(self):
        """Removes population density that has developed out of this stage.
        
        Returns:
          A float representing the total density of population that has 
          developed out of this stage.
        """

        transfers = torch.sum(self.pop*self.xs_for_transfer)
        self.pop = self.pop*~self.xs_for_transfer
        return transfers
        
    def add_transfers(self, transfers=0):
        """Adds population density to initial development bin.
        
        Args:
          transfers: A float representing the density of individuals to be added
            to this stage.
        """

        self.pop = self.pop + transfers*self.input_xs

    def run_one_step(self, met, incoming=0):
        """Completes one daily time step of the model.
        
        Completes an entire daily time step. This follows the following order:
        1) Get temperatures,
        2) Apply mortality,
        3) Build kernel and project population,
        4) Remove outgoing population density,
        5) Add incoming population density,
        6) Save population status.
        
        Args:
          met:  A Pandas Dataframe representing temperatures and time stamps.
          incoming:  A float representing the incoming population density from 
            the previous stage.

        Returns:
          A float representing the outgoing population density from this stage.
        """

        temps = self._validate_temps(met)
        self.apply_mortality()
        self.project_pop(temps)
        outgoing = self.get_transfers()
        self.add_transfers(incoming)
        if self.save:
            year = met['year'].iloc[0]
            yday = met['yday'].iloc[0]
            self.save_pop(year, yday)
        return outgoing
    
    def write(self):
        """Writes population information.

        Writes population information to disk (or memory if `file_path` is set 
        to 'memory'). Precision of values written disk are determined by the
        `precision` attribute.
        """

        # Saves output to disk.
        if self.file_path == 'memory':
            self.saved_pops.extend(self.hist_pops)
        else:
            # Aggregate outputs into single array and turn convert into
            # Pandas DataFrame which has nicer writing utilities.
            hist_pops = [pop.numpy().reshape(1, -1) 
                         for pop in self.hist_pops]
            arr = np.concatenate(hist_pops, axis=0)
            index = pd.MultiIndex.from_arrays([self.years, self.ydays],
                                            names=['year', 'yday'])
            df = pd.DataFrame(data=arr, 
                            index=index,
                            columns=self.xs.numpy())
            df.to_csv(self.file_path, 
                    mode='a', # Append to the write file if is exists
                    header = not os.path.exists(self.file_path), # Add Header once.
                    float_format=f'{{:.{self.precision}f}}'.format)

    def save_pop(self, year, yday):
        """Saves population information for writing.

        Saves population information for writing every `save_rate` calls. After
        `write_rate` saves, information in the buffer is written to disk and the
        buffer is flushed.

        Args:
          year:  An integer representing the current year of the model.
          yday: An integer representing the current day of year of the model.
        """

        # Records current population status. If the save buffer
        # has been filled, then initiates a write to disk.
        if (self.num_iters % self.save_rate) == 0:
            self.hist_pops.append(self.pop.detach())
            self.years.append(year)
            self.ydays.append(yday)
            self.num_saves += 1
            if (self.num_saves % self.write_rate) == 0:
                self.write()
                self.years = []
                self.ydays = []
                self.hist_pops = []
        self.num_iters += 1

    def _validate_temps(self, met):
        """Validates that a proper Dataframe of temperatures has been passed.
        
        Args:
          met:  A Pandas Dataframe representing temperatures and associated
            time stamps for the current model step.

        Returns:
          A list of floats representing sub-daily temperatures.
        """

        if type(met) is pd.DataFrame:
            # Check that all day nums are the same
            days = met['yday'].to_numpy()
            if (days[0] != days).any():
                raise Exception("Provided temps must correspond to a single day.")
            temps = met['temp']
        else:
            temps = met
        if len(temps) == 0:
            raise Exception("Must provide non-empty temps array to build kernel.")
        return temps
    
    def build_kernel(self, temps):
        """Computes a projection kernel for a one-day time step.
        
        Computes a daily projection kernel based on provided temps. Average
        development for a day is assumed to be the sum of development rates over
        sub-daily time steps. Note, this method requires a `calc_mu` method,
        which must be defined by inheriting classes.

        Args:
          temps:  A list of floats representing sub-daily temperatures.

        Returns:
          A PyTorch Tensor representing a projection kernel for a one-day
          time step.
        """

        mu = torch.tensor(0, dtype=self.config.dtype)
        for temp in temps:
            mu = mu + self.calc_mu(temp)
        kernel = util.LnormCDF(self.x_dif - 1/(2*(self.n_bins-1)), 
                               mu, self.sigma)
        kernel = torch.diff(kernel, 
                            dim=0, 
                            append=torch.ones((1,
                                               self.n_bins)))
        kernel = util.validate(kernel, mu)
        return kernel


class Prediapause(_LifeStage):
    def __init__(self, 
                 config, 
                 n_bins=100,
                 min_x=0,
                 max_x=1,
                 save=False,
                 file_path='',
                 save_rate=5, 
                 write_rate=10, 
                 precision=4, 
                 sigma=1.1, 
                 mortality=0.1):
        """Initializes a Prediapause life stage.
        
        Args:
          config, n_bins, min_x, max_x, save, file_path, save_rate, 
            write_rate, precision:  See _LifeStage.__init__ for details.
          sigma:  A float representing the initial shape to be used when
            generating developmental variability in kernels.
          mortality:  The default mortality rate to be applied each time step.
        """

        super().__init__(config, save, file_path, save_rate, 
                         write_rate, precision)
        self.init_kernel_helpers(n_bins, min_x, max_x)

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
        """Calculates mean development under in the specified temperature.
        
        Args:
          temp: A float representing the current temperature.

        Returns:
          A 0-dim tensor representing the mean development under one sub-daily
          time step at the specified temperature.
        """
        mu = (
            self.config.delta_t
            * util.Logan_TM1(temp, 
                             self.psi, 
                             self.rho, 
                             self.t_max, 
                             self.crit_temp_width))
        return mu
    

class Diapause(_LifeStage):
    def __init__(self, 
                 config, 
                 save=False,
                 file_path='',
                 save_rate=5, 
                 write_rate=10, 
                 precision=4, 
                 n_bins_I=100, 
                 min_I=0,
                 max_I=1.1880,
                 sigma_I=1.5, 
                 n_bins_D=100, 
                 min_D=0,
                 max_D=1,
                 sigma_D=1.5, 
                 mortality=0.1):
        """Initializes a Diapause life stage.
        
        Args:
          config, save, file_path, save_rate, write_rate, precision:  
            See _LifeStage.__init__ for details.
          n_bins_I:  An integer representing the number of bins to use for the
            inhibitor concentration.
          min_I:  A float representing the smallest amount of inhibitor depleted
            tracked. The first bin will be centered on this value. Inhibitor 
            concentration is tracked as the concentration depleted to facilitate
              computations.
          max_I:  A float representing the largest amount of inhibitor depleted 
            tracked. The last bin will be centered on this value. Inhibitor 
            concentration is tracked as the concentration depleted to facilitate 
            computations.
          sigma_I:  A float representing the initial shape to be used when
            generating inhibitor depletion variability in kernels.
          n_bins_D:  An integer representing the number of bins to use for the
            developmental age.
          min_D:  A float representing the smallest developmental age tracked. 
            The first bin will be centered on this value.
          max_D:  A float representing the largest developmental age tracked. 
            The last bin will be centered on this value.
          sigma_D:  A float representing the initial shape to be used when
            generating developmental variability in kernels.
          mortality:  The default mortality rate to be applied each time step.
        """

        super().__init__(config, save, file_path, save_rate, 
                         write_rate, precision)
        self.init_kernel_helpers(n_bins_I, min_I, max_I, n_bins_D, min_D, max_D)

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
                                      dtype=self.config.dtype, 
                                      requires_grad=True)
        
    def init_kernel_helpers(self, 
                            n_bins_I, 
                            min_I, 
                            max_I, 
                            n_bins_D, 
                            min_D, 
                            max_D):
        """Initializes helper data needed for computing kernels.

        Args:
          n_bins_I:  An integer representing the number of bins to use for the
            inhibitor concentration.
          min_I:  A float representing the smallest amount of inhibitor depleted
            tracked. The first bin will be centered on this value. Inhibitor 
            concentration is tracked as the concentration depleted to facilitate
              computations.
          max_I:  A float representing the largest amount of inhibitor depleted 
            tracked. The last bin will be centered on this value. Inhibitor 
            concentration is tracked as the concentration depleted to facilitate 
            computations.
          n_bins_D:  An integer representing the number of bins to use for the
            developmental age.
          min_D:  A float representing the smallest developmental age tracked. 
            The first bin will be centered on this value.
          max_D:  A float representing the largest developmental age tracked. 
            The last bin will be centered on this value.
        """
                
        # Save settings for kernel
        self.n_bins_I = n_bins_I
        self.min_I = min_I
        self.max_I = max_I
        self.n_bins_D = n_bins_D
        self.min_D = min_D
        self.max_D = max_D

        # Create helper tensors for computing kernels
        self.shape = (n_bins_I, n_bins_D)
        ## Variables values of bin centers
        self.Is = torch.linspace(min_I, max_I, n_bins_I)
        self.Ds = torch.linspace(min_D, max_D, n_bins_D) 
        ## Reshape bin center tensors for broadcasting
        self.from_I = torch.reshape(self.Is, (n_bins_I, 1, 1, 1)) 
        self.to_I = torch.reshape(self.Is, (1, 1, n_bins_I, 1))
        self.from_D = torch.reshape(self.Ds, (1, n_bins_D, 1, 1))
        self.to_D = torch.reshape(self.Ds, (1, 1, 1, n_bins_D))
        ## Create tensors representing growth increments
        ## These are used to calculate probabilities of moving between bins
        self.I_dif = torch.maximum(torch.tensor(0), self.to_I - self.from_I)
        self.D_dif = torch.maximum(torch.tensor(0), self.to_D - self.from_D)
        ## Create tensors for adding/removing individuals from this life stage.
        self.grid2d = torch.squeeze(torch.ones_like(self.from_I)*self.from_D)
        self.grid2d_for_transfer = self.grid2d >= 1
        self.input_grid2d = torch.zeros_like(self.grid2d)
        self.input_grid2d[0, 0] = 1

    def calc_mu_I(self, temp):
        """Calculates mean inhibitor depletion in the specified temperature.
        
        Args:
          temp: A float representing the current temperature.

        Returns:
          A tensor representing the mean inhibitor depletion under one 
          sub-daily time step at the specified temperature. This tensor has 
          shape (num_I, num_D, 1, 1). Mean values vary across the first 
          dimension but are constant across the second dimension.
        """

        Z = (self.t_max - temp) / (self.t_max - self.t_min)
        rp = 1 + self.rp_c*(torch.exp(Z)**6)
        rs = self.rs_c + self.rs_rp*rp
        # Here we calculate dI/dt from I* = I_0 - I
        mu_I = (
            self.config.delta_t
            * (torch.maximum(
                torch.tensor(0.0),
                -1 * (torch.maximum(
                    -self.I_0 + self.from_I,
                    (torch.log(rp)
                    * (-self.from_I
                       - rs)))))))
        # Change is expressed over entire input space, since
        # inhibitor depletion does not depend on development rate
        mu_I = torch.tile(mu_I, (1, self.n_bins_D, 1, 1)) 
        return mu_I
    
    def calc_mu_D(self, temp):
        """Calculates mean development in the specified temperature.
        
        Args:
          temp: A float representing the current temperature.

        Returns:
          A tensor representing the mean development under one sub-daily time 
          step at the specified temperature. This tensor has shape 
          (num_I, num_D, 1, 1). Mean values vary across both of these 
          dimensions.
        """

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
                            * (1 - (self.I_0 - self.from_I)*A)))))
        return mu_D

    def build_kernel(self, temps, twoD=True):
        """Computes a projection kernel for a one-day time step.
        
        Computes a daily projection kernel based on provided temps. Average
        development for a day is assumed to be the sum of development rates over
        sub-daily time steps. Note, this method requires a `calc_mu` method,
        which must be defined by inheriting classes.

        Args:
          temps:  A list of floats representing sub-daily temperatures.
          twoD:  A boolean representing whether the kernel should be reshaped
            into two dimensions.

        Returns:
          A PyTorch Tensor of shape (num_I, num_D, num_I, num_D) (or
          (num_I*num_D, num_I*numD) if `twoD` is specified) representing a 
          projection kernel for a one-day time step.
        """

        # Current strategy is to compute as a 4-D tensor to take advantage of broadcasting, then to 
        # reshape into a 2D matrix to take advantage of matrix multiplication.
        # To simplify calculations, we keep track of 1-I rather than I, so that
        # all traits are always increasing.
        mu_I = torch.tensor(0, dtype=self.config.dtype)
        mu_D = torch.tensor(0, dtype=self.config.dtype)
        for temp in temps:
            mu_I = mu_I + self.calc_mu_I(temp)
            mu_D = mu_D + self.calc_mu_D(temp)
        
        kernel_I_4D = util.LnormCDF(self.I_dif - 1/(2*(self.n_bins_I-1)), 
                                    mu_I, self.sigma_I)
        kernel_I_4D = torch.diff(kernel_I_4D, 
                            dim=2, 
                            append=torch.ones((self.n_bins_I,
                                               self.n_bins_D,
                                               1,
                                               1)))

        kernel_D_4D = util.LnormCDF(self.D_dif - 1/(2*(self.n_bins_D-1)), 
                                    mu_D, self.sigma_D)
        kernel_D_4D = torch.diff(kernel_D_4D, 
                            dim=3, 
                            append=torch.ones((self.n_bins_I,
                                               self.n_bins_D,
                                               1,
                                               1)))

        kernel_4D = kernel_I_4D * kernel_D_4D
        
        # Nans can be generated as some of the "state space" currently
        # contains unreachable states.
        #kernel_4D = torch.nan_to_num(kernel_4D)
        if twoD:
            # Need to reshape kernel so that it can be 
            # used in matrix-vector multiplication.
            kernel_2D = torch.reshape(kernel_4D, 
                                      (self.n_bins_I, 
                                       self.n_bins_D, 
                                       self.n_bins_I*self.n_bins_D))
            kernel_2D = torch.permute(kernel_2D, (2, 0, 1))
            kernel_2D = torch.reshape(kernel_2D, 
                                      (self.n_bins_I*self.n_bins_D, 
                                       self.n_bins_I*self.n_bins_D))
            
            # Also reshape to create means
            mu = torch.reshape(mu_I+mu_D, (1, self.n_bins_I*self.n_bins_D))
            kernel_2D = util.validate(kernel_2D, mu)
            return kernel_2D
        else:
            return kernel_4D
    
    def init_pop(self, total, location_I, scale_I, 
                 location_D=None, scale_D=None):
        """Initializes a sample population using a lognormal distribution.

        Initializes a tensor attribute `pop` representing the sample population 
        using a lognormal distribution. This tensor will have size matching the
        attribute `xs`, so init_kernel_helpers must be called prior to calling
        this method.

        Args:
          total:  A float indicating the total population density.
          location_I: A float representing the location of the lognormal 
            distribution used to initialize the inhibitor concentrations
            of the population.
          scale_I: A float representing the scale of the lognormal distribution
            used to initialize the inhibitor concentrations of the population.
          location_D: A float representing the location of the lognormal 
            distribution used to initialize the developmental age of the 
            population. If None, uses the same value as `location_I`.
          scale_D: A float representing the scale of the lognormal distribution
            used to initialize the developmental age of the population. If none,
            uses the same value as `scale_I`
        """
        if (location_D == None) and (scale_D == None):
            location_D = location_I
            scale_D = scale_I
        pop_I = util.LnormPDF(self.from_I, 
                              torch.tensor(location_I), 
                              torch.tensor(scale_I))
        pop_D = util.LnormPDF(self.from_D, 
                              torch.tensor(location_D), 
                              torch.tensor(scale_D))
        self.pop = torch.flatten(pop_I * pop_D)
        self.pop = self.pop*total/self.pop.sum()

    def get_transfers(self):
        """Removes population density that has developed out of this stage. 
        
        Returns:
          A float representing the total density of population that has 
          developed out of this stage. This is solely determined by 
          developmental age, and not by inhibitor concetration.
        """

        pop_2D = torch.reshape(self.pop, self.shape)
        transfers = torch.sum(pop_2D*self.grid2d_for_transfer)
        pop_2D = pop_2D*~self.grid2d_for_transfer
        self.pop = torch.flatten(pop_2D)
        return transfers

    def add_transfers(self, transfers=0):
        """Adds population density to initial development bin.
        
        Args:
          transfers: A float representing the density of individuals to be added
            to this stage.
        """
                
        pop_2D = torch.reshape(self.pop, self.shape)
        pop_2D = pop_2D + transfers*self.input_grid2d
        self.pop = torch.flatten(pop_2D)

    def write(self):
        """Writes population information.

        Writes population information to disk (or memory if `file_path` is set 
        to 'memory'). Precision of values written disk are determined by the
        `precision` attribute.
        """

        # Aggregate outputs into single array and turn convert into
        # Pandas DataFrame which has nicer writing utilities.
        hist_pops = [pop.numpy().reshape(1, -1) 
                         for pop in self.hist_pops]
        arr = np.concatenate(hist_pops, axis=0)
        labels = [f'{x:.{self.precision}e}_{y:.{self.precision}e}' 
                  for x in self.Is for y in self.Ds]
        index = pd.MultiIndex.from_arrays([self.years, self.ydays],
                                          names=['year', 'yday'])
        df = pd.DataFrame(data=arr, 
                          index=index,
                          columns=labels)
        df.to_csv(self.file_path, 
                  mode='a', # Append to the write file if is exists
                  header = not os.path.exists(self.file_path), # Add Header once.
                  float_format=f'{{:.{self.precision}e}}'.format)


class Postdiapause(_LifeStage):
    def __init__(self, 
                 config, 
                 n_bins=100,
                 min_x=0,
                 max_x=1,
                 save=False, 
                 file_path='',
                 save_rate=5, 
                 write_rate=10, 
                 precision=4, 
                 sigma=1.1, 
                 mortality=0.1):
        """Initializes a Postdiapause life stage.
        
        Args:
          config, n_bins, min_x, max_x, save, file_path, save_rate, 
            write_rate, precision:  See _LifeStage.__init__ for details.
          sigma:  A float representing the initial shape to be used when
            generating developmental variability in kernels.
          mortality:  The default mortality rate to be applied each time step.
        """

        super().__init__(config, save, file_path, save_rate, 
                         write_rate, precision)
        self.init_kernel_helpers(n_bins, min_x, max_x)

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
        """Calculates mean development under in the specified temperature.
        
        Args:
          temp: A float representing the current temperature.

        Returns:
          A 0-dim tensor representing the mean development under one sub-daily
          time step at the specified temperature.
        """

        mu = (
            self.config.delta_t
            * (torch.maximum(
                (self.tau + torch.exp(self.delta*temp) # R_T(0)
                 + (self.from_x
                    * (self.omega 
                       + self.kappa*temp 
                       + self.psi*temp**2 
                       + self.zeta*temp**3))),
                torch.tensor(0)))) # a_T * A
        return mu
    
    def calc_starvation(self, temp):
        """Calculates starvation rate based on current temperature."""
        return (((temp < self.changepoint)
                * self.preincrease)
                + ((temp > self.changepoint)
                * (self.slope
                    * (temp - self.changepoint) 
                    + self.preincrease)))
        

class FirstInstar(_LifeStage):
    def __init__(self, 
                 config,
                 n_bins=100,
                 min_x=0,
                 max_x=1, 
                 save=False, 
                 file_path='',
                 save_rate=5, 
                 write_rate=10, 
                 precision=4, 
                 sigma=1.1, 
                 mortality=0.1):
        """Initializes a FirstInstar life stage.
        
        Args:
          config, n_bins, min_x, max_x, save, file_path, save_rate, 
            write_rate, precision:  See _LifeStage.__init__ for details.
          sigma:  A float representing the initial shape to be used when
            generating developmental variability in kernels.
          mortality:  The default mortality rate to be applied each time step.
        """

        super().__init__(config, save, file_path, save_rate, 
                         write_rate, precision)
        self.init_kernel_helpers(n_bins, min_x, max_x)

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
        """Calculates mean development under in the specified temperature.
        
        Args:
          temp: A float representing the current temperature.

        Returns:
          A 0-dim tensor representing the mean development under one sub-daily
          time step at the specified temperature.
        """

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
    def __init__(self, 
                 config, 
                 n_bins=100,
                 min_x=0,
                 max_x=1,
                 save=False,
                 file_path='', 
                 save_rate=5, 
                 write_rate=10, 
                 precision=4, 
                 sigma=1.1, 
                 mortality=0.1):
        """Initializes a SecondInstar life stage.
        
        Args:
          config, n_bins, min_x, max_x, save, file_path, save_rate, 
            write_rate, precision:  See _LifeStage.__init__ for details.
          sigma:  A float representing the initial shape to be used when
            generating developmental variability in kernels.
          mortality:  The default mortality rate to be applied each time step.
        """

        super().__init__(config, save, file_path, save_rate, 
                         write_rate, precision)
        self.init_kernel_helpers(n_bins, min_x, max_x)

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
        """Calculates mean development under in the specified temperature.
        
        Args:
          temp: A float representing the current temperature.

        Returns:
          A 0-dim tensor representing the mean development under one sub-daily
          time step at the specified temperature.
        """

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
    def __init__(self, 
                 config,
                 n_bins=100,
                 min_x=0,
                 max_x=1, 
                 save=False, 
                 file_path='',
                 save_rate=5, 
                 write_rate=10, 
                 precision=4, 
                 sigma=1.1, 
                 mortality=0.1):
        """Initializes a ThirdInstar life stage.
        
        Args:
          config, n_bins, min_x, max_x, save, file_path, save_rate, 
            write_rate, precision:  See _LifeStage.__init__ for details.
          sigma:  A float representing the initial shape to be used when
            generating developmental variability in kernels.
          mortality:  The default mortality rate to be applied each time step.
        """

        super().__init__(config, save, file_path, save_rate, 
                         write_rate, precision)
        self.init_kernel_helpers(n_bins, min_x, max_x)

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
        """Calculates mean development under in the specified temperature.
        
        Args:
          temp: A float representing the current temperature.

        Returns:
          A 0-dim tensor representing the mean development under one sub-daily
          time step at the specified temperature.
        """

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
    def __init__(self, 
                 config, 
                 n_bins=100,
                 min_x=0,
                 max_x=1,
                 save=False, 
                 file_path='',
                 save_rate=5, 
                 write_rate=10, 
                 precision=4, 
                 sigma=1.1, 
                 mortality=0.1):
        """Initializes a FourthInstar life stage.
        
        Args:
          config, n_bins, min_x, max_x, save, file_path, save_rate, 
            write_rate, precision:  See _LifeStage.__init__ for details.
          sigma:  A float representing the initial shape to be used when
            generating developmental variability in kernels.
          mortality:  The default mortality rate to be applied each time step.
        """

        super().__init__(config, save, file_path, save_rate, 
                         write_rate, precision)
        self.init_kernel_helpers(n_bins, min_x, max_x)

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
        """Calculates mean development under in the specified temperature.
        
        Args:
          temp: A float representing the current temperature.

        Returns:
          A 0-dim tensor representing the mean development under one sub-daily
          time step at the specified temperature.
        """

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
    def __init__(self, 
                 config, 
                 n_bins=100,
                 min_x=0,
                 max_x=1,
                 save=False, 
                 file_path='',
                 save_rate=5, 
                 write_rate=10, 
                 precision=4, 
                 sigma=1.1, 
                 mortality=0.1):
        """Initializes a FemaleFifthSixthInstar life stage.
        
        Args:
          config, n_bins, min_x, max_x, save, file_path, save_rate, 
            write_rate, precision:  See _LifeStage.__init__ for details.
          sigma:  A float representing the initial shape to be used when
            generating developmental variability in kernels.
          mortality:  The default mortality rate to be applied each time step.
        """

        super().__init__(config, save, file_path, save_rate, 
                         write_rate, precision)
        self.init_kernel_helpers(n_bins, min_x, max_x)

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
        """Calculates mean development under in the specified temperature.
        
        Args:
          temp: A float representing the current temperature.

        Returns:
          A 0-dim tensor representing the mean development under one sub-daily
          time step at the specified temperature.
        """

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
    def __init__(self, 
                 config, 
                 n_bins=100,
                 min_x=0,
                 max_x=1,
                 save=False, 
                 file_path='',
                 save_rate=5, 
                 write_rate=10, 
                 precision=4, 
                 sigma=1.1, 
                 mortality=0.1):
        """Initializes a MaleFifthInstar life stage.
        
        Args:
          config, n_bins, min_x, max_x, save, file_path, save_rate, 
            write_rate, precision:  See _LifeStage.__init__ for details.
          sigma:  A float representing the initial shape to be used when
            generating developmental variability in kernels.
          mortality:  The default mortality rate to be applied each time step.
        """

        super().__init__(config, save, file_path, save_rate, 
                         write_rate, precision)
        self.init_kernel_helpers(n_bins, min_x, max_x)

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
        """Calculates mean development under in the specified temperature.
        
        Args:
          temp: A float representing the current temperature.

        Returns:
          A 0-dim tensor representing the mean development under one sub-daily
          time step at the specified temperature.
        """

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
    def __init__(self, 
                 config, 
                 n_bins=100,
                 min_x=0,
                 max_x=1,
                 save=False, 
                 file_path='',
                 save_rate=5, 
                 write_rate=10, 
                 precision=4, 
                 sigma=1.1, 
                 mortality=0.1):
        """Initializes a FemalePupae life stage.
        
        Args:
          config, n_bins, min_x, max_x, save, file_path, save_rate, 
            write_rate, precision:  See _LifeStage.__init__ for details.
          sigma:  A float representing the initial shape to be used when
            generating developmental variability in kernels.
          mortality:  The default mortality rate to be applied each time step.
        """

        super().__init__(config, save, file_path, save_rate, 
                         write_rate, precision)
        self.init_kernel_helpers(n_bins, min_x, max_x)

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
        """Calculates mean development under in the specified temperature.
        
        Args:
          temp: A float representing the current temperature.

        Returns:
          A 0-dim tensor representing the mean development under one sub-daily
          time step at the specified temperature.
        """

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
    def __init__(self, 
                 config, 
                 n_bins=100,
                 min_x=0,
                 max_x=1,
                 save=False,
                 file_path='', 
                 save_rate=5, 
                 write_rate=10, 
                 precision=4, 
                 sigma=1.1, 
                 mortality=0.1):
        """Initializes a MalePupae life stage.
        
        Args:
          config, n_bins, min_x, max_x, save, file_path, save_rate, 
            write_rate, precision:  See _LifeStage.__init__ for details.
          sigma:  A float representing the initial shape to be used when
            generating developmental variability in kernels.
          mortality:  The default mortality rate to be applied each time step.
        """

        super().__init__(config, save, file_path, save_rate, 
                         write_rate, precision)
        self.init_kernel_helpers(n_bins, min_x, max_x)

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
        """Calculates mean development under in the specified temperature.
        
        Args:
          temp: A float representing the current temperature.

        Returns:
          A 0-dim tensor representing the mean development under one sub-daily
          time step at the specified temperature.
        """

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
    def __init__(self, 
                 config, 
                 n_bins=100,
                 min_x=0,
                 max_x=1,
                 save=False, 
                 file_path='',
                 save_rate=5, 
                 write_rate=10, 
                 precision=4, 
                 sigma=1.1, 
                 mortality=0.1):
        """Initializes a Adult life stage.
        
        Args:
          config, n_bins, min_x, max_x, save, file_path, save_rate, 
            write_rate, precision:  See _LifeStage.__init__ for details.
          sigma:  A float representing the initial shape to be used when
            generating developmental variability in kernels.
          mortality:  The default mortality rate to be applied each time step.
        """

        super().__init__(config, save, file_path, save_rate, 
                         write_rate, precision)
        self.init_kernel_helpers(n_bins, min_x, max_x)

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
        """Calculates mean development under in the specified temperature.
        
        Args:
          temp: A float representing the current temperature.

        Returns:
          A 0-dim tensor representing the mean development under one sub-daily
          time step at the specified temperature.
        """

        mu = (
            self.config.delta_t 
            * (torch.maximum(
                (self.b 
                 + (self.m
                    *(temp-10))),
                torch.tensor(0))))
        return mu