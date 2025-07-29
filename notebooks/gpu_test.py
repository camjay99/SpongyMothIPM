import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import SpongyMothIPM.meteorology as met
from SpongyMothIPM.config import Config
import SpongyMothIPM.util as util
import SpongyMothIPM.kernels as kernels
import SpongyMothIPM.visualization as viz

# Model setup
class SimpleModel():
    def __init__(self):
        # Build life stages
        self.prediapause = kernels.Prediapause(
            config, save=False, save_rate=1, mortality=0, sigma=1.0516)
        self.diapause = kernels.Diapause(
            config, n_bins_I=50, n_bins_D=50, save=False, save_rate=1, mortality=0, sigma_I=1.5501, sigma_D=1.5500)
        self.postdiapause = kernels.Postdiapause(
            config, save=False, save_rate=1, mortality=0, sigma=1.0543)
        self.first_instar = kernels.FirstInstar(
            config, save=False, save_rate=1, mortality=0, 
            file_path='../outputs/mont_st_hilaire/first_instar.csv', sigma=1.1458)
        self.second_instar = kernels.SecondInstar(
            config, save=False, save_rate=1, mortality=0, sigma=1.1483)
        self.third_instar = kernels.ThirdInstar(
            config, save=False, save_rate=1, mortality=0, sigma=1.0491)
        self.fourth_instar = kernels.FourthInstar(
            config, save=False, save_rate=1, mortality=0, sigma=1.0468)
        self.male_late_instar = kernels.MaleFifthInstar(
            config, save=False, save_rate=1, mortality=0, sigma=1.0449)
        self.female_late_instar = kernels.FemaleFifthSixthInstar(
            config, save=False, save_rate=1, mortality=0, sigma=1.1476)
        self.male_pupae = kernels.MalePupae(
            config, save=False, save_rate=1, mortality=0, sigma=1.0474)
        self.female_pupae = kernels.FemalePupae(
            config, save=False, save_rate=1, mortality=0, sigma=1.1211)
        self.adults = kernels.Adult(
            config, save=False, save_rate=1, mortality=0, sigma=1.1469)
        
        # Gather parameters together for optimization.
        self.parameters = [self.prediapause.sigma,
                           self.diapause.sigma_D,
                           self.diapause.sigma_I,
                           self.postdiapause.sigma,
                           self.first_instar.sigma,
                           self.second_instar.sigma,
                           self.third_instar.sigma,
                           self.fourth_instar.sigma,
                           self.male_late_instar.sigma,
                           self.female_late_instar.sigma,
                           self.male_pupae.sigma,
                           self.female_pupae.sigma,
                           self.adults.sigma]
        
    def init_pop(self):
        # Initiate populations
        mu = 0.2
        sigma = 1.1
        total = 1
        empty = 0
        self.prediapause.init_pop(empty, mu, sigma)
        # Individuals all begin in the prediapause stage at the same time.
        self.prediapause.add_transfers(total) 
        self.diapause.init_pop(empty, mu, sigma)
        self.postdiapause.init_pop(empty, mu, sigma)
        self.first_instar.init_pop(empty, mu, sigma)
        self.second_instar.init_pop(empty, mu, sigma)
        self.third_instar.init_pop(empty, mu, sigma)
        self.fourth_instar.init_pop(empty, mu, sigma)
        self.male_late_instar.init_pop(empty, mu, sigma)
        self.female_late_instar.init_pop(empty, mu, sigma)
        self.male_pupae.init_pop(empty, mu, sigma)
        self.female_pupae.init_pop(empty, mu, sigma)
        self.adults.init_pop(empty, mu, sigma)
    
    def run_one_time_step(self, 
                          year, 
                          doy, 
                          doy_temps,
                          record_start_year, 
                          record_start_doy,
                          record_end_year,
                          record_end_doy,
                          save):
        transfers = self.prediapause.run_one_step(doy_temps)
        transfers = self.diapause.run_one_step(doy_temps, transfers)
        transfers = self.postdiapause.run_one_step(doy_temps, transfers)

        # Record transfers to first instars
        if record_start_year <= year <= record_end_year: 
            if record_start_doy <= doy <= record_end_doy:
                if save and not self.first_instar.save:
                    self.first_instar.save = True
                self.hatched[year].append(transfers)
            else:
                self.first_instar.save = False

        transfers = self.first_instar.run_one_step(doy_temps, transfers)
        transfers = self.second_instar.run_one_step(doy_temps, transfers)
        transfers = self.third_instar.run_one_step(doy_temps, transfers)
        transfers_dif = self.fourth_instar.run_one_step(doy_temps, transfers)
        transfers = self.male_late_instar.run_one_step(doy_temps, transfers_dif/2)
        to_adult = self.male_pupae.run_one_step(doy_temps, transfers)
        transfers = self.female_late_instar.run_one_step(doy_temps, transfers_dif/2)
        to_adult += self.female_pupae.run_one_step(doy_temps, transfers)
        transfers = self.adults.run_one_step(doy_temps, to_adult)
        self.prediapause.add_transfers(transfers/2)

    def forward(self, 
                start_doy,
                record_start_year, 
                record_start_doy,
                record_end_year,
                record_end_doy,
                save):
        # Run Model
        start_year = temps['year'].min()
        end_year = temps['year'].max()
        start = (24//sample_period)*(start_doy-1)
        # For tracking emerging eggs
        self.hatched = {year:[] 
                        for year in range(record_start_year, 
                                          record_end_year+1)}
        for year in range(start_year, end_year+1):
            #print(f"Starting year {year}")
            days = temps.loc[temps['year'] == year, 'yday'].max()
            if year > start_year:
                start_doy = 1
            for doy in range(start_doy, days+1):
                end = start + (24//sample_period)
                doy_temps = temps.iloc[start:end]
                if not (record_start_year <= year <= record_end_year):
                    # While we are not in the year of interest, we will not bother
                    # tracking gradients as it is very memory intensive.
                    with torch.no_grad():
                        self.run_one_time_step(year, 
                                              doy, 
                                              doy_temps,
                                              record_start_year, 
                                              record_start_doy,
                                              record_end_year,
                                              record_end_doy,
                                              save)
                else:
                    self.run_one_time_step(year, 
                                          doy, 
                                          doy_temps,
                                          record_start_year, 
                                          record_start_doy,
                                          record_end_year,
                                          record_end_doy,
                                          save)
                start = end

    def print_params(self):
        print('Prediapause: ', self.prediapause.sigma, self.prediapause.sigma.grad)
        print('Diapause I: ', self.diapause.sigma_I, self.diapause.sigma_I.grad)
        print('Diapause D: ', self.diapause.sigma_D, self.diapause.sigma_D.grad)
        print('Postdiapause: ', self.postdiapause.sigma, self.postdiapause.sigma.grad)
        print('First Instar: ', self.first_instar.sigma, self.first_instar.sigma.grad)
        print('Second Instar: ', self.second_instar.sigma, self.second_instar.sigma.grad)
        print('Third Instar: ', self.third_instar.sigma, self.third_instar.sigma.grad)
        print('Fourth Instar: ', self.fourth_instar.sigma, self.fourth_instar.sigma.grad)
        print('Male Late Instar: ', self.male_late_instar.sigma, self.male_late_instar.sigma.grad)
        print('Female Late Instar: ', self.female_late_instar.sigma, self.female_late_instar.sigma.grad)
        print('Male Pupae: ', self.male_pupae.sigma, self.male_pupae.sigma.grad)
        print('Female Pupae: ', self.female_pupae.sigma, self.female_pupae.sigma.grad)
        print('Adult: ', self.adults.sigma, self.adults.sigma.grad)

    def compute_gradients(self, validation, year, verbose=False):
        # Create a tensor with the relative abundances at each time point.
        self.cum_hatched = [0]*len(self.hatched[year])
        self.cum_hatched[0] = self.hatched[year][0]
        for i in range(1, len(self.hatched[year])):
            self.cum_hatched[i] = self.cum_hatched[i-1] + self.hatched[year][i]
        self.cum_hatched = torch.stack(self.cum_hatched)

        # Compute loss and gradients
        loss = torch.mean((self.cum_hatched - validation)**2)
        print('Loss: ', loss.item())
        loss.backward()

    def update_params(self, validation, year, verbose=False):
        self.compute_gradients(validation, year)

        # Use gradients to update trainable parameters
        with torch.no_grad():
            if verbose:
                self.print_params()
            # Prediapause
            self.prediapause.sigma -= self.prediapause.sigma.grad * learning_rate
            self.prediapause.sigma.grad.data.zero_()
            # Diapause
            self.diapause.sigma_I -= self.diapause.sigma_I.grad * learning_rate
            self.diapause.sigma_I.grad.data.zero_()
            self.diapause.sigma_D -= self.diapause.sigma_D.grad * learning_rate
            self.diapause.sigma_D.grad.data.zero_()
            # Postdiapause
            self.postdiapause.sigma -= self.postdiapause.sigma.grad * learning_rate
            self.postdiapause.sigma.grad.data.zero_()
            # First Instar
            self.first_instar.sigma -= self.first_instar.sigma.grad * learning_rate
            self.first_instar.sigma.grad.data.zero_()
            # Second Instar
            self.second_instar.sigma -= self.second_instar.sigma.grad * learning_rate
            self.second_instar.sigma.grad.data.zero_()
            # Thrid Instar
            self.third_instar.sigma -= self.third_instar.sigma.grad * learning_rate
            self.third_instar.sigma.grad.data.zero_()
            # Fourth Instar
            self.fourth_instar.sigma -= self.fourth_instar.sigma.grad * learning_rate
            self.fourth_instar.sigma.grad.data.zero_()
            # Male Fifth Instar
            self.male_late_instar.sigma -= self.male_late_instar.sigma.grad * learning_rate
            self.male_late_instar.sigma.grad.data.zero_()
            # Female Fifth/Sixth Instar
            self.female_late_instar.sigma -= self.female_late_instar.sigma.grad * learning_rate
            self.female_late_instar.sigma.grad.data.zero_()
            # Male Pupae
            self.male_pupae.sigma -= self.male_pupae.sigma.grad * learning_rate
            self.male_pupae.sigma.grad.data.zero_()
            # Female Pupae
            self.female_pupae.sigma -= self.female_pupae.sigma.grad * learning_rate
            self.female_pupae.sigma.grad.data.zero_()
            # Adults
            self.adults.sigma -= self.adults.sigma.grad * learning_rate
            self.adults.sigma.grad.data.zero_()


def run_adam(model, lr, num_iters, validation, year, verbose=False):
    optim = torch.optim.Adam(model.parameters, lr=lr)
    
    time1 = time.time()
    for i in range(num_iters):
        model.init_pop()
        model.forward(300, 1988, 0, 1990, 365, False)
        model.compute_gradients(validation, year, verbose)
        if verbose and (i % 10 == 0):
            print("Iteration: ", i)
            model.print_params()
        optim.step()
        optim.zero_grad()
        time2 = time.time()
        print("Time: ", time2-time1)
        time1 = time2

with torch.device('cuda'):
    print("Loading Meteorlogical Data")
    df = met.load_daymet_data('../data/mont_st_hilaire/mont_st_hilaire_1980_1991.csv')
    low_time = 1
    high_time = 13
    sample_period = 4
    sample_start_time = 1
    temps = met.daymet_to_diurnal(df, 
                                  low_time, 
                                  high_time, 
                                  sample_period, 
                                  sample_start_time)

    config = Config(dtype=torch.float,
                    delta_t=sample_period/24)

    days = torch.tensor(len(temps)//(24//sample_period))
    learning_rate = torch.tensor(0.00000001)

    print("Loading Validation Data")
    validation = pd.read_csv('../data/mont_st_hilaire/hilaire_88.csv')
    print(validation)
    validation['doy'] = validation['doy'].round()
    validation = np.interp(np.arange(0, 365), 
                       validation['doy'],
                       validation['hatch'])
    validation = torch.tensor(validation)

    print("Running Adam")
    model = SimpleModel()
    run_adam(model, 1e-3, 100, validation, 1988, verbose=True)
