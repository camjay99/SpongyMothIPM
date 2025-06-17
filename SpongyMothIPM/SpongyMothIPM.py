import math

import torch
import matplotlib.pyplot as plt

from SpongthMothIPM.config import Config
import SpongyMothIPM.util as util
import SpongyMothIPM.kernels as kernels

config = Config(dtype=torch.float,
                n_bins=200,
                min_x=0,
                max_x=0,
                delta_t=1)

num_days = 100
temp = 15

##############
# Model Driver
##############

prediapause = kernels.Prediapause(config).init_pop(0.2, 1.1)
diapause = kernels.Diapause(config).init_pop(0.2, 1.1)
postdiapause = kernels.Postdiapause(config).init_pop(0.2, 1.1)
first_instar = kernels.FirstInstar(config).init_pop(0.2, 1.1)
second_instar = kernels.SecondInstar(config).init_pop(0.2, 1.1)
third_instar = kernels.ThirdInstar(config).init_pop(0.2, 1.1)
fourth_instar = kernels.FourthInstar(config).init_pop(0.2, 1.1)
male_late_instar = kernels.MaleFifthInstar(config).init_pop(0.2, 1.1)
female_late_instar = kernels.FemaleFifthSixthInstar(config).init_pop(0.2, 1.1)
male_pupae = kernels.MalePupae(config).init_pop(0.2, 1.1)
female_pupae = kernels.FemalePupae(config).init_pop(0.2, 1.1)
adults = kernels.Adult(config).init_pop(0.2, 1.1)

for day in range(num_days):
    transfers = prediapause.run_one_step(temp)
    transfers = postdiapause.run_one_step(temp, transfers)
    transfers = diapause.run_one_step(temp, transfers)
    transfers = first_instar.run_one_step(temp, transfers)
    transfers = second_instar.run_one_step(temp, transfers)
    transfers = third_instar.run_one_step(temp, transfers)
    transfers_dif = fourth_instar.run_one_step(temp, transfers)
    transfers = male_late_instar.run_one_step(temp, transfers_dif/2)
    to_adult = male_pupae.run_one_step(temp, transfers)
    transfers = female_late_instar.run_one_step(temp, transfers_dif/2)
    to_adult += female_pupae.run_one_step(temp, transfers)
    transfers = adults.run_one_step(temp, to_adult)
    prediapause.add_transfers(transfers)