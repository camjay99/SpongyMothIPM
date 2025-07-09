if __name__ == '__main__':
    import math

    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    import SpongyMothIPM.meteorology as met
    from SpongyMothIPM.config import Config
    import SpongyMothIPM.util as util
    import SpongyMothIPM.kernels as kernels

    ###################
    # Load Weather Data
    ###################
    df = met.load_daymet_data('./data/mont_st_hilaire/mont_st_hilaire_1980_1991.csv')
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

    days = len(temps)//(24//sample_period)

    ##############
    # Model Driver
    ##############

    with torch.no_grad():
        # Build life stages
        prediapause = kernels.Prediapause(config, save=False, mortality=0)
        diapause = kernels.Diapause(config, n_bins_I=45, n_bins_D=45, save=False, mortality=0)
        postdiapause = kernels.Postdiapause(config, save=False, mortality=0)
        first_instar = kernels.FirstInstar(config, save=False, file_path='outputs/mont_st_hilaire/first_instar.csv', save_rate=1, mortality=0)
        second_instar = kernels.SecondInstar(config, save=False, mortality=0)
        third_instar = kernels.ThirdInstar(config, save=False, mortality=0)
        fourth_instar = kernels.FourthInstar(config, save=False, mortality=0)
        male_late_instar = kernels.MaleFifthInstar(config, save=False, mortality=0)
        female_late_instar = kernels.FemaleFifthSixthInstar(config, save=False, mortality=0)
        male_pupae = kernels.MalePupae(config, save=False, mortality=0)
        female_pupae = kernels.FemalePupae(config, save=False, mortality=0)
        adults = kernels.Adult(config, save=False, mortality=0)

        # Initiate populations
        mu = 0.2
        sigma = 1.1
        total = 10
        empty = 0
        prediapause.init_pop(empty, mu, sigma)
        diapause.init_pop(total, mu, sigma)
        postdiapause.init_pop(empty, mu, sigma)
        first_instar.init_pop(empty, mu, sigma)
        second_instar.init_pop(empty, mu, sigma)
        third_instar.init_pop(empty, mu, sigma)
        fourth_instar.init_pop(empty, mu, sigma)
        male_late_instar.init_pop(empty, mu, sigma)
        female_late_instar.init_pop(empty, mu, sigma)
        male_pupae.init_pop(empty, mu, sigma)
        female_pupae.init_pop(empty, mu, sigma)
        adults.init_pop(empty, mu, sigma)

        # Run Model
        start_year = temps['year'].min()
        end_year = temps['year'].max()
        start = 0
        for year in range(start_year, end_year+1):
            print(f"Starting year {year}")
            days = temps.loc[temps['year'] == year, 'yday'].max()
            for day in range(1, days+1):
                end = start + (24//sample_period)
                day_temps = temps.iloc[start:end]
                transfers = prediapause.run_one_step(day_temps)
                transfers = diapause.run_one_step(day_temps, transfers)
                transfers = postdiapause.run_one_step(day_temps, transfers)
                transfers = first_instar.run_one_step(day_temps, transfers)
                transfers = second_instar.run_one_step(day_temps, transfers)
                transfers = third_instar.run_one_step(day_temps, transfers)
                transfers_dif = fourth_instar.run_one_step(day_temps, transfers)
                transfers = male_late_instar.run_one_step(day_temps, transfers_dif/2)
                to_adult = male_pupae.run_one_step(day_temps, transfers)
                transfers = female_late_instar.run_one_step(day_temps, transfers_dif/2)
                to_adult += female_pupae.run_one_step(day_temps, transfers)
                transfers = adults.run_one_step(day_temps, to_adult)
                prediapause.add_transfers(transfers/2)

                if (year >= 1988) and (day >= 100) and not first_instar.save:
                    print("Started saving population distributions.")
                    first_instar.save = True
                if (day >= 200) and (first_instar.save):
                    print("Stopped saving population distributions.")
                    first_instar.save = False

                start = end