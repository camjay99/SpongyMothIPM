if __name__ == '__main__':
    import math

    import matplotlib.pyplot as plt
    import torch

    import SpongyMothIPM.meteorology as met
    from SpongyMothIPM.config import Config
    import SpongyMothIPM.util as util
    import SpongyMothIPM.kernels as kernels

    ###################
    # Load Weather Data
    ###################
    df = met.load_daymet_data('./data/11752_lat_41.7074_lon_-77.0846_2025-06-11_123955.csv')
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
                    n_bins=100,
                    min_x=0,
                    max_x=1.5,
                    delta_t=sample_period/24)

    days = 10

    ##############
    # Model Driver
    ##############

    def quick_test(diapause, temps):
        kernel = diapause.build_kernel(temps).detach()
        col_sums = kernel.sum(dim=0)
        print((col_sums - torch.ones(col_sums.shape)).sum())
        torch.testing.assert_close(col_sums, torch.ones(col_sums.shape)) 

    with torch.no_grad():
        # Build life stages
        prediapause = kernels.Prediapause(config, save=True, mortality=0)
        diapause = kernels.Diapause(config, save=True, mortality=0)
        postdiapause = kernels.Postdiapause(config, save=True, mortality=0)
        first_instar = kernels.FirstInstar(config, save=True, mortality=0)
        second_instar = kernels.SecondInstar(config, save=True, mortality=0)
        third_instar = kernels.ThirdInstar(config, save=True, mortality=0)
        fourth_instar = kernels.FourthInstar(config, save=True, mortality=0)
        male_late_instar = kernels.MaleFifthInstar(config, save=True, mortality=0)
        female_late_instar = kernels.FemaleFifthSixthInstar(config, save=True, mortality=0)
        male_pupae = kernels.MalePupae(config, save=True, mortality=0)
        female_pupae = kernels.FemalePupae(config, save=True, mortality=0)
        adults = kernels.Adult(config, save=True, mortality=0)

        # Initiate populations
        mu = 0.2
        sigma = 1.1
        prediapause.init_pop(mu, sigma)
        diapause.init_pop(mu, sigma, mu, sigma)
        postdiapause.init_pop(mu, sigma)
        first_instar.init_pop(mu, sigma)
        second_instar.init_pop(mu, sigma)
        third_instar.init_pop(mu, sigma)
        fourth_instar.init_pop(mu, sigma)
        male_late_instar.init_pop(mu, sigma)
        female_late_instar.init_pop(mu, sigma)
        male_pupae.init_pop(mu, sigma)
        female_pupae.init_pop(mu, sigma)
        adults.init_pop(mu, sigma)

        # Run Model
        for day in range(days):
            start = day*(24//sample_period)
            end = (day+1)*(24//sample_period)
            day_temps = temps[start:end]
            transfers = prediapause.run_one_step(day_temps)
            transfers = diapause.run_one_step(day_temps, transfers)
            print(day_temps)
            quick_test(diapause, day_temps)
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
            prediapause.add_transfers(transfers)

    fig, ax = plt.subplots()

    ax.plot(prediapause.abundances, label='Prediapause')
    ax.plot(diapause.abundances, label='Diapause')
    ax.plot(postdiapause.abundances, label='Postdiapause')
    ax.plot(first_instar.abundances, label='L1')
    ax.plot(second_instar.abundances, label='L2')
    ax.plot(third_instar.abundances, label='L3')
    ax.plot(fourth_instar.abundances, label='L4')
    ax.plot(male_late_instar.abundances, label='L5 male')
    ax.plot(female_late_instar.abundances, label='L5/L6 female')
    ax.plot(male_pupae.abundances, label='Pupae male')
    ax.plot(female_pupae.abundances, label='Pupae female')
    ax.plot(adults.abundances, label='Adult')
    ax.legend()
    plt.show()