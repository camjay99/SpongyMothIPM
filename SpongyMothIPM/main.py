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
    df = met.load_daymet_data('./data/11932_lat_42.4687_lon_-76.3787_2025-06-26_132939.csv')
    low_time = 1
    high_time = 13
    sample_period = 4
    sample_start_time = 1
    temps = met.daymet_to_diurnal(df, 
                                low_time, 
                                high_time, 
                                sample_period, 
                                sample_start_time,
                                365)


    config = Config(dtype=torch.float,
                    n_bins=100,
                    min_x=0,
                    max_x=1.5,
                    delta_t=sample_period/24)

    days = len(temps)//(24//sample_period)
    years = 1

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
        prediapause = kernels.Prediapause(config, save=True, file_path='outputs/test.csv', save_rate=1, mortality=0)
        diapause = kernels.Diapause(config, save=False, mortality=0)
        postdiapause = kernels.Postdiapause(config, save=False, mortality=0)
        first_instar = kernels.FirstInstar(config, save=False, mortality=0)
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
        prediapause.init_pop(total, mu, sigma)
        diapause.init_pop(empty, mu, sigma)
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
        for year in range(years):
            for day in range(days):
                start = day*(24//sample_period)
                end = (day+1)*(24//sample_period)
                day_temps = temps[start:end]
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

    fig, axes = plt.subplots(ncols=2, sharey=False)

    total_abundance = np.zeros(len(prediapause.abundances))
    for stage in [prediapause, diapause, postdiapause, first_instar, second_instar, third_instar, fourth_instar, 
                  male_late_instar, female_late_instar, male_pupae, female_pupae, adults]:
        total_abundance += np.array(stage.abundances)
    for year in range(years):
        axes[0].plot(prediapause.abundances[year*364:(year+1)*365], label='Prediapause', color='blue', linestyle='-')
        axes[0].plot(diapause.abundances[year*364:(year+1)*365], label='Diapause', color='blue', linestyle='--')
        axes[0].plot(postdiapause.abundances[year*364:(year+1)*365], label='Postdiapause', color='blue', linestyle=':')
        axes[0].plot(first_instar.abundances[year*364:(year+1)*365], label='L1', color='orange', linestyle='-')
        axes[0].plot(second_instar.abundances[year*364:(year+1)*365], label='L2', color='orange', linestyle='--')
        axes[0].plot(third_instar.abundances[year*364:(year+1)*365], label='L3', color='orange', linestyle=':')
        axes[0].plot(fourth_instar.abundances[year*364:(year+1)*365], label='L4', color='orange', linestyle='-.')
        axes[0].plot(male_late_instar.abundances[year*364:(year+1)*365], label='L5 male', color='red', linestyle='-')
        axes[0].plot(female_late_instar.abundances[year*364:(year+1)*365], label='L5/L6 female', color='yellow', linestyle='-')
        axes[0].plot(male_pupae.abundances[year*364:(year+1)*365], label='Pupae male', color='red', linestyle='--')
        axes[0].plot(female_pupae.abundances[year*364:(year+1)*365], label='Pupae female', color='red', linestyle='--')
        axes[0].plot(adults.abundances[year*364:(year+1)*365], label='Adult', color='brown', linestyle='-')
        axes[0].plot(total_abundance[year*364:(year+1)*365], label='Total', linestyle='dotted', color='black')
        axes[0].legend()
        axes[0].set_title("Sample 2-year Time Series")

    stage = diapause
    reduce = True
    for i in range(0, len(stage.hist_pops), 10):
        if reduce:
            pop = stage.hist_pops[i].reshape((config.n_bins, config.n_bins))
            pop = torch.sum(pop, dim=0)
        else:
            pop = stage.hist_pops[i]
        axes[1].plot(pop)
    axes[1].set_title("Sample Diapause Age Dists.")

    plt.show()