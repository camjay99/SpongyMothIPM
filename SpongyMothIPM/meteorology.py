import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

def two_point_sine(x1, y1, x2, y2, start_time, end_time):
    factor = np.pi/((end_time-start_time)%24)
    alpha = ((y2-y1) 
             / (np.cos(factor*((x2-start_time)%24)) 
                - np.cos(factor*((x1-start_time)%24))))
    beta = y1 - alpha*np.cos(factor*((x1-start_time)%24))
    return alpha, beta

def load_daymet_data(file_path):
    df = pd.read_csv(file_path, header=6)
    return df

def daymet_to_diurnal(df, low_time, high_time, sample_period, sample_start_time, num_days=None):
    """Take daymet point record and computes diurnal cycle at the specified
       sampling frequency. Assumptions of when the low and high temperatures
       occurred are required. Algorithm will repeat first/last day as 
       necessary to ensure full coverage of provided period (midnight-midnight)."""
    if 24 % sample_period != 0:
        raise Exception('Sampling rate must be a divisor of 24')
    daily_obs = 24 // sample_period
    min_temps = df['tmin (deg c)'].to_numpy()
    max_temps = df['tmax (deg c)'].to_numpy()
    years = df['year'].to_numpy()
    ydays = df['yday'].to_numpy()

    if num_days is not None:
        min_temps = min_temps[:num_days]
        max_temps = max_temps[:num_days]
        years = years[:num_days]
        ydays = ydays[:num_days]
   
    # Based on which time occurs first, alter
    # computations to ensure we have estimates
    # from the very start to the very end.
    # P1 models are period fully located within
    # study period according to assumptions.
    # P2 models
    if low_time < high_time:
        start = low_time
        end = high_time
        min_temps = np.insert(min_temps, -1, 
                              min_temps[-1])
        max_temps = np.insert(max_temps, 0, 
                              max_temps[0])
        alphas_p1, betas_p1 = (
            two_point_sine(low_time, 
                           min_temps[:-1], 
                           high_time, 
                           max_temps[1:],
                           low_time,
                           high_time))
        alphas_p2, betas_p2 = (
            two_point_sine(high_time, 
                           max_temps, 
                           low_time, 
                           min_temps,
                           high_time,
                           low_time))
    else:
        start = high_time
        end = low_time
        min_temps = np.insert(min_temps, 0, 
                              min_temps[0])
        max_temps = np.insert(max_temps, -1, 
                              max_temps[-1])
        alphas_p1, betas_p1 = (
            two_point_sine(high_time, 
                           max_temps[:-1], 
                           low_time, 
                           min_temps[1:],
                           high_time,
                           low_time))
        alphas_p2, betas_p2 = (
            two_point_sine(low_time, 
                           min_temps, 
                           high_time, 
                           max_temps,
                           low_time,
                           high_time))

    # Calculate temps at each daily time point and interleave
    temps = np.zeros((len(min_temps)-1)*daily_obs)
    for i, time in enumerate(range(sample_start_time, 25, sample_period)):
        if (time >= start) and (time < end):
            # Increasing temperature
            temps[i::daily_obs] = (
                    alphas_p1
                    * np.cos(np.pi/((end-start)%24) * ((time-start)%24))
                    + betas_p1
            )
        elif time < start:
            # Decreasing temperature
            # Undefined at start of study period, repeat first cycle
            temps[i] = (
                    alphas_p2[0]
                    * np.cos(np.pi/((start-end)%24) * ((time-end)%24))
                    + betas_p2[0]
            )
            temps[i+daily_obs::daily_obs] = (
                    alphas_p2[1:-1]
                    * np.cos(np.pi/((start-end)%24) * ((time-end)%24))
                    + betas_p2[1:-1]
            )
        else:
            # Decreasing temperature
            # Undefined at end of study period, repeat last cycle
            temps[-(24-i)] = (
                    alphas_p2[-1]
                    * np.cos(np.pi/((start-end)%24) * ((time-end)%24))
                    + betas_p2[-1]
            )
            temps[i:-daily_obs:daily_obs] = (
                    alphas_p2[1:-1]
                    * np.cos(np.pi/((start-end)%24) * ((time-end)%24))
                    + betas_p2[1:-1]
            )
    # Reattach year/day/hour information.
    time_stamps = [(year, yday, hour) 
                    for year, yday in zip(years, ydays)
                    for hour in range(sample_start_time, 25, sample_period)]
    timed_temps = pd.DataFrame.from_records(data=time_stamps, 
                                            columns=['year', 'yday', 'hour'])
    timed_temps['temp'] = temps
    return timed_temps

if __name__ == '__main__':
    df = load_daymet_data('./data/11752_lat_41.7074_lon_-77.0846_2025-06-11_123955.csv')
    low_time = 20
    high_time = 10
    sample_period = 1
    sample_start_time = 1
    temps = daymet_to_diurnal(df, 
                              low_time, 
                              high_time, 
                              sample_period, 
                              sample_start_time)
    print(temps[40:120])
    fig, ax = plt.subplots()
    xs = [sample_start_time + (sample_period)*x for x in range(len(temps))]
    ax.plot(xs, temps['temp'], color='black')
    ax.scatter(list(range(low_time, 24*len(df)+low_time, 24)), df['tmin (deg c)'], color='blue')
    ax.scatter(list(range(high_time, 24*len(df)+high_time, 24)), df['tmax (deg c)'], color='red')
    ax.set_xlim(2000, 2220)
    plt.show()
