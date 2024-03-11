import pandas as pd
from matplotlib import pyplot

from neurnn.io import get_df_for_challenge

datafolder = 'data/crcns_challenge'
challengename = 'crcns-ch-epfl-2009-challengeA'
time_points_file = "challengeA_input_times.csv"
total_time_in_seconds = 60
time_points = pd.read_csv('%s/%s' % (datafolder, time_points_file), header=0, sep=",", skipinitialspace=True)

df = get_df_for_challenge(datafolder, challengename, total_time_in_seconds)

pd_idx = pd.IndexSlice
buffer = 0.01
for _, row in time_points.iterrows():
    start_time = row["start"] - buffer
    end_time = row["end"] + buffer
    end_time = min(end_time, total_time_in_seconds)

    df_temp = df.loc[pd_idx[start_time:end_time], :]
    fig, axes = pyplot.subplots(2, 1, sharex=True)
    axes[0].set_title(row["type"])
    axes[0].plot(df_temp["t"], df_temp["I"])
    axes[1].plot(df_temp["t"], df_temp[0])
    pyplot.show()
