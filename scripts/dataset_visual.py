import pandas as pd
import matplotlib.pyplot as plt

def get_df_for_challenge(datafolder,challengename , total_time_in_seconds):
    CURRENT_FILE = "current.txt"
    VOLTAGE_FILE = "voltage_allrep.txt"
    READ_CSV_CONFIG = {
        "header": None,
        "delim_whitespace": True
    }
    df_currents = pd.read_csv('%s/%s/%s' % (datafolder,challengename,  CURRENT_FILE), **READ_CSV_CONFIG, names=["I"])
    df_voltages = pd.read_csv('%s/%s/%s' % (datafolder,challengename,  VOLTAGE_FILE), **READ_CSV_CONFIG)

    df = df_currents.join(df_voltages)

    df["t"] = pd.timedelta_range(start='0 seconds', end='%d seconds' % total_time_in_seconds, periods=len(
        df))
    df["t"] = df["t"].dt.total_seconds()
    df = df.set_index("t", drop=False)
    return df

# insert here the path to the dataset, or link with path variables
#
datafolder = 'crcns-ch-epfl-2009-challengeA'
challengename = ''
time_points_file = "challengeA_input_times.csv"
total_time_in_seconds = 60
time_points = pd.read_csv('%s' % ( time_points_file), header=0, sep=",", skipinitialspace=True)

df = get_df_for_challenge(datafolder,challengename, total_time_in_seconds)

pd_idx = pd.IndexSlice
buffer = 0.01

for _, row in time_points.iterrows():
    start_time = row["start"] - buffer
    end_time = row["end"] + buffer
    end_time = min(end_time, total_time_in_seconds)

    df_temp = df.loc[pd_idx[start_time:end_time], :]
    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].set_title(row["type"])
    axes[0].set_ylabel('$p_A$ (I)')
    axes[0].plot(df_temp["t"], df_temp["I"])
    axes[1].plot(df_temp["t"], df_temp[0])
    plt.xlabel('time (s)')
    plt.ylabel('$V_m$ (V)')
    plt.show()

