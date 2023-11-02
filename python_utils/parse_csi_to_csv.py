import re
import pandas as pd
from math import sqrt, atan2

if __name__ == "__main__":
    """
    This script file demonstrates how to transform raw CSI out from the ESP32 into CSI-amplitude and CSI-phase.
    """

    FILE_NAME = "D:\\Wifi_Sensing\\esp32-wifi-sensing\\datasets\\tch-prep\\tch-csi-10 - Copy.csv"

    f = open(FILE_NAME)
    col_amp = []
    col_pha = []
    df = pd.DataFrame()
    #df = df.drop(0)
    for j, l in enumerate(f.readlines()):
        imaginary = []
        real = []
        amplitudes = []
        phases = []

        # Parse string to create integer list
        csi_string = re.findall(r"\[(.*)\]", l)[0]
        csi_raw = [int(x) for x in csi_string.split(" ") if x != '']

        # Create list of imaginary and real numbers from CSI
        for i in range(len(csi_raw)):
            if i % 2 == 0:
                imaginary.append(csi_raw[i])
            else:
                real.append(csi_raw[i])

        # Transform imaginary and real into amplitude and phase
        for i in range(int(len(csi_raw) / 2)):
            amplitudes.append(sqrt(imaginary[i] ** 2 + real[i] ** 2))
            phases.append(atan2(imaginary[i], real[i]))
        col_amp.append(amplitudes)
        col_pha.append(phases)

    df['amplitude'] = col_amp
    df['phase'] = col_pha
    df.to_csv("D:\\Wifi_Sensing\\esp32-wifi-sensing\\datasets\\tch-prep\\tch-prep-amp.csv")
