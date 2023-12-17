import sys
import matplotlib.pyplot as plt
import math
import numpy as np
import collections
from wait_timer import WaitTimer
from read_stdin import readline, print_until_first_csi_line

packet_count = 0

# Deque definition
perm_amp = collections.deque(maxlen=200)
perm_phase = collections.deque(maxlen=100)


def process(res):
    # Parser
    all_data = res.split(',')
    csi_data = all_data[25].split(" ")
    csi_data[0] = csi_data[0].replace("[", "")
    csi_data[-1] = csi_data[-1].replace("]", "")

    csi_data.pop()
    csi_data = [int(c) for c in csi_data if c]
    imaginary = []
    real = []
    for i, val in enumerate(csi_data):
        if i % 2 == 0:
            imaginary.append(val)
        else:
            real.append(val)

    csi_size = len(csi_data)
    amplitudes = []
    phases = []
    if len(imaginary) > 0 and len(real) > 0:
        for j in range(int(csi_size / 2)):
            amplitude_calc = math.sqrt(imaginary[j] ** 2 + real[j] ** 2)
            phase_calc = math.atan2(imaginary[j], real[j])
            amplitudes.append(amplitude_calc)
            phases.append(phase_calc)

        perm_phase.append(phases)
        perm_amp.append(amplitudes)

while True:
    line = readline()
    if "CSI_DATA" in line:
        process(line)
        packet_count += 1
        if(packet_count >= 200):
            print("Chunk", perm_amp[-1])
            packet_count = 0
        #total_packet_counts += 1

        #if render_plot_wait_timer.check() and len(perm_amp) > 2:
           # render_plot_wait_timer.update()
           # carrier_plot(perm_amp)#


