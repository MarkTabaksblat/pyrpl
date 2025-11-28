import os
import sys

import numpy as np
import matplotlib.pyplot as plt 


parentpath = os.path.dirname(os.getcwd())
sys.path.append(parentpath)

import pyrpl
from pyrpl.async_utils import sleep

FPGAsource = os.path.join(os.path.abspath(parentpath), 'pyrpl','fpga', 'red_pitaya.bin')
configfilename = 'config_LD_V1.yml'
CONFIGsource = os.path.join(os.path.abspath(os.getcwd()), 'configs', configfilename)

p = pyrpl.Pyrpl(config=CONFIGsource, filename=FPGAsource, gui = False)

s = p.rp.scope

T = 0.5  # seconds
s.duration = T

# acquire N full traces and store channel 1 (zeroed at t=0)
N = 3
curves = []
times = []

for i in range(N):
    print('acq', i)
    """ 
    fut = s.single()
    # use pyrpl.async_utils.sleep so the event loop can run
    sleep(T + 0.2)
    res = fut.result()  # res is array([ch1, ch2])
    """ 
    res = s.single(timeout=None)  # blocking call
    ch1 = res[0] - res[0][0]  # remove DC offset
    curves.append(ch1)
    t = np.array(s.times)
    times.append(t + t[-1])  # shift time axis for each acquisition

choice = 1

if choice == 0: 
    # plot all acquired traces vs time
    for i, ch in enumerate(curves):
        plt.plot(times[i], np.abs(ch)) 
    plt.xlabel('Time [s]')
    plt.ylabel('Voltage')
    plt.ylim(-1, 1) 
    plt.show()

elif choice == 1:
    mean_error = np.mean(curves, axis=0)
    plt.plot(times[0], np.abs(mean_error))
    plt.xlabel('Time [s]')
    plt.ylabel('Voltage')
    plt.ylim(-1, 1) 
    plt.show()