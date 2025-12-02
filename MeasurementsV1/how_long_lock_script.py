import os
import sys

import numpy as np
import matplotlib.pyplot as plt 
import time


parentpath = os.path.dirname(os.getcwd())
sys.path.append(parentpath)

import pyrpl
from pyrpl.async_utils import sleep
from datetime import datetime

FPGAsource = os.path.join(os.path.abspath(parentpath), 'pyrpl','fpga', 'red_pitaya.bin')
configfilename = 'config_LD_V1.yml'
CONFIGsource = os.path.join(os.path.abspath(os.getcwd()), 'configs', configfilename)

p = pyrpl.Pyrpl(config=CONFIGsource, filename=FPGAsource, gui = False)
p.rp.asg0.output_direct = 'off'  # direct output to troublehsoot

s = p.rp.scope

def save_drift_data(res, t, T, delay, foldername = "", base_folder="drift_data"):
    """
    Save drift measurement result (res, t) and plot into a new subfolder.
    Also store delay and T in params.txt if the folder is newly created.
    """

    # 1. Create main directory if missing
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)

    # 2. Create a new timestamped subfolder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if foldername == "": 
        foldername = f"run_{timestamp}"
    folder = os.path.join(base_folder, foldername)
    os.makedirs(folder)

    # 3. Save numerical data
    np.save(os.path.join(folder, "res.npy"), res)
    np.save(os.path.join(folder, "t.npy"), t)

    # 4. Save the plot
    plt.savefig(os.path.join(folder, "plot.png"), dpi=300)

    # 5. Create and save parameters (always for a new run)
    with open(os.path.join(folder, "params.txt"), "w") as f:
        f.write(f"delay = {delay}\n")
        f.write(f"T = {T}\n")

    print(f"Saved drift data to: {folder}")
    return folder

def take_curve_and_plot(fut, T, delay, abs = False, label = ""): 
    T_left = 2 * T - delay
    #res = s.single(timeout=None)  # blocking call
    sleep(T_left)  # wait for acquisition to complete
    print("Curve ready:", s.curve_ready())
    res = fut.result()  # blocking call
    ch1 = res[0] # channel 1 data
    ch2 = res[1] # channel 2 data
    t = np.array(s.times)
    #t = t - t[0]  # zero time axis
    if abs:
        ch1 = np.abs(ch1)
    plt.plot(t, ch1, label = label + 'iq0')
    plt.plot(t, ch2, label = label + 'pid0')
    plt.xlabel('Time [s]')
    plt.ylabel(s.input1 + ' [V]') 
    plt.title('UMZI Lock Drift Measurement')
    plt.legend()
    plt.show()
    save_drift_data(res, t, T, delay)

def unlock_and_measure_drift(T = 1, delay = 0.1):
    s.duration = T  # set duration
    s.input2 = 'pid0' # monitor lock error signal

    s.trig_source = 'immediately'
    fut = s.single_async()  # start acquisition
    v_start = s.voltage_in1
    print(f"Starting voltage: {v_start:.4f} V")

    sleep(delay)  # wait a bit
 
    p.rp.pid0.paused = True  # unlock
    print("Curve ready:", s.curve_ready())

    take_curve_and_plot(fut, T, delay, abs = False)
    s.input2 = 'iq1'  # restore input

def check_locked(threshold=0.04): 
    count = 0 
    while (count < 5): 
        if abs(s.voltage_in1) < threshold: 
            count += 1
            sleep(0.1)
        else:
            return False
    return True

tries = 0
MAXtries = 40
p.rp.pid0.paused = False  # ensure lock is active
while not check_locked() and tries < MAXtries:
    tries += 1
    print(f"Waiting for lock times {tries}")
    time.sleep(0.1)
    if p.rp.pid0.ival > 3.9:
        p.rp.pid0.ival = 0  # reset integrator to help acquire lock
        print("ival reset to 0")

if not True: #check_locked():
    exit("Could not acquire lock")

else:
    print("Lock acquired with s.voltage_in1 =", s.voltage_in1)
    unlock_and_measure_drift()
    exit("We plotted")
