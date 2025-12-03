import os
import sys
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


parentpath = os.path.dirname(os.getcwd())
sys.path.append(parentpath)

import pyrpl
from pyrpl.async_utils import sleep
from datetime import datetime

class DriftMeasurement:
    """Encapsulate drift measurement configuration and actions.

    Attributes:
        parent_path: base path to the repo (auto-detected by default)
        config_filename: name of the YAML config in ./configs
        fpga_filename: relative path from parent_path to FPGA binary
        base_folder: where to save measurement output
        T: acquisition duration (s)
        delay: delay between unlocking and measurement (s)
        max_tries: maximum attempts to acquire lock
    """

    def __init__(self,
                 config_filename='config_LD_V1.yml',
                 fpga_filename=os.path.join('pyrpl', 'fpga', 'red_pitaya.bin'),
                 gui=False,
                 base_folder='drift_data',
                 T=1.0,
                 delay=0.1,
                 max_tries=80,
                 threshold=0.02,
                 subfolder = None):
        self.folderpath = os.getcwd()
        self.parentfolder_path = os.path.dirname(self.folderpath)
        self.config_filename = config_filename
        self.config_source = os.path.join(self.folderpath, 'configs', self.config_filename)

        self.folder = None
        self.subfolder = subfolder

        self.fpga_source = os.path.join(self.parentfolder_path, *fpga_filename.split(os.sep))
        

        self.T = T
        self.delay = delay
        self.base_folder = base_folder
        self.max_tries = max_tries
        self.threshold = threshold

        # Initialize hardware interface
        self.p = pyrpl.Pyrpl(config=self.config_source, filename=self.fpga_source, gui=gui)

        self.s = self.p.rp.scope

    def make_experiment_folder(self, foldername=""):
        if not os.path.exists(self.base_folder):
            os.makedirs(self.base_folder)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if foldername == "":
            foldername = f"run_{timestamp}"
        self.folder = os.path.join(self.base_folder, foldername)
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)


    def save_drift_data(self, res, t, T, delay):
        """Save arrays, the current figure and parameters to a timestamped folder."""
        self.save_folder = self.folder
        if self.subfolder is not None:
            self.save_folder = os.path.join(self.folder, self.subfolder)
            os.makedirs(self.save_folder) 
        print("Save folder:", self.save_folder)

        np.save(os.path.join(self.save_folder, "res.npy"), res)
        np.save(os.path.join(self.save_folder, "t.npy"), t)
        plt.savefig(os.path.join(self.save_folder, "plot.png"), dpi=300)

        with open(os.path.join(self.save_folder, "params.txt"), "w") as f:
            f.write(f"delay = {delay}\n")
            f.write(f"T = {T}\n")

        print(f"Saved drift data to: {self.save_folder}")

    def take_curve_and_plot(self, fut, T=None, delay=None, absolute=False, label=""):
        res = fut.result()
        ch1 = res[0]
        ch2 = res[1]
        t = np.array(self.s.times)
        if absolute:
            ch1 = np.abs(ch1)
        plt.plot(t, ch1, label=label + ' iq0')
        plt.plot(t, ch2, label=label + ' pid0')
        plt.xlabel('Time [s]')
        plt.ylabel(self.s.input1 + ' [V]')
        plt.title('UMZI Lock Drift Measurement')
        plt.legend()
        
        self.save_drift_data(res, t, T, delay) 

    def unlock_and_measure_drift(self, T=None, delay=None):
        T = T if T is not None else self.T
        delay = delay if delay is not None else self.delay
        self.s.duration = T
        self.s.input2 = 'pid0'
        try:
            self.s.trig_source = 'immediately'
        except Exception:
            self.s.trigger_source = 'immediately'

        fut = self.s.single_async()
        v_start = self.s.voltage_in1
        print(f"Starting voltage: {v_start:.4f} V")
        sleep(delay)

        self.p.rp.pid0.paused = True

        print("Curve ready:", self.s.curve_ready())
        T_left = 2 * T - delay
        sleep(T_left)
        print("Curve ready:", self.s.curve_ready())

        if self.s.curve_ready():
            self.take_curve_and_plot(fut, T=T, delay=delay, absolute=False)
        else:
            exit("Abort: curve was not ready after waiting period")

        self.s.input2 = 'iq1'

    def check_locked(self):
        count = 0
        while count < 5:
            if abs(self.s.voltage_in1) < self.threshold:
                count += 1
                sleep(0.1)
            else:
                return False
        return True

    def run(self):
        tries = 0
        self.p.rp.pid0.paused = False
        for tries in range(self.max_tries):
            print(f"Waiting for lock times {tries}")
            time.sleep(0.1)
            
            if self.check_locked(): 
                print("Lock acquired with s.voltage_in1 =", self.s.voltage_in1)
                self.unlock_and_measure_drift()
                print("We plotted")
                return True

        if self.p.rp.pid0.ival > 3.9:
            self.p.rp.pid0.ival = 0
            print("ival reset to 0")

        print("Could not acquire lock")
        return False

    def run_multiple(self, N):
        self.make_experiment_folder()
        for exp_num in range(N):
            self.subfolder = str(exp_num)
            self.run()

if __name__ == '__main__':
    dm = DriftMeasurement(T=1.0, delay=0.1)

    dm.run_multiple(3)