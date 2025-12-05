import yaml
import time
import numpy as np
import xarray as xr
from tqdm import tqdm
from lmfit import Model
from pyrpl import Pyrpl
from yamlcore import CoreLoader
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
#from pyrpl.UMZI_models import SineModel, CosineModel
from pyrpl.UMZI_models2 import SinusoidalModel

####################
#    MAIN CLASS    #
####################
class RPLockboxMZI2(Pyrpl):
    '''
        Wrapper class for a PyRPL configuration that is designed for locking an Unbalanced Mach-Zehnder Interferometer (UMZI).
        This class inherits from the Pyrpl class and provides additional functionality for setting up and calibrating the system.
    '''
    #config_file is for starting up and saving the data, yml file is for setting up the modules
    def __init__(self, hostname, yml_file, gui=False, config_file='default_config.yml'): #Om RPLockbox te initen init je de parent class pyrpl en zet je modules op
        super().__init__(hostname=hostname, gui=gui, config=config_file)
        self._setup_all_modules(yml_file)
        self.rp.pid1.free() #zorg dat je PID feedback kunt applyen via PID1
        self.cal_amp = None
        self.cal_freq = None

    def __str__(self):
        return f"RPLockboxMZI instance connected to {self.rp.hostname}"

    def _setup_all_modules(self, yml_file): # laadt zet je de params van alle modules
        self._get_module_settings(yml_file)
        self._apply_settings()

    def _apply_settings(self): #of the modules
        if not hasattr(self, 'settings'):
            raise ValueError("Module settings not loaded. Please load settings first.")
        
        for module in self.settings.keys():
            getattr(self.rp, module).setup(**self.settings[module])

    def _get_module_settings(self, yml_file): #from the config file
        with open(yml_file, 'r') as stream:
            try:
                self.settings = yaml.load(stream, Loader=CoreLoader)
            except yaml.YAMLError as exc:
                raise ValueError(f"Error loading YAML file: {exc}")
            
    def calibrate(self, do_plot=False):
        '''
        Calibration function for the UMZI locking setup that measures the phase-to-voltage relation for both the detector and IQ channels.
        '''
        self.take_calibration_data()
        self.fit_calibration_data()
        if do_plot:
            self.plot_calibrations()

    def setup_locking(self, phase_setpoint):
        '''
        Setup function for the locking scheme that calculates the required magnitudes of both IQ outputs
        that are required to lock the interferometer at a given phase setpoint.
        '''
        self.phase_setpoint = phase_setpoint

        #There is a 0->1 and 1-> 2 'issue' because the iq works with 0 and 1 but the scope works wiht ch1 and ch2
        new_quadrature_factor_iq0 = - np.cos(phase_setpoint) * self.iq_calib_gain_ch1 / (self.iq_model_ch1.fit_result.params[self.iq_model_ch1.prefix+'amp'].value)
        new_quadrature_factor_iq1 = np.sin(phase_setpoint) * self.iq_calib_gain_ch2 / (self.iq_model_ch2.fit_result.params[self.iq_model_ch2.prefix+'amp'].value)
        #I DONT THINK  THIS IS CORRECT YET! IT IS JUST SET BY THE ALREADY EXISTING
        #QUADRATURES IN take_calibration data!!!!!!


        # First two cases are required to avoid rounding errors for locking near 0, pi/2, pi, ...
        # where either only the cosine or sine should be used as the error signal.
        if abs(new_quadrature_factor_iq0) < 1e-5:
            self.rp.iq0.setup(quadrature_factor = 0)
            self.rp.iq1.setup(quadrature_factor=new_quadrature_factor_iq1)
        elif abs(new_quadrature_factor_iq1) < 1e-5:
            self.rp.iq0.setup(quadrature_factor=new_quadrature_factor_iq0)
            self.rp.iq1.setup(quadrature_factor = 0)
        else: #set the quadrature factors we calculated above
            self.rp.iq0.setup(quadrature_factor=new_quadrature_factor_iq0)
            self.rp.iq1.setup(quadrature_factor=new_quadrature_factor_iq1)

        #differential_mode_enabled ==> you set the inputof PID0  to prvsly input PID0 - PID1
        self.rp.pid0.setup(setpoint=0, input="iq0", differential_mode_enabled=True) 
        #WHY DO WE TURN ON THE DIFFERENTIAL MODE??????????????????????????????????????????????????????????????????????????????????????????????
        #WHY DO WE TURN ON THE PID FEEBACK??????????????
        
        self.rp.pid1.setup(input="iq1", output_direct='off') #of course then PID1 should not be doing anything!
        self.rp.iq0.setup(output_direct="out2")

    def start_locking(self, phase_setpoint=None): 
        '''
        Function to start the locking process. This function turns on the PID loop, turns off all ASG outputs and configures
        the scope to give a graphical clue for fine-adjusting the locking parameters.

        COULD YOU ALSO JUST LOAD A CONFIG FILE HERE? I think he loads the data that he takes from the yml file in the init function

        '''
        if phase_setpoint is not None: #check if you have set a phase setpoint
            self.setup_locking(phase_setpoint)
        else: ##############################Why do you still need the assert statement here? 
            assert hasattr(self, 'phase_setpoint'), "Phase setpoint not defined. Please call setup_locking() first."

        self.rp.asg0.setup(output_direct='off')
        self.rp.asg1.setup(output_direct='off')
        self.rp.pid0.setup(output_direct='out2') #Here I changed to out2 from out1!!!!!!!
        self.rp.iq0.setup(output_direct='out2') #Here I changed to out2 from out1!!!!!!!
        self.rp.iq1.setup(output_direct='off')
        self.rp.scope.setup(
            input1="in1",
            input2="out2",  #Here I changed from in2 to out2 since I am using a differential PD
                            #So I can also look at the output (in1 is now 'unconnected')
            ch_math_active=True,
            math_formula=f"{self.det_model_ch1.phase_relation(self.phase_setpoint)}*ch1/ch1",
            trigger_source="ext_positive_edge",
            duration = 10 / self.rp.iq0.frequency, #10 modulation oscillations happen in this time. 
            ##############################Is there a good reason for this number?##############################
            run_continuous=True, 
            rolling_mode=False,
            trigger_delay=0.0
        )
        self.rp.iq0.synchronize_iqs()
        self.rp.pid0.reg_integral = 0 #Set integral value of feedback to 0

    def take_calibration_data(self):
        '''
        Execution of all the required steps to take the calibration data, both for the detector voltages
        as well as the IQ channels.

        It turns off the PID feedback, sets the scope, and synchronizes iqs

        AGAIN, CONFIG FILE??? 

        '''
        ##########################
        #      IQ channels       #
        ##########################
        self.cal_amp = 0.8
        self.cal_freq = 10
        self.rp.asg0.setup(output_direct='out2', waveform='ramp', amplitude=self.cal_amp, frequency=self.cal_freq) #create a phi_env ~ t to see cos(At)
        self.rp.scope.setup(
            input1='iq0',
            input2='iq1',
            ch_math_active=False, 
            trigger_source='asg0', 
            duration=0.4/(self.rp.asg0.frequency) / 2,   #Here you see two oscillations (no, I think four) of the sin and cos
            trigger_delay = 0.5/(self.rp.asg0.frequency) / 2,
            rolling_mode=False
        )

        self.rp.iq0.setup(output_direct='out2', quadrature_factor=self.settings['iq0']['quadrature_factor'])
        self.rp.iq1.setup(quadrature_factor=self.settings['iq1']['quadrature_factor'])
        self.rp.pid0.setup(output_direct='off')
        self.rp.iq2.setup(output_direct='off', input="off") #I changed this from in2 to off because we have the differential photodiode
        self.rp.iq0.synchronize_iqs() #again, I don't think this will do anything. 
        self.rp.scope.single() #Take the sin and the cos from the scope
        iq_calib_data_ch1, iq_calib_data_ch2 = self.rp.scope.save_curve() # iq_calib_data_ch1 = sin, iq_calib_data_ch1 = cos

        ##########################
        #   Detector channels    #
        ##########################
        """
        self.rp.scope.setup(
            input1='in1',
            #input2='in2',
            ch_math_active=False, 
            #math_formula='ch1-ch2', 
            trigger_source='asg0', 
            duration=2/(self.rp.asg0.frequency),   #Here you see two oscillations (no, I think four) of the sin and cos
            trigger_delay = 1/(2*self.rp.asg0.frequency), #####################################WHY THIS TRIGGER DELAY????
        )

        self.rp.iq0.setup(output_direct='off') #We dont want any modulation messing up the signals
        self.rp.scope.single() 

        # Data saving is done after taking both singles to minimize the time between the two measurements,
        # thereby minimizing the drift of the phase.
        iq_calib_data_time, iq_calib_data_ch1 = iq_calib_data_ch1.data #iq_calib_data_ch1 = sin, iq_calib_data_time = t
        _, iq_calib_data_ch2 = iq_calib_data_ch2.data #iq_calib_data_ch1 = cos

        #det stands for detector
        det_calib_data_ch1, _ = self.rp.scope.save_curve() 
        det_calib_data_time, det_calib_data_ch1 = det_calib_data_ch1.data #det_calib_data_ch1 = the output of the DIFFERENTIAL photodiode

        # Condition for masking the part of the ramp that is far enough from the kinks
        # to get proper cosine/sine fits.
        # This is really nice, but wait, why do we then even save data over the course of multiple ramp kinks? 
        time_filter_cond_iq = np.logical_and(
            iq_calib_data_time > 0.1 / self.rp.asg0.frequency, 
            iq_calib_data_time < 0.5 / self.rp.asg0.frequency #Why do we get so close to the peak? Why not 0.4
        )
        time_filter_cond_det = np.logical_and(
            det_calib_data_time > 0.1 / self.rp.asg0.frequency,
            det_calib_data_time < 0.5 / self.rp.asg0.frequency #Why do we get so close to the peak? Why not 0.4
        )
        self.iq_calib_time = iq_calib_data_time[time_filter_cond_iq] #Here we take the time array and only include the comps far from the kinks
        self.iq_calib_ch1, self.iq_calib_ch2 = iq_calib_data_ch1[time_filter_cond_iq], iq_calib_data_ch2[time_filter_cond_iq] #same for the sin & cos
        
        #And here we do the same as above, but then for the two signals of the PDs. 
        self.det_calib_time = det_calib_data_time[time_filter_cond_det] 
        self.det_calib_ch1 = det_calib_data_ch1[time_filter_cond_det]
        """
        self.rp.scope.input1 = 'in1'
        
        self.rp.iq0.setup(output_direct='off') #We dont want any modulation messing up the signals
        self.rp.scope.single() 

        # Data saving is done after taking both singles to minimize the time between the two measurements,
        # thereby minimizing the drift of the phase.
        iq_calib_data_time, iq_calib_data_ch1 = iq_calib_data_ch1.data #iq_calib_data_ch1 = sin, iq_calib_data_time = t
        _, iq_calib_data_ch2 = iq_calib_data_ch2.data #iq_calib_data_ch1 = cos

        #det stands for detector
        det_calib_data_ch1, _ = self.rp.scope.save_curve() 
        det_calib_data_time, det_calib_data_ch1 = det_calib_data_ch1.data #det_calib_data_ch1 = the output of the DIFFERENTIAL photodiode

        self.iq_calib_time = iq_calib_data_time
        self.iq_calib_ch1, self.iq_calib_ch2 = iq_calib_data_ch1, iq_calib_data_ch2
        
        #And here we do the same as above, but then for the two signals of the PDs. 
        self.det_calib_time = det_calib_data_time
        self.det_calib_ch1 = det_calib_data_ch1


        self.iq_calib_gain_ch1, self.iq_calib_gain_ch2 = self.rp.iq0.quadrature_factor, self.rp.iq1.quadrature_factor
        #OK, BUT WHERE DO  WE APPLY A CORRECTION? 


    def fit_calibration_data(self):
        '''
        Fitting function for the calibration data. It creates CosineModel and SineModel instances,
        which are lmfit.Model subclasses. These custom models are used to fit the calibration data
        and extract the required parameters for locking the interferometer.
        '''
        self.det_model_ch1 = SinusoidalModel("det", 1, guesses = [0,0.2,0.15,np.pi]) #WHAT DO THESE ARGUMENTS DO? 
        self.iq_model_ch1 = SinusoidalModel("iq", 1, guesses = [0,0.5,0.15,np.pi])
        self.iq_model_ch2 = SinusoidalModel("iq", 2, guesses = [0,0.6,0.15,np.pi])
        #offset, amplitude, T, phase
        
        det_params_ch1 = self.det_model_ch1.guess(self.det_calib_ch1, self.det_calib_time)
        self.det_model_ch1.perform_fit(self.det_calib_ch1, det_params_ch1, time=self.det_calib_time)

        # The time-to-phase relation is locked to the same as the detector model,
        # such that we can measure the relative phase offset between IQ and det.
        iq_params_ch1 = self.iq_model_ch1.guess(self.iq_calib_ch1, self.iq_calib_time)
        iq_params_ch2 = self.iq_model_ch2.guess(self.iq_calib_ch2, self.iq_calib_time)
        self.iq_model_ch1.lock_phase_guess(iq_params_ch1, self.det_model_ch1)
        self.iq_model_ch2.lock_phase_guess(iq_params_ch2, self.det_model_ch1) 
        #You take over the  Vpi from another model
        #And set as a guess the phase according to this model
        #For the rest, you take the params of the iq models
        self.iq_model_ch1.perform_fit(self.iq_calib_ch1, iq_params_ch1, time=self.iq_calib_time)
        self.iq_model_ch2.perform_fit(self.iq_calib_ch2, iq_params_ch2, time=self.iq_calib_time)

    def calculateVPis(self): #should return a constant array!
        VPis = []
        for model in [self.det_model_ch1, self.iq_model_ch1, self.iq_model_ch2]:
            T = model.T_fit
            Vpi = self.cal_amp * self.cal_freq * T / 2 #in Volts because amp is in Volts
            VPis.append(Vpi)
        return VPis

    def plot_calibrations(self):
        __, iq_ax_1 = self.iq_model_ch1.create_calibration_plot()
        __, iq_ax_2 = self.iq_model_ch2.create_calibration_plot()
        __, det_ax_1 = self.det_model_ch1.create_calibration_plot()
        #__, det_ax_2 = self.det_model_ch2.create_calibration_plot()

        fig, ax = combine_axes_to_subplots([iq_ax_1, iq_ax_2, det_ax_1], ncols=2, nrows=2, figsize=(12,7))
        fig.tight_layout()

        return fig, ax        

    def unlock(self): #############################Why do you not turn off the IQ modulation?????????????????????????????????????????????????
        ''' Stop the PID loop and turn off all outputs.'''
        self.rp.pid0.setup(output_direct='off')
        self.rp.asg1.setup(output_direct='off')
        self.rp.pid0.reg_integral = 0
        self.rp.scope.setup(input1='in1', input2='in2', ch_math_active=False, rolling_mode=True, duration=1.07, trigger_source='immediately')
        self.rp.scope.run_continuous = True

    def test_intermittent_locking(self, phase_setpoint, num_shots, photon_timing, total_unlock_duration):
        '''
        Function that characterizes the quality of the intermittent locking scheme. It outputs a figure that summarizes
        the statistical results and an xarray.Dataset that contains all the relevant data for further analysis.
        '''
        self.phase_setpoint = phase_setpoint
        self.unlock_duration = total_unlock_duration

        self.start_locking(phase_setpoint)
        time.sleep(0.1)

        time_data, ch1_data, ch2_data = self.measure_lock_accuracy(num_shots, photon_timing, self.unlock_duration)
        photon_time_idx = np.argmin(np.abs(time_data - photon_timing))
        ch1_photon_data = ch1_data[:, photon_time_idx]
        ch2_photon_data = ch2_data[:, photon_time_idx]
        phases_channel1 = self.det_model_ch1.get_phase_from_voltage(ch1_photon_data, function_increasing=phase_setpoint < np.pi)
        phases_channel2 = self.det_model_ch2.get_phase_from_voltage(ch2_photon_data, function_increasing=phase_setpoint > np.pi)
        fidelities_ch1 = 1/2 * (1 + np.cos(np.abs(phases_channel1 - phase_setpoint)))
        fidelities_ch2 = 1/2 * (1 + np.cos(np.abs(phases_channel2 - phase_setpoint)))

        fig, ax = plt.subplots(3, 2, figsize=(8, 8), sharey=True)

        ax[0,0].hist(ch1_photon_data, bins=50, label='ch1', color="red")
        ax[0,0].axvline(self.det_model_ch1.phase_relation(phase_setpoint), color='red', linestyle='--', label='expected')
        ax[0,0].set_title(
            r"$V_1 = %.2f \pm %.2f$ V, expected %.2f" % (np.mean(ch1_photon_data), np.std(ch1_photon_data), self.det_model_ch1.phase_relation(phase_setpoint))
        )
        ax[0,0].set_xlabel("Detector voltage [V]")
        ax[0,0].set_ylabel("Counts")
        ax[0,0].set_xlim(
            min(np.mean(ch1_photon_data), self.det_model_ch1.phase_relation(phase_setpoint)) - 0.05, 
            max(np.mean(ch1_photon_data), self.det_model_ch1.phase_relation(phase_setpoint)) + 0.05
        )

        ax[0,1].hist(ch2_photon_data, bins=50, label='ch2', color="b")
        ax[0,1].axvline(self.det_model_ch2.phase_relation(phase_setpoint), color='blue', linestyle='--', label='expected')
        ax[0,1].set_xlabel("Detector voltage [V]")
        ax[0,1].set_title(
            r"$V_2 = %.2f \pm %.2f$ V, expected %.2f" % (np.mean(ch2_photon_data), np.std(ch2_photon_data), self.det_model_ch2.phase_relation(phase_setpoint))
        )
        ax[0,1].set_xlim(
            min(np.mean(ch2_photon_data), self.det_model_ch2.phase_relation(phase_setpoint)) - 0.05, 
            max(np.mean(ch2_photon_data), self.det_model_ch2.phase_relation(phase_setpoint)) + 0.05
        )
        ax[1,0].hist(phases_channel1/np.pi, bins=50, label='ch1', color="red")
        ax[1,0].axvline(phase_setpoint/np.pi, color='red', linestyle='--', label='expected')
        ax[1,0].set_title(
            r"$\phi_1 = (%.2f \pm %.2f) \; \pi$ rad, expected $%.2f \pi$" % (np.mean(phases_channel1)/np.pi, np.std(phases_channel1)/np.pi, phase_setpoint/np.pi)
        )
        ax[1,0].set_xlim(phase_setpoint/np.pi - 0.25, phase_setpoint/np.pi + 0.25)
        ax[1,0].set_xlabel(r"$\phi$ [$\pi$ rad]")
        ax[1,0].set_ylabel("Counts")

        ax[1,1].hist(phases_channel2/np.pi, bins=50, label='ch2', color="b")
        ax[1,1].axvline(phase_setpoint/np.pi, color='blue', linestyle='--', label='expected')
        ax[1,1].set_xlabel(r"$\phi$ [$\pi$ rad]")
        ax[1,1].set_title(
            r"$\phi_2 = (%.2f \pm %.2f) \; \pi$ rad, expected $%.2f \pi$" % (np.mean(phases_channel2)/np.pi, np.std(phases_channel2)/np.pi, phase_setpoint/np.pi)
        )
        ax[1,1].set_xlim(phase_setpoint/np.pi - 0.25, phase_setpoint/np.pi + 0.25)

        ax[2,0].hist(fidelities_ch1, bins=50, label='ch1', color="red")
        ax[2,0].axvline(1, c="grey", linestyle='--', label='Ideal')
        ax[2,0].set_title(
            r"$F_1 = (%.2f \pm %.2f)$, expected $1$" % (np.mean(fidelities_ch1), np.std(fidelities_ch1))
        )
        ax[2,0].set_xlabel("Fidelity")
        ax[2,0].set_ylabel("Counts")

        ax[2,1].hist(fidelities_ch2, bins=50, label='ch2', color="b")
        ax[2,1].axvline(1, c="grey", linestyle='--', label='Ideal')
        ax[2,1].set_title(
            r"$F_2 = (%.2f \pm %.2f)$, expected $1$" % (np.mean(fidelities_ch2), np.std(fidelities_ch2))
        )
        ax[2,1].set_xlabel("Fidelity")

        fig.suptitle((
            r"Data for $\phi=%.2f \pi$" % (self.phase_setpoint/np.pi) + f" at {photon_timing*1e6:.0f}us w.r.t. freeze trigger, \n" +
            f"Settings: setpoint {self.rp.pid0.setpoint:.3f}, P={self.rp.pid0.p:.3f}, I={self.rp.pid0.i:.3f}"
        ))
        fig.tight_layout()

        ds = xr.Dataset()
        ds['repetition'] = np.arange(num_shots)
        ds['ch1_detector_voltage'] = xr.DataArray(ch1_photon_data, dims=['repetition'])
        ds['ch2_detector_voltage'] = xr.DataArray(ch2_photon_data, dims=['repetition'])
        ds['ch1_phase'] = xr.DataArray(phases_channel1, dims=['repetition'])
        ds['ch2_phase'] = xr.DataArray(phases_channel2, dims=['repetition'])
        ds['ch1_fidelity'] = xr.DataArray(fidelities_ch1, dims=['repetition'])
        ds['ch2_fidelity'] = xr.DataArray(fidelities_ch2, dims=['repetition'])
    
        return ds

    def measure_lock_accuracy(self, num_shots, photon_time_after_trigger, unlock_duration):
        '''
        Function that measures the locking accuracy by taking multiple shots of the detector voltages
        and outputting the data of all repetitions.
        '''
        self.unlock_duration = unlock_duration
        self.rp.pid0.reg_integral = 0
        self.rp.scope.setup(
            input1='in1',
            input2='in2',
            ch_math_active=False,
            trigger_source="ext_positive_edge",
            duration=unlock_duration*3,
            trigger_delay=unlock_duration/5
        )
        self.rp.scope.run_continuous = False
        self.rp.scope.single()
        curve1, curve2 = self.rp.scope.save_curve()
        time_data = curve1.data[0]
        unlocked_mask = np.logical_and(
            time_data > photon_time_after_trigger - self.unlock_duration/5,
            time_data < photon_time_after_trigger + self.unlock_duration/5
        )

        ch1_data = np.zeros((int(num_shots), np.sum(unlocked_mask)))
        ch2_data = np.zeros((int(num_shots), np.sum(unlocked_mask)))

        for idx in tqdm(range(num_shots)):
            self.rp.scope.single()
            curve1, curve2 = self.rp.scope.save_curve()
            time.sleep(0.01)
            time_data, ch1 = curve1.data
            _, ch2 = curve2.data
            ch1_data[idx, :] = ch1[unlocked_mask]
            ch2_data[idx, :] = ch2[unlocked_mask]

        return time_data[unlocked_mask], ch1_data, ch2_data


def combine_axes_to_subplots(axes, nrows=1, ncols=None, figsize=(10, 5)):
    """Combine a list of axes into a single figure with subplots."""
    if ncols is None:
        ncols = len(axes)
    new_fig, new_axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    
    # Ensure new_axes is always a list
    if nrows * ncols == 1:
        new_axes = [new_axes]
    else:
        new_axes = new_axes.flatten()
    
    for old_ax, new_ax in zip(axes, new_axes):
        # Copy lines
        for line in old_ax.get_lines():
            new_line = mlines.Line2D(
                xdata=line.get_xdata(),
                ydata=line.get_ydata(),
                linestyle=line.get_linestyle(),
                linewidth=line.get_linewidth(),
                color=line.get_color(),
                marker=line.get_marker(),
                markersize=line.get_markersize(),
                label=line.get_label(),
            )
            new_ax.add_line(new_line)
        
        # Copy other properties
        new_ax.set_xlim(old_ax.get_xlim())
        new_ax.set_ylim(old_ax.get_ylim())
        new_ax.set_title(old_ax.get_title())
        new_ax.set_xlabel(old_ax.get_xlabel())
        new_ax.set_ylabel(old_ax.get_ylabel())
        new_ax.legend(handles=new_ax.get_lines()[:-1])
        
    # Close original figures
    for ax in axes:
        plt.close(ax.figure)
    
    return new_fig, new_axes

if __name__ == "__main__":
    obj = RPLockboxMZI2(hostname="172.16.20.41", gui=True, config_file='travis_global_config', yml_file='lockbox_config.yml')