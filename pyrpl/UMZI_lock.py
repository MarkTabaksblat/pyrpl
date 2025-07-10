import yaml
import time
import numpy as np
from tqdm import tqdm
from lmfit import Model
from pyrpl import Pyrpl
from yamlcore import CoreLoader
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pyrpl.UMZI_models import SineModel, CosineModel

####################
#    MAIN CLASS    #
####################
class RPLockboxMZI(Pyrpl):
    '''
        Wrapper class for a PyRPL configuration that is designed for locking an Unbalanced Mach-Zehnder Interferometer (UMZI).
        This class inherits from the Pyrpl class and provides additional functionality for setting up and calibrating the system.
    '''
    def __init__(self, hostname, yml_file, gui=False, config_file='default_config.yml'):
        super().__init__(hostname=hostname, gui=gui, config=config_file)
        self._setup_all_modules(yml_file)
        self.rp.pid1.free()

    def __str__(self):
        return f"RPLockboxMZI instance connected to {self.rp.hostname}"

    def _setup_all_modules(self, yml_file):
        self._get_module_settings(yml_file)
        self._apply_settings()

    def _apply_settings(self):
        if not hasattr(self, 'settings'):
            raise ValueError("Module settings not loaded. Please load settings first.")
        
        for module in self.settings.keys():
            getattr(self.rp, module).setup(**self.settings[module])

    def _get_module_settings(self, yml_file):
        with open(yml_file, 'r') as stream:
            try:
                self.settings = yaml.load(stream, Loader=CoreLoader)
            except yaml.YAMLError as exc:
                raise ValueError(f"Error loading YAML file: {exc}")
            
    def calibrate(self, do_plot=False):
        self.take_calibration_data()
        self.fit_calibration_data()
        # self.fit_calibration_data(cal_type="det")
    
        if do_plot:
            self.plot_calibration(cal_type="det")
            self.plot_calibration(cal_type="iq")

    def setup_locking(self, phase_setpoint):
        self.phase_setpoint = phase_setpoint

        new_quadrature_factor_iq0 = np.cos(phase_setpoint) * self.iq_calib_gain_ch1 / (self.iq_model_ch1.fit_result.params[self.iq_model_ch1.prefix+'amp'].value)
        new_quadrature_factor_iq1 = np.sin(phase_setpoint) * self.iq_calib_gain_ch2 / (self.iq_model_ch2.fit_result.params[self.iq_model_ch2.prefix+'amp'].value)

        # First two cases are required to avoid rounding errors for locking near 0, pi/2, pi, ...
        # where either only the cosine or sine should be used as the error signal.
        if abs(new_quadrature_factor_iq0) < 1e-5:
            self.rp.iq0.setup(quadrature_factor = 0)
            self.rp.iq1.setup(quadrature_factor=new_quadrature_factor_iq1)
        elif abs(new_quadrature_factor_iq1) < 1e-5:
            self.rp.iq0.setup(quadrature_factor=new_quadrature_factor_iq0)
            self.rp.iq1.setup(quadrature_factor = 0)
        else:
            self.rp.iq0.setup(quadrature_factor=new_quadrature_factor_iq0)
            self.rp.iq1.setup(quadrature_factor=new_quadrature_factor_iq1)

        self.rp.pid0.setup(setpoint=0, input="iq1", differential_mode_enabled=True, pause_gains="pid", min_voltage=-1, max_voltage=1, p=0.1, i=1.0)
        self.rp.pid1.setup(input="iq0", output_direct='off')

    def take_calibration_data(self):
        # self.rp.asg0.setup(output_direct='out1', waveform='cos', amplitude=0.2, frequency=10) #0.55amp
        self.rp.asg0.setup(output_direct='out1', waveform='ramp', amplitude=0.08, frequency=10) #0.55amp
        self.rp.scope.setup(
            input1='iq0',
            input2='iq1',
            ch_math_active=True, 
            math_formula='ch1-ch2', 
            trigger_source='asg0', 
            duration=1/(self.rp.asg0.frequency), 
            trigger_delay = 1/(2*self.rp.asg0.frequency)
        )
        # self.rp.iq0.setup( **self.settings['iq0'])
        self.rp.iq0.setup(output_direct='out1', quadrature_factor=self.settings['iq0']['quadrature_factor'])
        self.rp.iq1.setup(quadrature_factor=self.settings['iq1']['quadrature_factor'])
        self.rp.pid0.setup(output_direct='off')
        self.rp.iq2.setup(output_direct='off', input="in2")
        self.rp.iq0.synchronize_iqs()

        self.rp.scope.single()
        iq_calib_data_ch1, iq_calib_data_ch2 = self.rp.scope.save_curve()
        self.rp.scope.setup(
            input1='in1',
            input2='in2',
            ch_math_active=True, 
            math_formula='ch1-ch2', 
            trigger_source='asg0', 
            duration=2/(self.rp.asg0.frequency),
            trigger_delay = 1/(2*self.rp.asg0.frequency)
        )
        self.rp.iq0.setup(output_direct='off')

        self.rp.scope.single()
        iq_calib_data_time, iq_calib_data_ch1 = iq_calib_data_ch1.data
        _, iq_calib_data_ch2 = iq_calib_data_ch2.data
        det_calib_data_ch1, det_calib_data_ch2 = self.rp.scope.save_curve()
        det_calib_data_time, det_calib_data_ch1 = det_calib_data_ch1.data
        _, det_calib_data_ch2 = det_calib_data_ch2.data

        # Condition for masking the part of the ramp that is far enough from the kinks
        # to get proper cosine/sine fits.
        time_filter_cond_iq = np.logical_and(
            iq_calib_data_time > 0.1 / self.rp.asg0.frequency,
            # iq_calib_data_time < 0.75 / self.rp.asg0.frequency
            iq_calib_data_time < 0.5 / self.rp.asg0.frequency
        )
        time_filter_cond_det = np.logical_and(
            det_calib_data_time > 0.1 / self.rp.asg0.frequency,
            # det_calib_data_time < 0.75 / self.rp.asg0.frequency
            det_calib_data_time < 0.5 / self.rp.asg0.frequency
        )
        self.iq_calib_time = iq_calib_data_time[time_filter_cond_iq]
        self.iq_calib_ch1, self.iq_calib_ch2 = iq_calib_data_ch1[time_filter_cond_iq], iq_calib_data_ch2[time_filter_cond_iq]
        self.det_calib_time = det_calib_data_time[time_filter_cond_det]
        self.det_calib_ch1, self.det_calib_ch2 = det_calib_data_ch1[time_filter_cond_det], det_calib_data_ch2[time_filter_cond_det]
        self.iq_calib_gain_ch1, self.iq_calib_gain_ch2 = self.rp.iq0.quadrature_factor, self.rp.iq1.quadrature_factor


    def fit_calibration_data(self):
        self.det_model_ch1 = CosineModel("det", 1)
        self.det_model_ch2 = CosineModel("det", 2)
        self.iq_model_ch1 = SineModel("iq", 1)
        self.iq_model_ch2 = CosineModel("iq", 2)
        
        det_params_ch1 = self.det_model_ch1.guess(self.det_calib_ch1, self.det_calib_time)
        det_params_ch2 = self.det_model_ch2.guess(self.det_calib_ch2, self.det_calib_time)
        self.det_model_ch1.perform_fit(self.det_calib_ch1, det_params_ch1, time=self.det_calib_time)
        self.det_model_ch2.perform_fit(self.det_calib_ch2, det_params_ch2, time=self.det_calib_time)

        avg_phase_measured = (self.det_model_ch1.fit_result.params['det_1_phase'].value + self.det_model_ch2.fit_result.params['det_2_phase'].value) / 2

        iq_params_ch1 = self.iq_model_ch1.guess(self.iq_calib_ch1, self.iq_calib_time)
        iq_params_ch2 = self.iq_model_ch2.guess(self.iq_calib_ch2, self.iq_calib_time)
        # iq_params_ch1['iq_1_phase'].set(value=avg_phase_measured, min=avg_phase_measured - np.pi/2, max=avg_phase_measured + np.pi/2, vary=True)
        # iq_params_ch2['iq_2_phase'].set(value=avg_phase_measured, min=avg_phase_measured - np.pi/2, max=avg_phase_measured + np.pi/2, vary=True)
        self.iq_model_ch1.perform_fit(self.iq_calib_ch1, iq_params_ch1, time=self.iq_calib_time)
        self.iq_model_ch2.perform_fit(self.iq_calib_ch2, iq_params_ch2, time=self.iq_calib_time)

    def setup_intermittent_locking(self, unlock_duration=50e-6):
        self.unlock_duration = unlock_duration
        print(f"Set picoscope output frequency to {1/(2*unlock_duration)} Hz please")
        self.rp.scope.setup(
            input1='in1', 
            input2='in2', 
            ch_math_active=False, 
            trigger_source='ext_positive_edge', 
            threshold=0.0, 
            duration=2*unlock_duration, 
            run_continuous=True, 
            trigger_delay=0.0
        )

    def unlock(self):
        self.rp.pid0.setup(output_direct='off')
        self.rp.asg1.setup(output_direct='off')
        self.rp.pid0.reg_integral = 0
        self.rp.scope.setup(input1='iq0', input2='iq1', ch_math_active=False, rolling_mode=True, duration=1.07, trigger_source='immediately')
        self.rp.scope.run_continuous = True

    def improve_lock_accuracy(self, num_shots, max_tolerance = 0.1, do_print=True, do_analyze=True):
        num_samples_per_shot = int(self.rp.scope.duration / self.rp.scope.sampling_time)
        ch1_data_init = np.empty((int(num_shots), num_samples_per_shot))
        ch2_data_init = np.empty((int(num_shots), num_samples_per_shot))
        for shot in tqdm(range(int(num_shots))):
            if self.rp.pid0.reg_integral > 0.5:
                self.rp.pid0.reg_integral = 0
                time.sleep(0.1)
            self.rp.scope.single()
            curve1, curve2 = self.rp.scope.save_curve()
            time_data, ch1_data_init[shot,:] = curve1.data
            _, ch2_data_init[shot,:] = curve2.data

        self.unlocked_ch1_data_init = ch1_data_init[:, np.logical_and(time_data>2e-6, time_data<9*self.unlock_duration/10)]
        self.unlocked_ch2_data_init = ch2_data_init[:, np.logical_and(time_data>2e-6, time_data<9*self.unlock_duration/10)]

        ## FOR CHANNEL 1 ##
        expected_ch1_voltage = self.det_model_ch1.phase_relation(self.phase_setpoint)
        ch1_phase_relation_is_increasing = self.det_model_ch1.phase_relation(self.phase_setpoint+0.0001) > expected_ch1_voltage
        actual_phase_ch1_locked_on = self.det_model_ch1.get_phase_from_voltage(np.mean(self.unlocked_ch1_data_init), function_increasing=ch1_phase_relation_is_increasing)
        calibrated_setpoint_ch1 = np.sin(actual_phase_ch1_locked_on - self.phase_setpoint)

        ## FOR CHANNEL 2 ##
        expected_ch2_voltage = self.det_model_ch2.phase_relation(self.phase_setpoint)
        ch2_phase_relation_is_increasing = self.det_model_ch2.phase_relation(self.phase_setpoint+0.0001) > expected_ch2_voltage
        actual_phase_ch2_locked_on = self.det_model_ch2.get_phase_from_voltage(np.mean(self.unlocked_ch2_data_init), function_increasing=ch2_phase_relation_is_increasing)
        calibrated_setpoint_ch2 = np.sin(actual_phase_ch2_locked_on - self.phase_setpoint)

        if do_print:
            print(
                f"Measured voltage ch1: {np.mean(self.unlocked_ch1_data_init):.3f} +/- {np.mean(np.std(self.unlocked_ch1_data_init, axis=1)):.3f} V, while expecting {expected_ch1_voltage:.3f} V --> new setpoint: {calibrated_setpoint_ch1:.3f}.\n" +
                f"Measured voltage ch2: {np.mean(self.unlocked_ch2_data_init):.3f} +/- {np.mean(np.std(self.unlocked_ch2_data_init, axis=1)):.3f} V, while expecting {expected_ch2_voltage:.3f} V --> new setpoint: {calibrated_setpoint_ch2:.3f}.\n"
                f"Taking the average of the two, the new setpoint is: {(calibrated_setpoint_ch1 + calibrated_setpoint_ch2) / 2:.3f}."
            )

        self.rp.pid0.setup(setpoint=(calibrated_setpoint_ch1 + calibrated_setpoint_ch2) / 2)
        time.sleep(0.1)  # Allow some time for the PID to adjust
        ch1_data_final = np.empty((int(num_shots), num_samples_per_shot))
        ch2_data_final = np.empty((int(num_shots), num_samples_per_shot))
        for shot in tqdm(range(int(num_shots))):
            self.rp.scope.single()
            curve1, curve2 = self.rp.scope.save_curve()
            time_data, ch1_data_final[shot,:] = curve1.data
            _, ch2_data_final[shot,:] = curve2.data

        self.unlocked_ch1_data_final = ch1_data_final[:, np.logical_and(time_data>self.unlock_duration/10, time_data<9*self.unlock_duration/10)]
        self.unlocked_ch2_data_final = ch2_data_final[:, np.logical_and(time_data>self.unlock_duration/10, time_data<9*self.unlock_duration/10)]

        self.rp.scope.run_continuous = True

        if do_print:
            print(
                f"Measured voltage ch1: {np.mean(self.unlocked_ch1_data_init):.3f} +/- {np.mean(np.std(self.unlocked_ch1_data_init, axis=1)):.3f} V -> {np.mean(self.unlocked_ch1_data_final):.3f} +/- {np.mean(np.std(self.unlocked_ch1_data_final, axis=1)):.3f} V (expected {expected_ch1_voltage:.3f} V) \n" +
                f"Measured voltage ch1: {np.mean(self.unlocked_ch2_data_init):.3f} +/- {np.mean(np.std(self.unlocked_ch2_data_init, axis=1)):.3f} V -> {np.mean(self.unlocked_ch2_data_final):.3f} +/- {np.mean(np.std(self.unlocked_ch2_data_final, axis=1)):.3f} V (expected {expected_ch2_voltage:.3f} V)"
            )

        if do_analyze:
            self.analyze_lock_improvement()

    def analyze_lock_improvement(self):
        function_increasing_ch1 = self.det_model_ch1.phase_relation(self.phase_setpoint+0.0001) > self.det_model_ch1.phase_relation(self.phase_setpoint)
        self.phases_ch1_init = np.array([
            self.det_model_ch1.get_phase_from_voltage(voltage, function_increasing=function_increasing_ch1) 
            for voltage in self.unlocked_ch1_data_init.mean(axis=1)
        ])
        function_increasing_ch2 = self.det_model_ch2.phase_relation(self.phase_setpoint+0.0001) > self.det_model_ch2.phase_relation(self.phase_setpoint)
        self.phases_ch2_init = np.array([
            self.det_model_ch2.get_phase_from_voltage(voltage, function_increasing=function_increasing_ch2) 
            for voltage in self.unlocked_ch2_data_init.mean(axis=1)
        ])
        self.phases_init = (self.phases_ch1_init + self.phases_ch2_init) / 2
        self.phases_ch1_final = np.array([
            self.det_model_ch1.get_phase_from_voltage(voltage, function_increasing=function_increasing_ch1) 
            for voltage in self.unlocked_ch1_data_final.mean(axis=1)
        ])
        self.phases_ch2_final = np.array([
            self.det_model_ch2.get_phase_from_voltage(voltage, function_increasing=function_increasing_ch2) 
            for voltage in self.unlocked_ch2_data_final.mean(axis=1)
        ])
        self.phases_final = (self.phases_ch1_final + self.phases_ch2_final) / 2

        plt.figure()
        plt.hist(self.phases_init/np.pi, bins=50, label='Init', alpha=0.5)
        plt.hist(self.phases_final/np.pi, bins=50, label='Final', alpha=1)
        plt.axvline(self.phase_setpoint/np.pi, color='k', ls='--', label='Setpoint')
        plt.xlabel("Phase [pi radians]")
        plt.ylabel("Counts")
        plt.legend()
        plt.title(
            f"({np.mean(self.phases_init/np.pi):.3f} +/- {np.std(self.phases_init/np.pi):.3f}) pi -> ({np.mean(self.phases_final/np.pi):.3f} +/- {np.std(self.phases_final/np.pi):.3f}) pi"
        )
        plt.show()

        return self.phases_init, self.phases_final

    def plot_calibration(self, cal_type):
        model_ch1 = getattr(self, f'{cal_type}_model_ch1')
        model_ch2 = getattr(self, f'{cal_type}_model_ch2')

        __, ax_1 = model_ch1.create_calibration_plot()
        __, ax_2 = model_ch2.create_calibration_plot()

        fig, ax = combine_axes_to_subplots([ax_1, ax_2], nrows=1, figsize=(12,3))
        fig.tight_layout()

    def plot_desired_lockpoint(self, phi_setpoint):
        det_time_ch1_at_setpoint, det_time_ch2_at_setpoint = self._phase_to_time(phi_setpoint+2*np.pi, cal_type="det")
        det_time_ch1_at_zero, det_time_ch2_at_zero = self.fit_result_det_calib_ch1.params['t0'].value, self.fit_result_det_calib_ch2.params['t0'].value

        iq_time_ch1_at_setpoint, iq_time_ch2_at_setpoint = self._phase_to_time(phi_setpoint+2*np.pi, cal_type="iq")
        iq_time_ch1_at_zero, iq_time_ch2_at_zero = self.fit_result_iq_calib_ch1.params['t0'].value, self.fit_result_iq_calib_ch2.params['t0'].value

        xlimits = (
            0.05 / self.rp.asg0.frequency,
            0.5 / self.rp.asg0.frequency
        )
        fig, ax = plt.subplots(3, 2, figsize=(15, 7), sharex=True)
        for axis in ax.flatten():
            axis.set_xlim(xlimits)

        ax[0,0].plot(self.det_calib_time, self.det_calib_ch1, label='Data Ch1', c="k", alpha=0.4)
        ax[0,0].plot(det_time_ch1_at_setpoint, self.fit_result_det_calib_ch1.eval(time=det_time_ch1_at_setpoint), "o", markersize=10, c="g")
        ax[0,0].axhline(self.fit_result_det_calib_ch1.params['offset'].value, c='k', ls='--', lw=1)
        ax[0,0].axvline(det_time_ch1_at_zero, c='k', ls='--', lw=1)
        ax[0,0].set_xlim(ax[1,1].get_xlim())
        ax[0,0].plot(np.linspace(ax[0,0].get_xlim()[0], ax[0,0].get_xlim()[1], 10000), self.fit_result_det_calib_ch1.eval(time=np.linspace(ax[0,0].get_xlim()[0], ax[0,0].get_xlim()[1], 10000)), label='Fit Ch1', c="purple")

        ax[1,0].plot(self.det_calib_time, self.det_calib_ch2, label='Data Ch2', c="k", alpha=0.4)
        ax[1,0].plot(det_time_ch2_at_setpoint, self.fit_result_det_calib_ch2.eval(time=det_time_ch2_at_setpoint), "o", markersize=10, c="g")
        ax[1,0].axhline(self.fit_result_det_calib_ch2.params['offset'].value, c='k', ls='--', lw=1)
        ax[1,0].axvline(det_time_ch2_at_zero, c='k', ls='--', lw=1)
        ax[1,0].set_xlim(ax[1,1].get_xlim())
        ax[1,0].plot(np.linspace(ax[1,0].get_xlim()[0], ax[1,0].get_xlim()[1], 10000), self.fit_result_det_calib_ch2.eval(time=np.linspace(ax[1,0].get_xlim()[0], ax[1,0].get_xlim()[1], 10000)), label='Fit Ch2', c="orange")
        
        ax[2,0].plot(self.det_calib_time, self.det_calib_ch1 - self.det_calib_ch2, label='Difference signal', c="k", alpha=0.4)
        ax[2,0].plot(det_time_ch2_at_setpoint, self.fit_result_det_calib_ch1.eval(time=det_time_ch2_at_setpoint) - self.fit_result_det_calib_ch2.eval(time=det_time_ch2_at_setpoint), "o", markersize=10, c="g")
        ax[2,0].axhline(self.fit_result_det_calib_ch1.params['offset'].value - self.fit_result_det_calib_ch2.params['offset'].value, c='k', ls='--', lw=1)
        ax[2,0].axvline(det_time_ch2_at_zero, c='k', ls='--', lw=1)
        ax[2,0].set_xlabel('Time (s)')
        ax[2,0].set_xlim(ax[1,1].get_xlim())
        ax[2,0].plot(np.linspace(ax[2,0].get_xlim()[0], ax[2,0].get_xlim()[1], 10000), self.fit_result_det_calib_ch1.eval(time=np.linspace(ax[2,0].get_xlim()[0], ax[2,0].get_xlim()[1], 10000)) - self.fit_result_det_calib_ch2.eval(time=np.linspace(ax[2,0].get_xlim()[0], ax[2,0].get_xlim()[1], 10000)), label='Fit', c="r")
        
        ax[0,1].plot(self.iq_calib_time, self.iq_calib_ch1, label='Data Ch1', c="k", alpha=0.4)
        ax[0,1].plot(iq_time_ch1_at_setpoint, self.fit_result_iq_calib_ch1.eval(time=iq_time_ch1_at_setpoint), "o", markersize=10, c="g")
        ax[0,1].axhline(self.fit_result_iq_calib_ch1.params['offset'].value, c='k', ls='--', lw=1)
        ax[0,1].axvline(iq_time_ch1_at_zero, c='k', ls='--', lw=1)
        ax[0,1].set_xlim(ax[1,1].get_xlim())
        ax[0,1].plot(np.linspace(ax[0,1].get_xlim()[0], ax[0,1].get_xlim()[1], 10000), self.fit_result_iq_calib_ch1.eval(time=np.linspace(ax[0,1].get_xlim()[0], ax[0,1].get_xlim()[1], 10000)), label='Fit Ch1', c="purple")
        
        ax[1,1].plot(self.iq_calib_time, self.iq_calib_ch2, label='Data Ch2', c="k", alpha=0.4)
        ax[1,1].plot(iq_time_ch2_at_setpoint, self.fit_result_iq_calib_ch2.eval(time=iq_time_ch2_at_setpoint), "o", markersize=10, c="g")
        ax[1,1].axhline(self.fit_result_iq_calib_ch2.params['offset'].value, c='k', ls='--', lw=1)
        ax[1,1].axvline(iq_time_ch2_at_zero, c='k', ls='--', lw=1)
        ax[1,1].set_xlim(ax[1,1].get_xlim())
        ax[1,1].plot(np.linspace(ax[1,1].get_xlim()[0], ax[1,1].get_xlim()[1], 10000), self.fit_result_iq_calib_ch2.eval(time=np.linspace(ax[1,1].get_xlim()[0], ax[1,1].get_xlim()[1], 10000)), label='Fit Ch2', c="orange")
        
        ax[2,1].set_xlabel('Time (s)')

        for axis in ax.flatten():
            
            axis.set_ylabel('Detector Signal (V)')
            axis.legend(fontsize=12, loc="upper right")

        fig.tight_layout(h_pad=0)

        return fig, ax

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
    obj = RPLockboxMZI(hostname="10.135.71.245", gui=True, config_file='travis_global_config', yml_file='lockbox_config.yml')