import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import Model, update_param_vals


#Why make a class out of this instead of just calling the functions where you are doing the fitting?
# # def signal_func():

# # if 
# Model(func1)
# #
# Model(func2)
class SinusoidalModel(Model):
    """Merged model that contains the functionality previously in UMZIModel2.

    This class wraps an lmfit.Model and provides helpers for guessing
    parameters, performing the fit, locking phase from another model,
    and utilities to extract phase/time relations.
    """
    def __init__(self, cal_type, channel, guesses=None):
        self.channel = channel
        self.cal_type = cal_type

        if self.cal_type == "det": 
            self.fit_func_str = "cos"
            self.func = lambda time, amp, offset, T, phase: \
                offset + amp * np.cos(2 * np.pi / T * time + phase)
            
        elif self.cal_type == 'iq' and self.channel == 1:   
            self.fit_func_str = "(-sin)"
            self.func = lambda time, amp, offset, T, phase: \
                offset - amp * np.sin(2 * np.pi / T * time + phase)

        elif self.cal_type == 'iq' and self.channel == 2:
            self.fit_func_str = "(-cos)"
            self.func = lambda time, amp, offset, T, phase: \
                offset - amp * np.cos(2 * np.pi / T * time + phase)

        super().__init__(self.func)

        self.prefix = f"{cal_type}_{channel}_"

        self.other_model = None
        self.guesses = guesses
        self.fit_result = None
        # placeholders populated after fit
        self.amp_fit = None
        self.offset_fit = None
        self.phase_fit = None
        self.T_fit = None
        #self.phase = None
        self.time_data = None
        self.signal_data = None

    def guess(self, data, time=None, **kwargs):
        """Set initial parameter hints for the model based on data."""
        self.set_param_hint("amp", value=np.max(data) - np.mean(data), vary=True, min=0.1)

        if self.cal_type == "iq":
            self.set_param_hint("offset", min=-0.1, max=0.1, vary=True)
        else:
            self.set_param_hint("offset", min=-1, max=1, vary=True)

        #CAN WE LEAVE OUT THESE VALUES? I THINK THEY ARE VERY SETUP SPECIFIC.
        self.set_param_hint("T", min=0.01, max=0.02, vary=True) 
        self.set_param_hint("phase", min=-1*np.pi, max=1*np.pi, vary=True)

        params = self.make_params()
        return update_param_vals(params, self.prefix, **kwargs)

    def perform_fit(self, data, params, time): #Fit the model to the data

        self.fit_result = self.fit(data, params, time=time, max_nfev=10_000, fit_kws={"ftol": 1e-100, "xtol": 1e-100, "gtol": 1e-100, "epsfcn": 1e-100})

        #Get the values of the optimal params from the fit done one line above
        self.amp_fit = self.fit_result.params[self.prefix+"amp"].value
        self.offset_fit = self.fit_result.params[self.prefix+"offset"].value
        self.phase_fit = self.fit_result.params[self.prefix+"phase"].value
        self.T_fit = self.fit_result.params[self.prefix+"T"].value
        """
        self.phase = self.get_phase_from_time(time) #function defined below. Get the phase when applying 0 V. 
        self.phase -= 2*np.pi * (np.max(self.phase)//(2*np.pi)) #Get the phase within 0 to 2 pi
        """
        self.time_data = time
        self.signal_data = data

        self.fit_result.params.pretty_print()

    #Called before doign a peform_fit. 
    def lock_phase_guess(self, params, other_model, **kwargs):
        """
        Takes over the V_pi from the detector model of the same channel. 
        Sets the phase very closely paramsto that of the detector model of the same channel. 
        """
        other_T = other_model.fit_result.params[other_model.prefix+"T"].value
        other_phase = other_model.fit_result.params[other_model.prefix+"phase"].value
        self.other_model = other_model
        params[self.prefix+"T"].set(
            value = other_T,
            vary = False
        )
        params[self.prefix+"phase"].set(
            value = other_phase,
            vary = True #phase can drift a little bit. 
        )
        return update_param_vals(params, self.prefix, **kwargs)
    
    def create_calibration_plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.time_data, self.signal_data, label='Data', color='blue')
        ax.plot(self.time_data, self.fit_result.best_fit, label='Fit', color='red', ls="--")
        #ax.plot(self.time_data, self.fit_result.init_fit, label='Initial Fit', color='grey', linestyle='--', alpha=0.1)

        ax.set_title(
            (
                f"""{self.cal_type}, {self.channel}: {self.amp_fit:.3f}""" + 
                self.fit_func_str + f"""( 2 pi t / ({self.T_fit:.3f}s) + {self.phase_fit:.3f} ) \
                + {self.offset_fit:.3f}"""
            ),
            fontsize=8
        )

        ax.axhline(self.offset_fit, color="k", linestyle="--")
        ax.set_xlim(self.time_data[0], self.time_data[-1])
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Signal")
        ax.legend()

        return fig, ax

    
    """
    def get_phase_from_time(self, time):  #does this return an aray?? 
        return 2 * np.pi/self.T_fit * time + self.phase_fit
    #2 pi / T * t + phase_fit

    def print_params(self):
        if self.fit_result is not None:
            self.fit_result.params.pretty_print()
        else:
            self.print_params()  
    
    def phase_relation(self, phase=None, amp=None, offset=None): #What does this function do??
        if phase is None and self.phase is not None:
            phase = self.phase
        if amp is None and self.amp_fit is not None:
            amp = self.amp_fit
        if offset is None and self.offset_fit is not None:
            offset = self.offset_fit
        
        return amp * np.sin(phase) + offset
    
    def get_phase_from_voltage(self, voltage, function_increasing=True):
        voltage_clipped = np.clip(voltage, self.offset_fit - np.abs(self.amp_fit), self.offset_fit + np.abs(self.amp_fit))
        # if voltage_clipped != voltage:
        #     print(f"Warning: Voltage {voltage} clipped to {voltage_clipped} to fit within the range of the {self.prefix} model.")

        answer_between_0_pi = np.arccos((voltage_clipped - self.offset_fit) / self.amp_fit)
        if self.channel == 1:
            # Channel 1 is the - cos(phi)
            answer_between_0_2pi = answer_between_0_pi if function_increasing else 2*np.pi - answer_between_0_pi
        elif self.channel == 2:
            # Channel 2 is the + cos(phi)
            answer_between_0_2pi = np.pi - answer_between_0_pi if function_increasing else answer_between_0_pi

        return answer_between_0_2pi
    """

""" Old garbage
class SineModel(UMZIModel2):
    def __init__(self, cal_type, channel):
        super().__init__(self.evaluate, cal_type)
        self.prefix = f"{cal_type}_{channel}_"
        self.channel = channel
        self.fit_func_str = "sin"

    def evaluate(self, time, amp, offset, V_pi, phase):  #Outputs the expected voltage. 
        # modulation = np.cos(2*np.pi*freq*time+t0)
        modulation = time #Why do you call it modulation?
        #WHAT IS THE DIFFERENCE WITH THE LINE ABOVE IN THE GENERAL MODEL? 
        actual_modulation_phase = np.pi/V_pi * modulation

        return amp * np.sin(actual_modulation_phase + phase) + offset
    
    def phase_relation(self, phase=None, amp=None, offset=None): #What does this function do??
        if phase is None and self.phase is not None:
            phase = self.phase
        if amp is None and self.amp_fit is not None:
            amp = self.amp_fit
        if offset is None and self.offset_fit is not None:
            offset = self.offset_fit
        
        return amp * np.sin(phase) + offset

    
class CosineModel(UMZIModel2):
    def __init__(self, cal_type, channel):
        super().__init__(self.evaluate, cal_type)
        self.prefix = f"{cal_type}_{channel}_"
        self.channel = channel
        self.fit_func_str = "cos"

    def evaluate(self, time, amp, offset, V_pi, phase):
        # modulation = np.cos(2*np.pi*freq*time+t0)
        modulation = time
        actual_modulation_phase = np.pi/V_pi * modulation
        return amp * np.cos(actual_modulation_phase + phase) + offset

    def phase_relation(self, phase=None, amp=None, offset=None):
        if phase is None and self.phase is not None:
            phase = self.phase
        if amp is None and self.amp_fit is not None:
            amp = self.amp_fit
        if offset is None and self.offset_fit is not None:
            offset = self.offset_fit
        
        return amp * np.cos(phase) + offset
    
    def get_phase_from_voltage(self, voltage, function_increasing=True):
        voltage_clipped = np.clip(voltage, self.offset_fit - np.abs(self.amp_fit), self.offset_fit + np.abs(self.amp_fit))
        # if voltage_clipped != voltage:
        #     print(f"Warning: Voltage {voltage} clipped to {voltage_clipped} to fit within the range of the {self.prefix} model.")

        answer_between_0_pi = np.arccos((voltage_clipped - self.offset_fit) / self.amp_fit)
        if self.channel == 1:
            # Channel 1 is the - cos(phi)
            answer_between_0_2pi = answer_between_0_pi if function_increasing else 2*np.pi - answer_between_0_pi
        elif self.channel == 2:
            # Channel 2 is the + cos(phi)
            answer_between_0_2pi = np.pi - answer_between_0_pi if function_increasing else answer_between_0_pi

        return answer_between_0_2pi
"""


