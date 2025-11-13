import numpy as np
import matplotlib.pyplot as plt
from lmfit.models import Model, update_param_vals


#Why make a class out of this instead of just calling the functions where you are doing the fitting?

class UMZIModel(Model):  #A parent class of the cos_model and sin_model, and that is all I think...
    def __init__(self, func, cal_type):
        super().__init__(func)
        self.cal_type = cal_type #are we fitting iq or det
        self.other_model = None #not sure what this is used for yet

    def guess(self, data, time, **kwargs): #This function sets the params of it's parent the Model depending on the variable cal_type of the parent
                                            #We can delete the time argument
        """ I changed this to below
        if self.cal_type == "iq":
            amp_neg = self.channel == 1
        elif self.cal_type == "det":
            amp_neg = self.channel == 1 ###########################What is det????????????????????????????????????
        """ 
        if self.cal_type == "iq" or self.cal_type == "det": #If we are either fitting the iq or the det, then we set amp_neg to 
                                                            #True if channel is 1
            amp_neg = self.channel == 1                     #I CANNOT FIND ANYTHING ABOUT CHANNEL IN THE MODEL OR THIS CLASS??????????
            
        #WHY DOES A CHANNEL = 1 CALL FOR A NEGATIVE AMPLITUDE??? 
        if amp_neg: #If we are either fitting iq or det, and channel == 1, then we set the first guess for the amplitude of the model as: 
            self.set_param_hint("amp", value=np.min(data) - np.mean(data), vary=True, max=-0.1) 
        else:
            self.set_param_hint("amp", value=np.max(data) - np.mean(data), vary=True, min=0.1)
        
        if self.cal_type == "iq": #set the offset for the iq model, of course we want it to be zero for iq
            self.set_param_hint("offset", value=0, min=-0.3, max=0.3, vary=True)
        else:
            self.set_param_hint("offset", value=np.max(data)/2, min=-1, max=1, vary=True)

        # self.set_param_hint("V_pi", value=0.13, min=0, max=1, vary=True)
        self.set_param_hint("V_pi", value=0.009, min=0.001, max=0.1, vary=True) #What is this V_pi? 
        self.set_param_hint("phase", value=0, min=-2*np.pi, max=2*np.pi, vary=True)

        params = self.make_params()
        return update_param_vals(params, self.prefix, **kwargs) #(inhereted) function of the class to set the param guesses according to the hints

    def perform_fit(self, data, params, time): #Fit the model to the data

        #Fit the data parsed. Perhaps right before this is called, they call a guess function, but I would call it here!!!!!!!!!!!!!!!!!!!!!!!
        self.fit_result = self.fit(data, params, time=time, max_nfev=10_000, fit_kws={"ftol": 1e-100, "xtol": 1e-100, "gtol": 1e-100, "epsfcn": 1e-100})

        #Get the values of the optimal params from the fit done one line above
        self.amp_fit = self.fit_result.params[self.prefix+"amp"].value
        self.offset_fit = self.fit_result.params[self.prefix+"offset"].value
        self.phase_fit = self.fit_result.params[self.prefix+"phase"].value
        self.V_pi_fit = self.fit_result.params[self.prefix+"V_pi"].value

        self.phase = self.get_phase_from_time(time) #function defined below
        self.phase -= 2*np.pi * (np.max(self.phase)//(2*np.pi)) #Get the phase within 0 to 2 pi
        #Why is phase an array??? 

        self.time_data = time
        self.signal_data = data

        if self.cal_type == "iq": #?????????????????????????????????????????????????????????????????????????
            self.phi_prime = self.phase_fit - self.other_model.phase_fit
    
    def lock_phase_guess(self, params, other_model, **kwargs):
        other_Vpi = other_model.fit_result.params[other_model.prefix+"V_pi"].value
        other_phase = other_model.fit_result.params[other_model.prefix+"phase"].value
        self.other_model = other_model
        params[self.prefix+"V_pi"].set(
            value = other_Vpi,
            vary = False
        )
        params[self.prefix+"phase"].set(
            value = other_phase,
            vary = True
        )
        return update_param_vals(params, self.prefix, **kwargs)
    
    def get_phase_from_time(self, time):  #does this return an aray?? 
        modulation = time
        return np.pi/self.V_pi_fit * modulation + self.phase_fit
    
    def create_calibration_plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.time_data, self.signal_data, label='Data', color='blue')
        ax.plot(self.time_data, self.fit_result.best_fit, label='Fit', color='red', ls="--")
        ax.plot(self.time_data, self.fit_result.init_fit, label='Initial Fit', color='grey', linestyle='--', alpha=0.1)
        if self.other_model:
            ax.set_title(
                r"$S_{%.i}$"%self.channel + 
                f" = {self.amp_fit:.3f} {self.fit_func_str}(" + r"$\phi$" + 
                (f" + {(self.phase_fit - self.other_model.phase_fit)/np.pi:.3f}" if self.phase_fit > self.other_model.phase_fit else f" - {-(self.phase_fit - self.other_model.phase_fit)/np.pi:.3f}") +
                r"$\pi$)" + 
                (f" + {self.offset_fit:.3f}" if self.offset_fit > 0 else f" - {np.abs(self.offset_fit):.3f}"),
                fontsize=8
            )
        else:
            ax.set_title(
                (
                    r"$S_{%.i}$"%self.channel + 
                    f" = {self.amp_fit:.3f} {self.fit_func_str}(" +
                    r"$\phi$" + 
                    (f") + {self.offset_fit:.3f} \n" if self.offset_fit > 0 else f") - {np.abs(self.offset_fit):.3f} \n") +
                    r"$\phi$" + 
                    (f" = {np.pi/self.V_pi_fit:.3f} * t + {self.phase_fit/np.pi:.3f}" if self.phase_fit > 0 else f" = {np.pi/self.V_pi_fit:.3f} * t - {np.abs(self.phase_fit/np.pi):.3f}") +
                    r"$\pi$"
                ),
                fontsize=8
            )
        ax.axhline(self.offset_fit, color="k", linestyle="--")
        ax.set_xlim(self.time_data[0], self.time_data[-1])
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Signal")
        ax.legend()

        return fig, ax

    def print_params(self):
        if self.fit_result is not None:
            self.fit_result.params.pretty_print()
        else:
            self.print_params()


class SineModel(UMZIModel):
    def __init__(self, cal_type, channel):
        super().__init__(self.evaluate, cal_type)
        self.prefix = f"{cal_type}_{channel}_"
        self.channel = channel
        self.fit_func_str = "sin"

    def evaluate(self, time, amp, offset, V_pi, phase):  #What does this function do??
        # modulation = np.cos(2*np.pi*freq*time+t0)
        modulation = time
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

    
class CosineModel(UMZIModel):
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



