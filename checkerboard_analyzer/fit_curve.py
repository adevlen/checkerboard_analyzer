'''
Code to model single drug response using the 4PL Hill equation and plot the resulting curve.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

class DoseResponseCurve:

    def __init__(self, concentrations, responses):
        ''' Initializes an instance of the DoseResponseCurve class. '''
        self.concentrations = np.asarray(concentrations)
        self.responses = np.asarray(responses)
        self.params = {}

    def initial_guesses(self):
        ''' Populates params member variable with initial guesses for Emin, Emax, EC50, h. '''
        emin_init = np.min(self.concentrations)
        emax_init = np.max(self.concentrations)
        ec50_init = np.median(self.concentrations)
        h_init = 1.0
        return [emin_init, emax_init, ec50_init, h_init]
        
    @staticmethod
    def _hill_equation(x, Emin, Emax, EC50, h):
        ''' Defines the 4PL Hill equation for the given parameters and returns the drug response. '''
        return Emin + (Emax - Emin) / (1.0 + (EC50 / x)**h)
    
    def curve_fit(self):
        ''' Populates params member variable and runs non-linear least squares optimization. '''
        p0 = self.initial_guesses()

        popt, _ = curve_fit(self._hill_equation, self.concentrations, self.responses, p0) # run optimization

        self.params = {         # store values in params
            "Emin" : popt[0],
            "Emax" : popt[1],
            "EC50" : popt[2],
            "EC50" : popt[3],
        }

    def predict(self, x):
        ''' Predict responses for given concentrations using the fitted 4PL model. '''
        if self.params is None:
            raise ValueError("DoseResponseCurve instance has not yet been fit.")
        else:
            return self._hill_equation(
                x, 
                self.params["Emin"], 
                self.params["Emax"], 
                self.params["EC50"], 
                self.params["h"]
            )

    def inverse_predict(self, response):
        ''' Calculates drug concentration based on response. '''
        if self.params is None:
            raise ValueError("DoseResponseCurve instance has not yet been fit.")
        else:
            second_term = ((self.params["Emax"] - self.params["Emin"] / (response - self.params["Emin"])) - 1)
            return self.params["EC50"] * (second_term**(1/self.params["h"]))



