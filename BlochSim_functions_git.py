#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 06:31:01 2023

@author: etorr
"""

from cmath import pi
import matplotlib.pyplot as plt
#import mpmath as mp
import numpy as np
import cmath
import scipy

## Defining classes
class PulseParameters:
    def __init__(self, R_value, BW_omega, Nt_Pts, pulse_order):
        self.R_value = R_value
        self.BW_omega = BW_omega
        self.Nt_Pts = Nt_Pts
        self.pulse_order = pulse_order
        self.Tp = R_value / (BW_omega)
        self.dwell_time = self.Tp / (Nt_Pts)
        
class PulseParameters_AM:
    def __init__(self, Nt_Pts, Tp):
        self.Nt_Pts = Nt_Pts
        self.Tp = Tp
        self.dwell_time = self.Tp / (Nt_Pts)
        

def plot_ET(num_subplots):
    fig, axs = plt.subplots(num_subplots, figsize=(7,4), dpi=300,  constrained_layout = True)
    fig.patch.set_facecolor('xkcd:white')
    fig.tight_layout()
    plt.style.use('classic') 
    
    return fig, axs

def HSpulse(R_value, pulse_order, Nt_Pts):

    #truncation level hard-coded
    trunc_level = 5.2983
    
    #dummy variable
    tau = np.linspace(-1, 1, Nt_Pts)
    
    # creation of AM function
    #sech_array = np.frompyfunc(mp.sech, 1, 1)
    #AM = sech_array(trunc_level*tau**pulse_order)
    AM = 1/(np.cosh(trunc_level*tau**pulse_order))

    # FM Function 
    dphiRF = np.flip(np.cumsum(AM**2))
    dphiRF /= max(abs(dphiRF))
    dphiRF -= 0.5
    dphiRF /= max(abs(dphiRF))
    dphiRF *=(.5) * R_value

    # PM function
    PM = 2*pi*np.cumsum(dphiRF)
    PM /= Nt_Pts

    return AM, PM


def Bloch_sim(M_state, AM_function, PM_function, w1max, tstep, off_Res):
    ## inputs: 
        ## - M_state: 3 by N numpy array, where N is number of points defining object
        ## - AM_function: 1 by NtPts numpy array, where NtPts is number of pulse, normalized to one
        ## - PM_function: 1 by NtPts numpy array, units: radians
        ## - w1max: gamma*B1 max numpy array, 1 by N numpy array
        ## - tstep: time-step in seconds
        ## - off_Res: 1 by N numpy array defining off-resonance in Hz
    ## Outputs: M_state
    
    if type(AM_function) == np.float64:
        Nt_Pts = 0
    else: 
        Nt_Pts = len(AM_function)
        
    B_field = np.zeros([3, len(w1max)])
    
    for jj in range(Nt_Pts):
        AM_value = w1max*AM_function[jj]
        PM_value = PM_function[jj]
        
        weff = np.sqrt(AM_value**2 + off_Res**2)
        phi = -2 * cmath.pi * weff * tstep
        if np.all(np.abs(phi) > 1.0e-12):
            B_field[0,:] = AM_value * np.cos(PM_value);
            B_field[1,:] = AM_value * np.sin(PM_value);
            B_field[2,:] = off_Res;
            
            
            cross_product = np.cross(B_field, M_state, axis = 0) 
            #dot_product = np.dot(np.squeeze(B_field), np.squeeze(M_state)) / weff
            dot_product = np.sum(np.multiply(B_field, M_state), 0) / weff
            
            cross_product = cross_product / weff
            M_state = np.cos(phi)*M_state + np.sin(phi)*cross_product + (1 - np.cos(phi))*dot_product*B_field / weff
    
    return M_state