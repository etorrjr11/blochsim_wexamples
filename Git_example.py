#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bloch Sim with simple example
@author: Efrain Torres, 
Institute: University of Minnesota

This code snippet demonstrates how to use the Bloch sim function and plot an example RF pulse. 

The RF pulse that is plotted is the HS pulse. We plot its performance across different
B1 and B0 field values to demonstrate its resilency to imperfections. 

"""

# Imports
import matplotlib.pyplot as plt
import numpy as np
import BlochSim_functions_git


def check_inversion_profile(pulse_parameters, AM, PM, w1max_values, off_Res_values): 
    Off_Res_pts = np.size(off_Res_values)
    Obj_Pts = 100
    Mz_profile = np.zeros([Off_Res_pts, Obj_Pts])
    
    for k in range(Off_Res_pts):
        off_Res = np.ones([1, Obj_Pts])*off_Res_values[k]
        # defines a 3xN vector that represents the magnetization
        M = np.zeros([3, Obj_Pts])
        M[2,:] += 1  
        
        #Utilizing the bloch sim to numerically simulate the Pulse
        M = BlochSim_functions_git.Bloch_sim(M, AM, PM, w1max_values, pulse_parameters.dwell_time, off_Res)
        
        #gathering output
        Mz_profile[k,:] = M[2,:]
    
    ## Plotting
    fig, axs = BlochSim_functions_git.plot_ET(1)
    plt.imshow(Mz_profile, cmap = 'jet', 
               extent=[min(w1max_values)/1000, max(w1max_values)/1000,(max(off_Res_values))/1000, -(max(off_Res_values))/1000])
    plt.xlabel('$\gamma B_{1} \cdot 2\pi^{-1} (kHz)$', fontsize = 15)
    plt.ylabel('$\Omega \cdot 2\pi^{-1} (kHz)$', fontsize = 15)
    plt.title('Inversion Profile of Pulse')
    
    cb = plt.colorbar()
    cb.set_label(label='abs' ,size = 15)
    axs.tick_params(labelsize=15)

    return Mz_profile

## Pulse parameters and determing the values of B1 and B0 frequencies that we are sweeping across
R_value_1 = 20
BW_omega  = 10e3 # Hz
Nt_Pts = 500 
Pulse_order = 1
w1max_values = np.linspace(0, 5e3, 100) # Hz
off_Res_values = np.linspace(-10e3, 10e3, 100) #Hz

## Example using an HS pulse
AFP1 = BlochSim_functions_git.PulseParameters(R_value_1, BW_omega, Nt_Pts, Pulse_order)
AM_1, PM_1 = BlochSim_functions_git.HSpulse(R_value_1, Pulse_order, Nt_Pts) ## Uses phase-modulated functions 
check_inversion_profile(AFP1, AM_1, PM_1, w1max_values, off_Res_values)

## Here is another example using a square pulse
hard_pulse = np.ones((1, 100))
Tp = 100e-6 # seconds 
AM_parameters = BlochSim_functions_git.PulseParameters_AM(hard_pulse.shape[0], Tp) 
check_inversion_profile(AM_parameters, hard_pulse, hard_pulse*0, w1max_values, off_Res_values) ## Requires PM input, AM pulses have none
