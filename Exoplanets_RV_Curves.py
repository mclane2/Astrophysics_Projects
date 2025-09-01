# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 12:24:00 2025

@author: marcl
"""

import numpy as np
import matplotlib.pyplot as plt

# Input system parameters
M_s = 0.99*(1.989e30) # Star mass in solar masses
M_p = 1.34*(1.898e27) # Planet mass in earth masses
e = 0.38              # Eccentricity
P = 8.61*(24*60*60)   # Period in days
i = np.radians(90)    # Inclination
phi = 0               # Phase Offset
G = 6.67e-11

# Computing a using Kelper's 3rd law
a = ( (G*(M_s+M_p)*(P**2))/(4*(np.pi**2)) )**(1/3)
# Computing RV semi-amplitude
K = (M_p * np.sin(i)) / ((M_p + M_s)**(2/3)) * ((2 * np.pi * G) / P)**(1/3) * (1 / np.sqrt(1 - e**2))

# For calculating the true anomaly f as a function of the eccentric anomaly E
def true_anomaly(E):

    # Definition of f
    F = np.arccos((np.cos(E) - e) / (1 - e * np.cos(E))) 

    # Solving for the correct quadrant 
    if E > np.pi:
        F = 2 * np.pi - F 
    
    return F

# Calculate the radial velocity as a function of f (which is a function of time) and argument of periastron (w)

def radial_velocity(f, w):
    f = np.array(f)
    
    # Compute the radial velocity of the planet
    v_r = K * (e * np.cos(w) + np.cos(f + w))
    
    return v_r

# Solve equations for two full orbital periods
time = np.linspace(0,2*P,1000)
f_array = []

for t in time:
    # Calculating mean anomaly 
    M = (t*((2*np.pi)/(P))) + phi
    
    # initial guesses
    tau = 100
    E = 5
    
    # Using Newton-Raphson method to solve Kepler's equation
    while np.abs(tau) > 0.000001:
        
        f = E - e*np.sin(E) - M # Kepler's equation
        df = 1 - e*np.cos(E)    # Derivative of Kepler's equation
        
        E_new = E - (f/df) # Newton-Raphson iterative update step
        
        tau = E_new-E # update convergence condition
        
        E = E_new # update E
    
    # Ensuring E always stays within the 0,2pi (one full period) range so that the solution at 0 degrees = solution 360 degrees 
    E_new = E_new % (2 * np.pi)
    # Computing the true anomaly f for the optimised value of E
    f_t = true_anomaly(E_new)
    f_array.append(f_t)

# Convert the time axis from seconds to days
time_days = time / 86400

# Plotting the RV curves for different arguments of periastrons
plt.figure(figsize=(10, 6))
plt.plot(time_days, radial_velocity(np.array(f_array),w=0), color='#FF0000', label=r'$\omega = 0$')
plt.plot(time_days, radial_velocity(f_array,w=np.pi/3), color='green', label=r'$\omega = \frac{\pi}{3}$')
plt.plot(time_days, radial_velocity(f_array,w=2*np.pi/3), color='#0000FF', label=r'$\omega = \frac{2\pi}{3}$')
plt.plot(time_days, radial_velocity(f_array,w=np.pi), color='#FF1493', label=r'$\omega = \pi}$')
plt.plot(time_days, radial_velocity(f_array,w=4*np.pi/3), color='#FFD700', label=r'$\omega = \frac{4\pi}{3}$')
plt.plot(time_days, radial_velocity(f_array,w=5*np.pi/3), color='#8A2BE2', label=r'$\omega = \frac{5\pi}{3}$')
plt.xlim(0,max(time_days))
plt.xlabel("Time [days]", fontsize=14)
plt.ylabel("Radial Velocity [m/s]", fontsize=14)
plt.legend(frameon=False, bbox_to_anchor=(1.01, 0.9), loc='center left')

# Computing the RV semi-amplitude
K0 = np.max(radial_velocity(np.array(f_array),w=0)) - np.min(radial_velocity(np.array(f_array),w=0))

plt.text(11.5, 160, f'K = {0.5*np.round(K0)}', fontsize=14,rotation=0)

plt.tick_params(direction='in', top=True, right=True, labelsize=12)
plt.tight_layout()
plt.savefig("ultra true RV Curve.png", dpi=300)
plt.show()
K0 = 0.5*(np.round(np.max(radial_velocity(np.array(f_array),w=0)) - np.min(radial_velocity(np.array(f_array),w=0))))
K1 = 0.5*(np.round(np.max(radial_velocity(np.array(f_array),w=np.pi/3)) - np.min(radial_velocity(np.array(f_array),w=np.pi/3))))
K2 = 0.5*(np.round(np.max(radial_velocity(np.array(f_array),w=2*np.pi/3)) - np.min(radial_velocity(np.array(f_array),w=2*np.pi/3))))
K3 = 0.5*(np.round(np.max(radial_velocity(np.array(f_array),w=np.pi)) - np.min(radial_velocity(np.array(f_array),w=np.pi))))
K4 = 0.5*(np.round(np.max(radial_velocity(np.array(f_array),w=4*np.pi/3)) - np.min(radial_velocity(np.array(f_array),w=4*np.pi/3))))
K5 = 0.5*(np.round(np.max(radial_velocity(np.array(f_array),w=5*np.pi/3)) - np.min(radial_velocity(np.array(f_array),w=5*np.pi/3))))
print("RV Semi amplitudes:",K0,K1,K2,K3,K4,K5)
