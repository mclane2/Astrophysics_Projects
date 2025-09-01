# Astrophysics_Projects
This repository contains some astrophysics projects I completed during my final year of studying "Physics and Astrophysics" at Trinity College Dublin. 

## Respository Contents
```Exoplanets_RV_curves.py```- This code plots RV curves for an exoplanet orbiting a star of known mass for any chosen value of eccentricity and inclination. The final result is a plot of RV curves for an exoplanet for several different arguments of periastron. At the beginning of the code, input the mass of the exoplanet, the mass of the host star, eccentricity, inclination, period and phase offset. At first this seems like a simple task, however for eccentric orbit we need to find the true anomaly as a function of time, which requires solving Kepler's equation. This equation must be solved numerically using the Newton-Raphson method by iteratively updating the mean anomaly until suitable percision is reached. My plots confirm the well known result that changing the argument of periastron doesn't change the RV semi-amplitude.

```RV_curve_proof.pdf``` - A short theoretical proof of the result that the RV semi-amplitude does not depend on the argument of periastron.

```Protoplanetary_disk_calculations.ipynb``` - This code computes a variety of useful/important quantites in the study of protoplanetary disks given numerical observations from a specific disk as input. The formulas and theory behind them is discussed in ```Planetary_System_Formation_Notes.pdf```. Follow the instructions in the markdowns to change the observed disk properties

```Planetary_System_Formation_Notes.pdf``` - 31 pages of notes that reflect my own summary and understanding of lectures presented by Professor Luca Matra for the module "Planetary and Space Science" at Trinity College Dublin. These notes are not just a summary of the course, but a detailed walk-through of the fundamental physics explaining planetary disks from gas cloud cores to the formation of terrestrial planets and gas giants.

## Installation
The minimum requirements for ```Exoplanets_RV_curves.py``` and ```Protoplanetary_disk_calculations.ipynb``` are Python 3.9+, numpy, scipy and matplotlib. All the required functions are defined within the Python file/Juypter notebook.

## Contact
Email: **Marc Lane** - lanem2@tcd.ie





