# coding: utf-8
# date: started on 11 April 2020
# author: matthieu VANDAMME
# Code associated to the manuscript "Modeling transient variations of permeability in coal seams at the reservoir scale" by N. Abouloifa, M. Vandamme, and P. Dangla

import matplotlib.pyplot as plt
import numpy as np

NPZ_FILES_TO_ANALYZE =['perm1E-13_simulated_equispaced.npz',
                       'perm1E-12_simulated_equispaced.npz',
                       'perm1E-11_simulated_equispaced.npz',
                       'perm1E-12_fitted_equispaced.npz']

NPZ_FILES_TO_ANALYZE =['perm1E-13_simulated_equispaced.npz']

# Nothing to modify below this line

plt.close('all')

# to find out all the indices of the times to display
def find_indices(t_to_display, t):
    indices = []
    for i in t_to_display:
        indices.append(np.abs(t - i).argmin())
    return indices

# Parameters of interaction between fluid and coal
MOLAR_MASS_OF_CO2 = 44E-3 # kg/mol
C_M_MAX = 2.4E3 * MOLAR_MASS_OF_CO2 # kg/m3, Langmuir parameter that governs the maximum adsorbed amount
LANGMUIR_PRESSURE = 1.6E6 # Pa, Langmuir characteristic pressure
# kg/m3, returns the adsorbed amount per unit volume of coal matrix based on 
# pressure p, 
def langmuir_isotherm(p):
    return C_M_MAX * p / (p + LANGMUIR_PRESSURE)

for i in NPZ_FILES_TO_ANALYZE:
    npz_file = np.load(i)
    times = npz_file['times']
    pressure = npz_file['pressure']
    permeability = npz_file['permeability']
    positions = npz_file['positions']
    adsorbed_amount = npz_file['adsorbed_amount']
    ref_permeability = npz_file['ref_permeability']
    type_of_kernel = npz_file['type_of_kernel']

    # plot of pressure profile at various times
    if (ref_permeability == 1E-13):
        times_to_display = [0, 10000, 30000, 50000, 100000, 150000, 200000]
        times_to_display = [0, 10000, 30000, 50000, 80000, 110000, 140000, 170000, 200000]
    if (ref_permeability == 1E-12):
        times_to_display = [0, 300, 1000, 3000, 10000, 30000, 100000]
    if (ref_permeability == 1E-11):
        times_to_display = [0, 25, 50, 100, 300, 1000, 3000, 10000, 30000, 100000]        
    indices_to_display = find_indices(times_to_display, times)
    f, ax = plt.subplots()
    for i in indices_to_display:
        plt.plot(positions, pressure[:, i]/1E6, '-o', markeredgecolor = 
                 'black', label = '$t$ = {:.0f} s'.format(times[i]))
    plt.xlabel('Position, m')
    plt.ylabel('Pressure, MPa')
    plt.gca().set_xlim(left=0, right = positions[-1])
    plt.gca().set_ylim(bottom=0)
    plt.legend(fontsize = 12, ncol = 2, loc = 'upper right')
    plt.show()
    plt.savefig('./figures/pressure_{}_{}.pdf'.format(ref_permeability, 
                type_of_kernel))
    
    # Figure to compare concentration on the edge and in the coal matrix, to 
    # assess whether we are far or close to thermodynamic equilibrium
    indices_to_display = find_indices(times_to_display, times)
    plt.figure()
    for i in indices_to_display:
        plt.plot(positions, langmuir_isotherm(pressure[:, i]), '-o', 
                 markeredgecolor = 'black')
    plt.gca().set_prop_cycle(None)
    for i in indices_to_display:
        plt.plot(positions, adsorbed_amount[:, i] + langmuir_isotherm(
            pressure[-1,0]), '-',
            label = '$t$ = {:.0f} s'.format(times[i]))
    plt.xlabel('Position, m')
    plt.ylabel('Amount of fluid adsorbed on edge \n of cleat or in coal matrix, kg/m$^3$')
    plt.gca().set_xlim(left=0, right = positions[-1])
    plt.legend(fontsize = 12, ncol = 2, loc = 'upper right')
    plt.show()
    plt.savefig('./figures/concentrations_{}_{}.pdf'.format(ref_permeability, 
                type_of_kernel))
    
    # plot of permeability profile at various times
    if (ref_permeability == 1E-13):
        times_to_display = [0, 100, 300, 1000, 3000, 5000, 10000, 20000, 50000,
                            100000, 150000, 200000]
        times_to_display = [0, 100, 300, 1000, 3000, 5000, 10000, 20000, 50000,
                            80000, 110000, 140000, 170000, 200000]
    if (ref_permeability == 1E-12):
        times_to_display = [0, 25, 50, 100, 200, 500,
                            1000, 2000, 5000, 10000, 20000,
                            40000, 60000, 100000]
    if (ref_permeability == 1E-11):
        times_to_display = [0, 25, 50, 100, 200, 500,
                            1000, 2000, 5000, 10000, 20000,
                            40000, 100000]
    indices_to_display = find_indices(times_to_display, times)
    plt.figure()
    for i in indices_to_display:
        plt.plot(positions, permeability[:, i], '-o', markeredgecolor = 
                 'black', label = '$t$ = {:.0f} s'.format(times[i]))
    plt.xlabel('Position, m')
    plt.ylabel('Permeability, m$^2$')
    plt.gca().set_xlim(left=0, right = positions[-1])
    plt.legend(fontsize = 12, loc = 'upper right')
    plt.show()
    plt.savefig('./figures/permeability_{}_{}.pdf'.format(ref_permeability,
                type_of_kernel))