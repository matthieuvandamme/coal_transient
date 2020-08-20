# coding: utf-8
# author: matthieu VANDAMME
# Code associated to the manuscript "Modeling transient variations of permeability in coal seams at the scale of the reservoir" by N. Abouloifa, M. Vandamme, and P. Dangla

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
import time

start_time = time.time()
plt.close('all')

# Main parameters to modify
# Parameters of coal
REF_PERM_OF_CLEATS = 1E-18 # m2, ref. perm. of coal rock in unstressed state
# choose between 1E-19, 1E-18, 1E-17
# Parameters of interaction between fluid and coal
TYPE_OF_KERNEL = 'simulated' # choose between 'simulated' and 'fitted'

# Parameters of computation
NUM_OF_ELEMENTS = 50
NUM_OF_NODES = NUM_OF_ELEMENTS + 1
if (REF_PERM_OF_CLEATS == 1E-17):
    time_step = 10 # s
    NUM_OF_STEPS = 5000
    TYPE_OF_TIME_STEP = 'equispaced' # older version made it possible to use increasing time steps
if (REF_PERM_OF_CLEATS == 1E-18):
    time_step = 20 # s
    NUM_OF_STEPS = 5000
    TYPE_OF_TIME_STEP = 'equispaced'
if (REF_PERM_OF_CLEATS == 1E-19):
    time_step = 100.0/3.0 # s
    NUM_OF_STEPS = 10000
    TYPE_OF_TIME_STEP = 'equispaced'
LENGTH = 0.20 # m
DELTA_X = LENGTH / NUM_OF_ELEMENTS # m, size of element
TEMPERATURE = 313.15 # K
IMPOSED_PRESSURE = 10E6 # Pa, step of pressure imposed on first node
INITIAL_PRESSURE = 1E6 # Pa, initial pressure at all nodes
# to define times
if (TYPE_OF_TIME_STEP == 'equispaced'):
    times = np.arange(NUM_OF_STEPS + 1) * time_step # s, vector with all times

# Constants
IDEAL_GAS_CONSTANT = 8.314 # J/K/mol, ideal gas constant

# Parameters of fluid
MOLAR_MASS_OF_CO2 = 44E-3 # kg/mol
VISCOSITY_CO2 = 47.82E-6 # Pa.s. value from NIST for CO2 at 10 MPa

# Parameters of coal
# For us, the porosity associated to the cleat system is imposed by the
# geometry of the cleat. Calculated as pi * a * b / V_0
REF_POROSITY = np.pi*0.25*0.25*0.03 # 1, porosity associated to the cleat system
BULK_MODULUS_OF_COAL = 1620E6 # Pa
BULK_MODULUS_MATRIX = 5000E6 # Pa
K_C_P = 1.042E8 # Pa, defines by how much the cleat opens because of pressure

# Parameters of interaction between fluid and coal
C_M_MAX = 2.4E3 * MOLAR_MASS_OF_CO2 # kg/m3, Langmuir parameter that governs the maximum adsorbed amount
LANGMUIR_PRESSURE = 1.6E6 # Pa, Langmuir characteristic pressure
ALPHA = 0.004 * 1E-3 / MOLAR_MASS_OF_CO2 # 1/(kg/m3), coefficient that relates the adsorbed concentration to the swelling
DELTA_PHI_C_MAX = 1.39E-3 # 1, Char. max. transient variation of cleat porosity for fitted kernel
TAU_DIFF = 13502.2 # s, characteristic time of diffusion through coal matrix for fitted kernel
TAU_CLOSING = 503.4 # s, characterisctic time of closure of cleats
PROP_FACTOR_KERN_M_M_TO_KERN_DELTA_C = 2.306E-4 # m3/kg, part of the kernel J_delta_c that is proportional to J_m_m

# Parameters of visualization
# to display given number of times equally spaced
TIMES_TO_DISPLAY = np.arange(0, times[-1], int(times[-1] / 20))

# Nothing to be changed below this line

# kg/m3, returns density of CO2 as a function of pressure p
def density_co2(p):
    return MOLAR_MASS_OF_CO2 * p / (IDEAL_GAS_CONSTANT * TEMPERATURE)

# kg/m3/Pa, returns the derivative of the density with respect to pressure p
def diff_density_co2(p):
    return MOLAR_MASS_OF_CO2 / (IDEAL_GAS_CONSTANT * TEMPERATURE) * p / p

# kg/m3, returns the adsorbed amount per unit volume of coal matrix based on 
# pressure p, 
def langmuir_isotherm(p):
    return C_M_MAX * p / (p + LANGMUIR_PRESSURE)

# kg/m3/Pa, returns the derivative of the adsorbed amount per unit volume of coal matrix based on 
# pressure p, 
def diff_langmuir_isotherm(p):
    return C_M_MAX * LANGMUIR_PRESSURE / ((p + LANGMUIR_PRESSURE)**2)

# m2, returns the actual permeability given actual profile of pressure p
# and full history of increments of concentration dC  
def calculate_permeability(p, dC):
    delta_v_c_hydro_over_v_c_0 = np.convolve(kernel_delta_c, dC[0,:])[:dC.shape[1]]
    for i in range(1, NUM_OF_NODES):
        delta_v_c_hydro_over_v_c_0 = np.vstack((
                delta_v_c_hydro_over_v_c_0, np.convolve(
                        kernel_delta_c, dC[i,:])[:dC.shape[1]]))
    delta_v_c_hydro_over_v_c_0 = delta_v_c_hydro_over_v_c_0[:, -1]
    return REF_PERM_OF_CLEATS * np.power(1 + (p / K_C_P) +
                                         delta_v_c_hydro_over_v_c_0, 3)

# kg/m3, returns the actual adsorbed amount m_m per unit volume of porous
# solid, based on the whole history of pressure. Needs to be divided by (1-REF_POROSITY) to give value per unit
# volume of coal matrix
def calculate_m_m(p_up_to_p):
    p_init = np.ones((NUM_OF_NODES,1)) * INITIAL_PRESSURE
    p_up_to_p = np.hstack((p_init, p_up_to_p))
    c = langmuir_isotherm(p_up_to_p)
    dC = c - np.roll(c, 1, axis=1)
    dC = dC[:,1:]
    m = np.convolve(kernel_m_m, dC[0,:])[:dC.shape[1]] # kg/m3, per unit volume of fractured coal
    for i in range(1, NUM_OF_NODES):
        m = np.vstack((m, np.convolve(kernel_m_m, dC[i,:])[:dC.shape[1]]))
    return m[:,-1]

# kg/m3, returns the actual amount of fluid in cleats per unit volume of
# porous solid.
def calculate_m_c(p_up_to_t):
    p = p_up_to_t[:, -1]
    return density_co2(p) * REF_POROSITY

# kg/m3, returns the actual amount of fluid in both cleats and coal matrix, per
# unit volume of porous solid
def calculate_m(p):
    return calculate_m_c(p) + calculate_m_m(p) 

# m3/kg, returns fitted kernel_delta_c
def generate_fitted_kernel_delta_c(t):
    return PROP_FACTOR_KERN_M_M_TO_KERN_DELTA_C * generate_fitted_kernel_m_m(t) - (
            DELTA_PHI_C_MAX / C_M_MAX / REF_POROSITY 
            * np.exp(-t /TAU_DIFF) * (1 - np.exp(-t / TAU_CLOSING)))

# 1, returns fitted kernel m_m
def generate_fitted_kernel_m_m(t):
    return  (1 - REF_POROSITY) * (1 - np.exp(-t/TAU_DIFF))

# 1/s, returns fitted kernel m_dot_m
def generate_fitted_kernel_m_dot_m(t):
    return (1 - REF_POROSITY) * np.exp(-t / TAU_DIFF) / TAU_DIFF

# returns simulated kernel, read from Excel file f and interpolated on
# times t
def generate_simulated_kernel(t, file_name):
    k = np.zeros(len(t))
    data = np.genfromtxt(file_name, skip_header = 1, delimiter=";")
    simulated_times = data[:,0]
    simulated_kernel = data[:,1]
    # to make sure that kernel_m_dot_m converges toward zero
    if (simulated_kernel[-1] < 1E6):
        simulated_kernel[-1] = 0
    # to extend the kernels to make sure there is no problem with interpolation
    simulated_times = np.append(simulated_times, 1E9)
    simulated_kernel = np.append(simulated_kernel, simulated_kernel[-1])
    f = interp1d(simulated_times, simulated_kernel) 
    for n in range(len(t)):
        k[n] = f(t[n])
    return k 

def generate_simulated_kernel_delta_c(t, file_name):
    k = np.zeros(len(t))
    data = np.genfromtxt(file_name, skip_header = 1, delimiter=";")
    simulated_times = data[:,0]
    simulated_kernel = data[:,1]
    # extends kernel on very long times to have no issue to run long simulations
    simulated_times = np.append(simulated_times, 1e9)
    simulated_kernel = np.append(simulated_kernel, simulated_kernel[-1])
    f = interp1d(simulated_times, simulated_kernel) 
    for n in range(len(t)):
        k[n] = f(t[n])
    return k 

def generate_simulated_kernel_m_m(t, file_name):
    k = np.zeros(len(t))
    data = np.genfromtxt(file_name, skip_header = 1, delimiter=";")
    simulated_times = data[:,0]
    simulated_kernel = data[:,1]
    # rescales m_m to make sure that converges toward expected value
    simulated_kernel = simulated_kernel * (1 - REF_POROSITY) / simulated_kernel[-1]
    # extends kernel on very long times to have no issue to run long simulations
    simulated_times = np.append(simulated_times, 1e9)
    simulated_kernel = np.append(simulated_kernel, simulated_kernel[-1])
    f = interp1d(simulated_times, simulated_kernel) 
    for n in range(len(t)):
        k[n] = f(t[n])
    return k 

def generate_simulated_kernel_m_dot_m(t, file_name):
    k = np.zeros(len(t))
    data = np.genfromtxt(file_name, skip_header = 1, delimiter=";")
    simulated_times = data[:,0]
    simulated_kernel = data[:,1]
    # shifts kernel by 1 time step so that non-zero at the beginning and enforces zero at the end
    simulated_kernel = np.append(simulated_kernel[1:], 0)
    # extends kernel on very long times to have no issue to run long simulations
    simulated_times = np.append(simulated_times, 1e9)
    simulated_kernel = np.append(simulated_kernel, simulated_kernel[-1])
    f = interp1d(simulated_times, simulated_kernel) 
    for n in range(len(t)):
        k[n] = f(t[n])
    return k

def calculate_E(p_tDt, p_up_to_t):
    p_t = p_up_to_t[:,-1]
    p_up_to_tDt = np.hstack((p_up_to_t, np.vstack((p_tDt))))
    k = density_co2(p_t) * permeability_to_save[:,-1] / VISCOSITY_CO2
    E = np.zeros((len(p_tDt),1))
    E = DELTA_X * (calculate_m(p_up_to_tDt) - calculate_m(p_up_to_t))
    E[-1] = E[-1] / 2
    E = E + time_step / DELTA_X * ((k + np.roll(k, 1))/2 *(p_tDt - np.roll(p_tDt, 1)) +
                                   (k + np.roll(k, -1))/2 *(p_tDt - np.roll(p_tDt, -1)))
    E[-1] = E[-1] - time_step / DELTA_X * ((k[-1] + k[0])/2 *(p_tDt[-1] - p_tDt[0]))
    E[0] = p_tDt[0] - IMPOSED_PRESSURE
    return E

def calculate_jacobian(p_t, dummy): # the need to introduce a dummy variable
    # comes from the fact that the function E takes an argument that is not
    # needed to calculate the jacobian
    k = density_co2(pressure_to_save[:,-1]) * permeability_to_save[:,-1] / VISCOSITY_CO2
    a_diag = DELTA_X * (REF_POROSITY * diff_density_co2(p_t) + 
                        kernel_m_m[1]/2*diff_langmuir_isotherm(p_t)) + (
                                time_step / DELTA_X * ((2*k + np.roll(k, 1) + 
                                                        np.roll(k, -1))/2))
    A = np.diag(a_diag)
    A[0, 0] = 1
    A[-1, -1] = DELTA_X / 2 * (REF_POROSITY * diff_density_co2(p_t[-1]) + 
                        kernel_m_m[1]/2*diff_langmuir_isotherm(p_t[-1])) + (
                                time_step / DELTA_X * (k[-1] + k[-2]) / 2)
    for i in range(0, len(p_t)-1):
        A[i, i+1] = - time_step / DELTA_X * (k[i] + k[i+1])/2
        A[i+1, i] = - time_step / DELTA_X * (k[i] + k[i+1])/2
    A[0, 1] = 0
    return A

# The next function is to check the Jacobian
# Compare check_jacobian and calculate_jacobian of (pressure_to_save[:,-1], pressure_to_save)
def check_jacobian(p, p_up_to_t):
    delta_p = 1. # increment of pressure over which to calculate the gradient
    a = np.zeros((len(p), len(p)))
    for i in range(len(p)):
        for j in range(len(p)):
            p_plus = np.copy(p)
            p_plus[j] = p_plus[j] + delta_p
            a[i,j] = (calculate_E(p_plus, p_up_to_t)[i] - calculate_E(p, p_up_to_t)[i]) / delta_p
    return a
            
# To choose whether to use simulated or fitted kernel
if TYPE_OF_KERNEL == 'fitted':
    kernel_delta_c = generate_fitted_kernel_delta_c(times)
    kernel_m_m = generate_fitted_kernel_m_m(times)
    kernel_m_dot_m = generate_fitted_kernel_m_dot_m(times)
if TYPE_OF_KERNEL =='simulated':
    kernel_delta_c = generate_simulated_kernel_delta_c(
            times, './kernels/kernel_delta_c_0_03.csv')
    kernel_m_m = generate_simulated_kernel_m_m(
            times, './kernels/kernel_m_m_0_03.csv')
    kernel_m_dot_m = generate_simulated_kernel_m_dot_m(
            times, './kernels/kernel_m_dot_m_0_03.csv')

# Very calculations

X = np.linspace(0., LENGTH, NUM_OF_NODES)
pressure = np.ones((NUM_OF_NODES,1)) * INITIAL_PRESSURE
pressure[0,0] = IMPOSED_PRESSURE
pressure_to_save = pressure # Pa, to initialize array that contains all pressure profiles to save
c_edge = langmuir_isotherm(pressure) # kg/m3, profile of concentrations along the edge of the fracture
c_edge_to_save = c_edge # kg/m3, to initialize array that contains all c_edge profiles to save
dc_edge = np.zeros((NUM_OF_NODES,1)) # kg/m3, profile of increments of concentrations along the edge of the fracture
m_m = np.zeros((NUM_OF_NODES,1)) # kg/m3, profile of adsorbed amounts per unit volume of porous solid
m_m_to_save = m_m # kg/m3, to initialize array that contains all m_m profiles to save
# We consider that the initial system was fully equilibrated at the initial
# pressure
dc_edge[0,0] = (langmuir_isotherm(IMPOSED_PRESSURE) -
       langmuir_isotherm(INITIAL_PRESSURE))
permeability = np.vstack(calculate_permeability(
        pressure[:,-1], np.zeros((NUM_OF_NODES,1)))) # m2, profile of permeability along the edge of the fracture
permeability_to_save = permeability # m2, to initialize array that contains all permeability profiles to save
times_to_save = [times[0]] # to store all times to be saved

for n in range(NUM_OF_STEPS):
    print(n)
    time_index = n + 1 # index of the time to which corresponds the calculation in the loop
    new_pressure = fsolve(calculate_E, pressure_to_save[:,-1], xtol= 1E-10,
                          fprime = calculate_jacobian, args = pressure_to_save)
    new_c_edge = langmuir_isotherm(new_pressure)
    new_dc_edge = new_c_edge - c_edge[:,-1]
    pressure = np.hstack((pressure, np.vstack(new_pressure)))
    pressure_to_save = np.hstack((pressure_to_save, np.vstack(new_pressure)))
    c_edge = np.hstack((c_edge, np.vstack(new_c_edge)))
    c_edge_to_save = np.hstack((c_edge_to_save, np.vstack(new_c_edge)))
    dc_edge = np.hstack((dc_edge, np.vstack(new_dc_edge)))
    new_m_m = calculate_m_m(pressure_to_save)
    m_m = np.hstack((m_m, np.vstack(new_m_m)))
    m_m_to_save = np.hstack((m_m_to_save, np.vstack(new_m_m)))
    new_permeability = calculate_permeability(pressure[:, -1], dc_edge)
    permeability = np.hstack((permeability, np.vstack(new_permeability)))
    permeability_to_save = np.hstack((permeability_to_save, np.vstack(new_permeability)))
    times_to_save = np.append(times_to_save, times[time_index])
    
# to save the data
    
if TYPE_OF_KERNEL == 'fitted':
    kernel_delta_c = generate_fitted_kernel_delta_c(times_to_save)
    kernel_m_m = generate_fitted_kernel_m_m(times_to_save)
    kernel_m_dot_m = generate_fitted_kernel_m_dot_m(times_to_save)
if TYPE_OF_KERNEL =='simulated':
    kernel_delta_c = generate_simulated_kernel_delta_c(
            times, './kernels/kernel_delta_c_0_03.csv')
    kernel_m_m = generate_simulated_kernel_m_m(
            times, './kernels/kernel_m_m_0_03.csv')
    kernel_m_dot_m = generate_simulated_kernel_m_dot_m(
            times, './kernels/kernel_m_dot_m_0_03.csv')

np.savez('perm1E-{:.0f}_{}_{}'.format(-np.log10(REF_PERM_OF_CLEATS),
         TYPE_OF_KERNEL, TYPE_OF_TIME_STEP), pressure = pressure_to_save,
    permeability = permeability_to_save,
    c_edge = c_edge_to_save, adsorbed_amount = m_m_to_save / (1-REF_POROSITY),
     ref_permeability = REF_PERM_OF_CLEATS,
    type_of_kernel = TYPE_OF_KERNEL, times = times_to_save, positions = X,
    kernel_m_dot_m = kernel_m_dot_m, kernel_m_m = kernel_m_m,
    kernel_delta_c = kernel_delta_c)
# m_m is divided by (1-REF_POROSITY) to express the adsorbed amount per unit
# volume of coal matrix
   
# Vizualization

# to find out all the indices of the times to display
indices_to_display = []
for t in TIMES_TO_DISPLAY:
    indices_to_display.append(np.abs(times - t).argmin())

plt.figure()

# plot of kernels
plt.subplot(2, 3, 1)
plt.plot(times_to_save, kernel_delta_c, '-o', markeredgecolor = 'black')
plt.xlabel('Time, s')
plt.ylabel('Kernel $J_c$, m$^3$/kg')
plt.gca().set_xlim(left=0)
plt.show()

plt.subplot(2, 3, 2)
plt.plot(times_to_save, kernel_m_m, '-o', markeredgecolor = 'black')
plt.xlabel('Time, s')
plt.ylabel('Kernel $J_{m_m}$, 1')
plt.gca().set_xlim(left=0)
plt.gca().set_ylim(bottom=0)
plt.show()

plt.subplot(2, 3, 3)
plt.plot(times_to_save, kernel_m_dot_m, '-o', markeredgecolor = 'black')
plt.xlabel('Time, s')
plt.ylabel('Kernel $J_{\dot{m}_m}$, s$^{-1}$')
plt.gca().set_xlim(left=0)
plt.gca().set_ylim(bottom=0)
plt.show()

# plot of pressure profile at various times
plt.subplot(2, 3, 4)
for i in indices_to_display:
    plt.plot(X*1E3, pressure[:, i]/1E6, '-o', markeredgecolor = 'black',
             label = '$t$ = {:.0f} s'.format(times[i]))
plt.xlabel('Distance, mm')
plt.ylabel('Pressure, MPa')
plt.gca().set_xlim(left=0, right = X[-1]*1E3)
plt.gca().set_ylim(bottom=0)
plt.legend(fontsize = 5, ncol = 2)
plt.show()

# plot of evolutions of pressure over time at various places
plt.subplot(2, 3, 5)
for i in range(pressure_to_save.shape[0]):
    plt.plot(times_to_save, pressure_to_save[i, :]/1E6, '-', markeredgecolor = 'black')
plt.xlabel('Time, s')
plt.ylabel('Pressure, MPa')
plt.gca().set_xlim(left=0)
plt.gca().set_ylim(bottom=0)
plt.show()

# plot of permeability profile at various times
plt.subplot(2, 3, 6)
for i in indices_to_display:
    plt.plot(X*1E3, permeability[:, i], '-o', markeredgecolor = 'black',
             label = '$t$ = {:.0f} s'.format(times[i]))
plt.xlabel('Distance, mm')
plt.ylabel('Permeability, m$^2$')
plt.gca().set_xlim(left=0, right = X[-1]*1E3)
plt.legend(fontsize = 5, ncol = 2)
plt.show()

plt.figure()

# plot of evolutions of permeability over time at various places
plt.subplot(2, 3, 1)
for i in range(permeability_to_save.shape[0]):
    plt.plot(times_to_save, permeability_to_save[i, :], '-', markeredgecolor = 'black')
plt.xlabel('Time, s')
plt.ylabel('Permeability, m$^2$')
plt.gca().set_xlim(left=0)
plt.gca().set_ylim(bottom=0)
plt.show()

# Calculation of rate of average adsorbed amount in coal matrix, per unit of
# coal matrix
adsorption_rate = np.convolve(kernel_m_dot_m, dc_edge[0,:])[:(
        NUM_OF_STEPS+1)]/(1-REF_POROSITY)
for i in range(1, NUM_OF_NODES):
    adsorption_rate = np.vstack((adsorption_rate, 
                                 np.convolve(kernel_m_dot_m, dc_edge[i,:])
                                 [:(NUM_OF_STEPS+1)]/(1-REF_POROSITY)))

# plot of rate of average adsorbed amount in coal matrix, per unit of coal
# matrix
plt.subplot(2, 3, 2)
for i in indices_to_display:
    plt.plot(X*1E3, adsorption_rate[:, i], '-o', markeredgecolor = 'black',
             label = '$t$ = {:.0f} s'.format(times[i]))
plt.xlabel('Distance, mm')
plt.ylabel('Rate adsorbed per unit \n volume of coal matrix, kg/m$^3$/s')
plt.gca().set_xlim(left=0, right = X[-1]*1E3)
plt.gca().set_ylim(bottom=0)
plt.legend(fontsize = 5, ncol = 2)
plt.show()

# Figure with average adsorbed amount in coal matrix, per unit of coal matrix.
adsorbed_amount = m_m / (1-REF_POROSITY)

plt.subplot(2, 3, 3)
for i in indices_to_display:
    plt.plot(X*1E3, adsorbed_amount[:, i], '-o', markeredgecolor = 'black',
             label = '$t$ = {:.0f} s'.format(times[i]))
plt.xlabel('Distance, mm')
plt.ylabel('Amount adsorbed per unit \n volume of coal matrix, kg/m$^3$')
plt.gca().set_xlim(left=0, right = X[-1]*1E3)
plt.gca().set_ylim(bottom=0)
plt.legend(fontsize = 5, ncol = 2)
plt.show()

# For verification purposes, calculation of average adsorbed amount in coal
# matrix, per unit of coal matrix by integration over time of the small
# increments of rate of adsorbed amount.
if (TYPE_OF_TIME_STEP == 'equispaced'):
    adsorbed_amount_from_diff = np.zeros((NUM_OF_NODES, 1))
    for i in range(NUM_OF_STEPS):
        adsorbed_amount_from_diff = np.hstack((adsorbed_amount_from_diff, 
                                               np.vstack(adsorbed_amount_from_diff[:,-1] + 
                                                         adsorption_rate[:,i] * time_step)))

    plt.subplot(2, 3, 4)
    for i in indices_to_display:
        plt.plot(X*1E3, adsorbed_amount_from_diff[:, i], '-o',
                 markeredgecolor = 'black',
                 label = '$t$ = {:.0f} s'.format(times[i]))
    plt.xlabel('Distance, mm')
    plt.ylabel('Amount adsorbed per unit \n volume of coal matrix, kg/m$^3$')
    plt.gca().set_xlim(left=0, right = X[-1]*1E3)
    plt.gca().set_ylim(bottom=0)
    plt.legend(fontsize = 5, ncol = 2)
    plt.show()

# Figure to compare concentration on the edge and in the coal matrix, to 
# assess whether we are far or close to thermodynamic equilibrium
plt.subplot(2, 3, 5)
for i in indices_to_display:
    plt.plot(X*1E3, adsorbed_amount[:, i] + langmuir_isotherm(
            INITIAL_PRESSURE), '-',
        label = 'in matrix at $t$ = {:.0f} s'.format(times[i]))
plt.gca().set_prop_cycle(None)
for i in indices_to_display:
#    plt.plot(X*1E3, langmuir_isotherm(pressure[:, i]), '--',
#             label = 'on cleat edge at $t$ = {:.0f} s'.format(times[i]))
    plt.plot(X*1E3, langmuir_isotherm(pressure[:, i]), '--')
plt.xlabel('Distance, mm')
plt.ylabel('Amount adsorbed on edge of cleat \n or in coal matrix, kg/m$^3$')
plt.gca().set_xlim(left=0, right = X[-1]*1E3)
plt.gca().set_ylim(bottom=0)
plt.legend(fontsize = 5, ncol = 2)
plt.show()

print('Time complete calculation: {} s'.format(int(time.time() - start_time)))

# Verification: long term value of the average adsorbed amount in coal matrix,
# per unit of coal matrix, should be equal to following expression
#langmuir_isotherm(IMPOSED_PRESSURE) - langmuir_isotherm(INITIAL_PRESSURE)