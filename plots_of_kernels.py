# coding: utf-8
# date: started on 17 April 2020
# author: matthieu VANDAMME
# Code associated to the manuscript "Modeling transient variations of permeability in coal seams at the scale of the reservoir" by N. Abouloifa, M. Vandamme, and P. Dangla

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize

plt.close('all')

MOLAR_MASS_OF_CO2 = 44E-3 # kg/mol
C_M_MAX = 2.4E3 * MOLAR_MASS_OF_CO2 # kg/m3, Langmuir parameter that governs the maximum adsorbed amount

# 1, returns fitted kernel m_m
def generate_fitted_kernel_m_m(t, por, tau_diff):
    return  (1 - por) * (1 - np.exp(-t / tau_diff))

# 1/s, returns fitted kernel m_dot_m
def generate_fitted_kernel_m_dot_m(t, por, tau_diff):
    return (1 - por) * np.exp(-t / tau_diff) / tau_diff

# m3/kg, returns fitted kernel_delta_c
def generate_fitted_kernel_delta_c(t, prop_factor_hydr_m_m, delta_phi, por, 
                                tau_diff, tau_clos):
    return prop_factor_hydr_m_m * generate_fitted_kernel_m_m(
            t, por, tau_diff) - (delta_phi / C_M_MAX / por * np.exp(-t / 
                            tau_diff) * (1 - np.exp(-t / tau_clos)))
    
def func_to_minimize_to_fit_kernel_m_m(x, times, por, data):
    tau = x[0]
    a = data - generate_fitted_kernel_m_m(times, por, tau)
    return np.mean(np.power(a, 2))

def func_to_minimize_to_fit_kernel_delta_c(x, t, prop_factor_hydr_m_m, por, 
                                tau_diff, data):
    delta_phi = x[0] / 1000
    tau_clos = x[1] * 100
    a = data - generate_fitted_kernel_delta_c(t, prop_factor_hydr_m_m, 
                                              delta_phi, por, 
                                              tau_diff, tau_clos)
    return np.mean(np.power(a, 2))
    
# returns simulated kernel, read from Excel file f and interpolated on
# times t
def generate_simulated_kernel(t, file_name):
    k = np.zeros(len(t))
    data = np.genfromtxt(file_name, skip_header = 1, delimiter=";")
    simulated_times = data[:,0]
    simulated_kernel = data[:,1]
    f = interp1d(simulated_times, simulated_kernel) 
    for n in range(len(t)):
        k[n] = f(t[n])
    return k

def funct(x, y, z):
    return np.power(x-y, 2)

class Kernels:
    def __init__(self, name, por, path_hydr, path_m_m, path_m_dot_m):
        a = np.genfromtxt(path_hydr, skip_header = 1, delimiter=";")
        b = np.genfromtxt(path_m_m, skip_header = 1, delimiter=";")
        c = np.genfromtxt(path_m_dot_m, skip_header = 1, delimiter=";")
        self.name = name
        self.times = a[:,0]
        self.kernel_delta_c_simulated = a[:,1]
        self.kernel_m_m_simulated = b[:,1]
        self.kernel_m_dot_m_simulated = c[:,1]
        self.porosity = por
        # to ensure that kernel_m_m_simulated converges toward (1 - porosity)
        self.rescaling_m_m = (1 - 
                              self.porosity) / self.kernel_m_m_simulated[-1]
        self.kernel_m_m_simulated = (self.kernel_m_m_simulated * 
                                     self.rescaling_m_m)
        self.kernel_m_dot_m_simulated = (self.kernel_m_dot_m_simulated *
                                         self.rescaling_m_m)
        self.prop_factor_m_m_to_delta_c = (self.kernel_delta_c_simulated[-1] / 
                                   self.kernel_m_m_simulated[-1])
        res = minimize(func_to_minimize_to_fit_kernel_m_m, 10000, tol = 1e-8,
                       args=(self.times, self.porosity, 
                             self.kernel_m_m_simulated))
        self.tau_diff = res.x[0]
        self.kernel_m_m_fitted = generate_fitted_kernel_m_m(
                self.times, self.porosity, self.tau_diff)
        self.kernel_m_dot_m_fitted = generate_fitted_kernel_m_dot_m(
                self.times, self.porosity, self.tau_diff)
        res2 = minimize(func_to_minimize_to_fit_kernel_delta_c, (1, 2),
                        tol = 1e-12,
                       args=(self.times, self.prop_factor_m_m_to_delta_c,
                             self.porosity, self.tau_diff, 
                             self.kernel_delta_c_simulated))
        self.delta_phi_c_max = res2.x[0] / 1000
        self.tau_closing = res2.x[1] * 100
        self.kernel_delta_c_fitted = generate_fitted_kernel_delta_c(
                self.times, self.prop_factor_m_m_to_delta_c,
                self.delta_phi_c_max, 
                self.porosity, self.tau_diff, self.tau_closing)
        
kernels_0_03 = Kernels('ellipse with a/A = 0.03', np.pi*0.25*0.25*0.03, 
                       './kernels/kernel_delta_c_0_03.csv', 
                       './kernels/kernel_m_m_0_03.csv', 
                       './kernels/kernel_m_dot_m_0_03.csv')

kernels_0_1 = Kernels('ellipse with a/A = 0.1', np.pi*0.25*0.25*0.1,
                      './kernels/kernel_delta_c_0_1.csv', 
                       './kernels/kernel_m_m_0_1.csv', 
                       './kernels/kernel_m_dot_m_0_1.csv')

kernels_0_3 = Kernels('ellipse with a/A = 0.3', np.pi*0.25*0.25*0.3, 
                      './kernels/kernel_delta_c_0_3.csv', 
                       './kernels/kernel_m_m_0_3.csv', 
                       './kernels/kernel_m_dot_m_0_3.csv')

kernels_1 = Kernels('ellipse with a/A = 1', np.pi*0.25*0.25, 
                    './kernels/kernel_delta_c_1.csv', 
                       './kernels/kernel_m_m_1.csv', 
                       './kernels/kernel_m_dot_m_1.csv')

kernels_rect = Kernels('rectangle', 0.5*0.05,
                       './kernels/kernel_delta_c_rect.csv', 
                       './kernels/kernel_m_m_rect.csv', 
                       './kernels/kernel_m_dot_m_rect.csv')

for i in [kernels_0_03, kernels_0_1, kernels_0_3, kernels_1, kernels_rect]:
    print ('{}: tau_diff = {:.2f}'.format(i.name, i.tau_diff))
    print ('{}: tau_clsoing = {:.2f}'.format(i.name, i.tau_closing))
    print ('{}: delta_phi_c_max = {}'.format(i.name, i.delta_phi_c_max))

plt.figure()
plt.plot(kernels_0_03.times, kernels_0_03.kernel_delta_c_simulated, '-',
         label = 'ellipse with $a/A = 0.03$, \n simulated', markeredgecolor = 'black')
plt.plot(kernels_0_1.times, kernels_0_1.kernel_delta_c_simulated, '-',
         label = 'ellipse with $a/A = 0.1$, \n simulated', markeredgecolor = 'black')
plt.plot(kernels_0_3.times, kernels_0_3.kernel_delta_c_simulated, '-',
         label = 'ellipse with $a/A = 0.3$, \n simulated', markeredgecolor = 'black')
plt.plot(kernels_1.times, kernels_1.kernel_delta_c_simulated, '-',
         label = 'ellipse with $a/A = 1$, \n simulated', markeredgecolor = 'black')
plt.plot(kernels_rect.times, kernels_rect.kernel_delta_c_simulated, '-',
         label = 'rectangle, simulated', markeredgecolor = 'black')
plt.gca().set_prop_cycle(None)
plt.plot(kernels_0_03.times, kernels_0_03.kernel_delta_c_fitted, '--',
         label = 'ellipse with $a/A = 0.03$, \n fitted', markeredgecolor = 'black')
plt.plot(kernels_0_1.times, kernels_0_1.kernel_delta_c_fitted, '--',
         label = 'ellipse with $a/A = 0.1$, fitted', markeredgecolor = 'black')
plt.plot(kernels_0_3.times, kernels_0_3.kernel_delta_c_fitted, '--',
         label = 'ellipse with $a/A = 0.3$, fitted', markeredgecolor = 'black')
plt.plot(kernels_1.times, kernels_1.kernel_delta_c_fitted, '--',
         label = 'ellipse with $a/A = 1$, fitted', markeredgecolor = 'black')
plt.plot(kernels_rect.times, kernels_rect.kernel_delta_c_fitted, '--',
         label = 'rectangle, fitted', markeredgecolor = 'black')
plt.xlabel('Time, s')
plt.ylabel('Kernel $J_{\delta_c}$, m$^3$.kg$^{-1}$')
plt.legend(fontsize = 12, loc = 'lower right')
plt.xscale('log')
plt.gca().set_xlim(left=25, right=1e7)
plt.show()
plt.savefig('./figures/kernels_delta_c.pdf')

plt.figure()
plt.plot(kernels_0_03.times, kernels_0_03.kernel_m_m_simulated, '-',
         label = 'ellipse with $a/A = 0.03$, simulated', markeredgecolor = 'black')
plt.plot(kernels_0_1.times, kernels_0_1.kernel_m_m_simulated, '-',
         label = 'ellipse with $a/A = 0.1$, simulated', markeredgecolor = 'black')
plt.plot(kernels_0_3.times, kernels_0_3.kernel_m_m_simulated, '-',
         label = 'ellipse with $a/A = 0.3$, simulated', markeredgecolor = 'black')
plt.plot(kernels_1.times, kernels_1.kernel_m_m_simulated, '-',
         label = 'ellipse with $a/A = 1$, simulated', markeredgecolor = 'black')
plt.plot(kernels_rect.times, kernels_rect.kernel_m_m_simulated, '-',
         label = 'rectangle, simulated', markeredgecolor = 'black')
plt.gca().set_prop_cycle(None)
plt.plot(kernels_0_03.times, kernels_0_03.kernel_m_m_fitted, '--',
         label = 'ellipse with $a/A = 0.03$, fitted', markeredgecolor = 'black')
plt.plot(kernels_0_1.times, kernels_0_1.kernel_m_m_fitted, '--',
         label = 'ellipse with $a/A = 0.1$, fitted', markeredgecolor = 'black')
plt.plot(kernels_0_3.times, kernels_0_3.kernel_m_m_fitted, '--',
         label = 'ellipse with $a/A = 0.3$, fitted', markeredgecolor = 'black')
plt.plot(kernels_1.times, kernels_1.kernel_m_m_fitted, '--',
         label = 'ellipse with $a/A = 1$, fitted', markeredgecolor = 'black')
plt.plot(kernels_rect.times, kernels_rect.kernel_m_m_fitted, '--',
         label = 'rectangle, fitted', markeredgecolor = 'black')
plt.xlabel('Time, s')
plt.ylabel('Kernel $J_{m_m}$, 1')
plt.legend(fontsize = 12)
plt.gca().set_xlim(left=0, right=120000)
plt.gca().set_ylim(bottom=0)
plt.show()
plt.savefig('./figures/kernels_m_m.pdf')

plt.figure()
plt.plot(kernels_0_03.times[1:], kernels_0_03.kernel_m_dot_m_simulated[1:], '-',
         label = 'ellipse with $a/A = 0.03$, simulated', markeredgecolor = 'black')
plt.plot(kernels_0_1.times[1:], kernels_0_1.kernel_m_dot_m_simulated[1:], '-',
         label = 'ellipse with $a/A = 0.1$, simulated', markeredgecolor = 'black')
plt.plot(kernels_0_3.times[1:], kernels_0_3.kernel_m_dot_m_simulated[1:], '-',
         label = 'ellipse with $a/A = 0.3$, simulated', markeredgecolor = 'black')
plt.plot(kernels_1.times[1:], kernels_1.kernel_m_dot_m_simulated[1:], '-',
         label = 'ellipse with $a/A = 1$, simulated', markeredgecolor = 'black')
plt.plot(kernels_rect.times[1:], kernels_rect.kernel_m_dot_m_simulated[1:], '-',
         label = 'rectangle, simulated', markeredgecolor = 'black')
plt.gca().set_prop_cycle(None)
plt.plot(kernels_0_03.times, kernels_0_03.kernel_m_dot_m_fitted, '--',
         label = 'ellipse with $a/A = 0.03$, fitted', markeredgecolor = 'black')
plt.plot(kernels_0_1.times, kernels_0_1.kernel_m_dot_m_fitted, '--',
         label = 'ellipse with $a/A = 0.1$, fitted', markeredgecolor = 'black')
plt.plot(kernels_0_3.times, kernels_0_3.kernel_m_dot_m_fitted, '--',
         label = 'ellipse with $a/A = 0.3$, fitted', markeredgecolor = 'black')
plt.plot(kernels_1.times, kernels_1.kernel_m_dot_m_fitted, '--',
         label = 'ellipse with $a/A = 1$, fitted', markeredgecolor = 'black')
plt.plot(kernels_rect.times, kernels_rect.kernel_m_dot_m_fitted, '--',
         label = 'rectangle, fitted', markeredgecolor = 'black')
plt.xlabel('Time, s')
plt.ylabel('Kernel $J_{\dot{m}_m}$, s$^{-1}$')
plt.legend(fontsize = 12)
plt.gca().set_xlim(left=0, right=60000)
plt.gca().set_ylim(bottom=0, top=0.0001)
plt.show()
plt.savefig('./figures/kernels_m_dot_m.pdf')

plt.figure()
plt.plot(kernels_0_03.times, kernels_0_03.kernel_delta_c_simulated, '-',
         label = 'ellipse with $a/A = 0.03$', markeredgecolor = 'black')
plt.plot(kernels_0_1.times, kernels_0_1.kernel_delta_c_simulated, '-',
         label = 'ellipse with $a/A = 0.1$', markeredgecolor = 'black')
plt.plot(kernels_0_3.times, kernels_0_3.kernel_delta_c_simulated, '-',
         label = 'ellipse with $a/A = 0.3$', markeredgecolor = 'black')
plt.plot(kernels_1.times, kernels_1.kernel_delta_c_simulated, '-',
         label = 'ellipse with $a/A = 1$', markeredgecolor = 'black')
plt.plot(kernels_rect.times, kernels_rect.kernel_delta_c_simulated, '-',
         label = 'rectangle', markeredgecolor = 'black')
plt.xlabel('Time, s')
plt.ylabel('Kernel $J_{\delta_c}$, m$^3$.kg$^{-1}$')
plt.legend(fontsize = 12, loc = 'lower right')
plt.xscale('log')
plt.gca().set_xlim(left=25, right=1e7)
plt.show()
plt.savefig('./figures/kernels_delta_c_simulated.pdf')

plt.figure()
plt.plot(kernels_0_03.times, kernels_0_03.kernel_m_m_simulated, '-',
         label = 'ellipse with $a/A = 0.03$', markeredgecolor = 'black')
plt.plot(kernels_0_1.times, kernels_0_1.kernel_m_m_simulated, '-',
         label = 'ellipse with $a/A = 0.1$', markeredgecolor = 'black')
plt.plot(kernels_0_3.times, kernels_0_3.kernel_m_m_simulated, '-',
         label = 'ellipse with $a/A = 0.3$', markeredgecolor = 'black')
plt.plot(kernels_1.times, kernels_1.kernel_m_m_simulated, '-',
         label = 'ellipse with $a/A = 1$', markeredgecolor = 'black')
plt.plot(kernels_rect.times, kernels_rect.kernel_m_m_simulated, '-',
         label = 'rectangle', markeredgecolor = 'black')
plt.xlabel('Time, s')
plt.ylabel('Kernel $J_{m_m}$, 1')
plt.legend(fontsize = 12)
plt.gca().set_xlim(left=0, right=120000)
plt.gca().set_ylim(bottom=0)
plt.show()
plt.savefig('./figures/kernels_m_m_simulated.pdf')

plt.figure()
plt.plot(kernels_0_03.times[1:], kernels_0_03.kernel_m_dot_m_simulated[1:], '-',
         label = 'ellipse with $a/A = 0.03$', markeredgecolor = 'black')
plt.plot(kernels_0_1.times[1:], kernels_0_1.kernel_m_dot_m_simulated[1:], '-',
         label = 'ellipse with $a/A = 0.1$', markeredgecolor = 'black')
plt.plot(kernels_0_3.times[1:], kernels_0_3.kernel_m_dot_m_simulated[1:], '-',
         label = 'ellipse with $a/A = 0.3$', markeredgecolor = 'black')
plt.plot(kernels_1.times[1:], kernels_1.kernel_m_dot_m_simulated[1:], '-',
         label = 'ellipse with $a/A = 1$', markeredgecolor = 'black')
plt.plot(kernels_rect.times[1:], kernels_rect.kernel_m_dot_m_simulated[1:], '-',
         label = 'rectangle', markeredgecolor = 'black')
plt.xlabel('Time, s')
plt.ylabel('Kernel $J_{\dot{m}_m}$, s$^{-1}$')
plt.legend(fontsize = 12)
plt.gca().set_xlim(left=0, right=60000)
plt.gca().set_ylim(bottom=0, top=0.0001)
plt.show()
plt.savefig('./figures/kernels_m_dot_m_simulated.pdf')