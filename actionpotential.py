import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Constants
Cm = 1.0  
g_Na = 120.0  
g_K = 36.0  
g_L = 0.3  
E_Na = 50.0  
E_K = -77.0  
E_L = -54.4 

# Defining gating variables
def alpha_m(V): return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
def beta_m(V): return 4 * np.exp(-(V + 65) / 18)

def alpha_h(V): return 0.07 * np.exp(-(V + 65) / 20)
def beta_h(V): return 1 / (1 + np.exp(-(V + 35) / 10))

def alpha_n(V): return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
def beta_n(V): return 0.125 * np.exp(-(V + 65) / 80)

# HH model
def HH_model(y, t, I_ext):
    V, m, h, n = y
    dmdt = alpha_m(V) * (1 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1 - h) - beta_h(V) * h
    dndt = alpha_n(V) * (1 - n) - beta_n(V) * n
    
    I_Na = g_Na * (m**3) * h * (V - E_Na)
    I_K = g_K * (n**4) * (V - E_K)
    I_L = g_L * (V - E_L)
    
    dVdt = (I_ext - (I_Na + I_K + I_L)) / Cm
    return [dVdt, dmdt, dhdt, dndt]

# Simulation parameters
time = np.linspace(0, 50, 10000)
y0 = [-65, 0.05, 0.6, 0.32]
I_ext = 10  

solution = odeint(HH_model, y0, time, args=(I_ext,))

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(time, solution[:, 0], label="Membrane Potential (mV)", color='b')
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.title("Action Potential Simulation using Hodgkin-Huxley Model")
plt.legend()
plt.grid()
plt.show()