import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Constants
Cm = 1.0  #Membrane capacitance (uF/cm²)
g_Na = 120.0  #Sodium conductance (mS/cm²)
g_K = 36.0  #Potassium conductance (mS/cm²)
g_L = 0.3  #Leak conductance (mS/cm²)
E_Na = 50.0  #Sodium reversal potential (mV)
E_K = -77.0  #Potassium reversal potential (mV)
E_L = -54.4  #Leak reversal potential (mV)


g_syn_exc = 0.5  #(mS/cm²)
E_syn_exc = 0.0  #(mV)

g_syn_inh = 0.5  #(mS/cm²)
E_syn_inh = -70.0  #(mV)

# defining functions
def alpha_m(V): return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))
def beta_m(V): return 4 * np.exp(-(V + 65) / 18)

def alpha_h(V): return 0.07 * np.exp(-(V + 65) / 20)
def beta_h(V): return 1 / (1 + np.exp(-(V + 35) / 10))

def alpha_n(V): return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
def beta_n(V): return 0.125 * np.exp(-(V + 65) / 80)

def S_syn(t, t_peak=5, decay=0.5):
    return (t/t_peak) * np.exp(-t / decay) if t > 0 else 0

def HH_syn_model(y, t, I_ext, g_Na_mod, g_K_mod, g_syn, E_syn):
    V, m, h, n = y
    dmdt = alpha_m(V) * (1 - m) - beta_m(V) * m
    dhdt = alpha_h(V) * (1 - h) - beta_h(V) * h
    dndt = alpha_n(V) * (1 - n) - beta_n(V) * n

    
    syn_current = g_syn * (V - E_syn) * S_syn(t - 10)
    
    I_Na = g_Na_mod * (m**3) * h * (V - E_Na)
    I_K = g_K_mod * (n**4) * (V - E_K)
    I_L = g_L * (V - E_L)

    dVdt = (I_ext - (I_Na + I_K + I_L + syn_current)) / Cm
    return [dVdt, dmdt, dhdt, dndt]

# Simulation parameters
time = np.linspace(0, 50, 10000)
y0 = [-65, 0.05, 0.6, 0.32]
I_ext = 10  # External stimulus


sol_normal = odeint(HH_syn_model, y0, time, args=(I_ext, g_Na, g_K, g_syn_exc, E_syn_exc))  #Excitatory
sol_inhibitory = odeint(HH_syn_model, y0, time, args=(I_ext, g_Na, g_K, g_syn_inh, E_syn_inh))  #Inhibitory
sol_TTX = odeint(HH_syn_model, y0, time, args=(I_ext, 0, g_K, g_syn_exc, E_syn_exc))  #TTX

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(time, sol_normal[:, 0], label="Normal (Excitatory)", linewidth=1.5)
plt.plot(time, sol_inhibitory[:, 0], label="Inhibitory Synapse", linestyle="dotted")
plt.plot(time, sol_TTX[:, 0], label="TTX (Na+ Blocked)", linestyle="dashed")


plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (mV)")
plt.title("Effects of Synaptic and Neurotoxic Inputs on Hodgkin-Huxley Neuron")
plt.legend()
plt.show()
