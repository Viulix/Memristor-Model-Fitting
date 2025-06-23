# models.py
"""
## Models
A collection of analytical models describing various electronic conduction mechanisms relevant to memristive and semiconductor devices. 
Each function implements a specific physical model and returns the current density J (in A/m²) as a function of the electric field and material parameters. 
The module includes models for Schottky emission, Fowler–Nordheim tunneling, direct tunneling, ohmic conduction, Poole–Frenkel effect, space-charge-limited conduction (SCLC), ionic conduction, nearest-neighbor hopping, variable-range hopping, and trap-assisted tunneling, among others.
Physical constants are sourced from scipy.constants or defined explicitly. 
A dictionary `models` provides convenient access to each model, its parameters (with suggested initial values and bounds), and a LaTeX representation for documentation or plotting.
Note:
- The models are simplified and may not capture all physical effects present in real devices.
- Parameter ranges and forms may require adjustment for specific materials or experimental setups.
- Intended for use in fitting experimental data and educational purposes.
### Author: Nira
"""

import numpy as np
from scipy import constants

# Fundamental constants
q   = constants.elementary_charge      # Electron charge (C)
kB  = constants.Boltzmann              # Boltzmann constant (J/K)
h   = constants.h                      # Planck constant (J·s)
hbar = constants.hbar                  # Reduced Planck constant (J·s)
eps0  = constants.epsilon_0              # Vacuum permittivity (F/m)
π   = np.pi
T = 300 # may be adjusted

zeroBuffer = 1e-12
maxExponent = 200

def J_schottky(E, A, phi_B, epsilon_r):
    # https://de.wikipedia.org/wiki/Edison-Richardson-Effekt
    phi_B_J = phi_B * q
    exponent = - (q * (phi_B_J - np.sqrt(q*np.abs(E) / (4*np.pi * epsilon_r * eps0)))) / (kB * T)
    exponent = np.where(exponent > 200, maxExponent, exponent)  # Avoid overflow
    J = A * T**2 * np.exp(exponent)
    return J


def J_fowler_nordheim(E, K_1, K_2, phi_B):
    # https://de.wikipedia.org/wiki/Feldemission
    phi_B_J = phi_B * q
    J = K_1 * (E**2 / phi_B_J) * np.exp(-K_2 * (phi_B_J**1.5) / np.abs(E))
    return J

#### Fehlerhaft. Bedarf Korrektur. (Es passiert kein Fit, sondern verbleibt konstant -> falsch!)
# Keine Abhängigkeit von E ?? -> soll das Proportional zu E sein? (Annahme: Lineare Beziehung)
def J_direct_tunneling(E, m_eff, phi_B, kappa, t_ox):
    phi_B_J = phi_B * q
    exponent = - (8 * π * np.sqrt(2 * q * np.abs(m_eff) * phi_B_J)) / (3 * h) * kappa * t_ox
    J = E*np.exp(exponent)
    return J

### Korrigiert, aber ungenau (?) -> wie ein Dauerhafter Offset. ggf mit Widerstand abgleichen
def J_ohmic(E, mu, N_C, E_C, E_F):
    exponent = - (E_C - E_F) / (kB * T)
    exponent = np.where(exponent > 200, maxExponent, exponent)
    J = q * mu * N_C * E * np.exp(exponent)
    return J

### Hier vers. Formeln gefunden. Eine mit E und eine mit V. Implementiert ist die mit E.
def J_poole_frenkel(E, mu, N_C, phi_T, epsilon_r):
    # https://en.wikipedia.org/wiki/Poole%E2%80%93Frenkel_effect
    phi_T_J = phi_T * q
    epsilon = eps0 * epsilon_r

    exponent = -q * (phi_T_J - np.sqrt(q * np.abs(E) / (π * epsilon))) / (kB * T)
    
    exponent = np.where(exponent > 200, maxExponent, exponent)
    J = q * mu * N_C * E * np.exp(exponent)
    return J

# Hier wird V statt E verwendet (?) -> Warum oder gibt es einen Zusammenhang?
def J_space_charge_limited(E, mu, epsilon_r, theta, d):
    # https://en.wikipedia.org/wiki/Space_charge
    epsilon = eps0 * epsilon_r
    J = (9.0 / 8.0) * epsilon * mu * theta * (E**2 / d)
    return J

### Das sigma_0 ist nicht in der Formel enthalten, sondern wird als Vorfaktor verwendet (da lediglich proportional zu E)
def J_ionic(E, dG, sigma_0):
    exponent = - dG / (kB * T)
    return sigma_0 * (E / T) * np.exp(exponent)

def J_nearest_neighbor_hopping(E, sigma_0, T0):
    exponent = - T0 / T
    return sigma_0 * np.exp(exponent) * E

def J_variable_range_hopping(E, sigma_0, T0):
    if T0 < 0: T0 = 0
    exponent = - (T0 / T)**0.25
    return sigma_0 * np.exp(exponent) * E

def J_trap_assisted_tunneling(E, A, m_eff, phi_T):
    phi_T_J = phi_T * q  
    exponent = 8 * np.pi * np.sqrt(2 * q * np.abs(m_eff)) * (np.abs(phi_T_J))**1.5 / (3 * h * E) 
    J = A * np.exp(exponent)
    return J

def linear_test(E, m, b):
    return m * E + b

# Dictionary of models for fitting
# Each parameter is now a dict with keys: 'init', 'min', 'max'
models = {
    'Schottky emission': {
        'func': J_schottky,
        'params': {
            'A':   {'init': 1,   'min': -np.inf,   'max': np.inf},
            'phi_B':    {'init': 1,   'min': 0,         'max': np.inf},
            'epsilon_r':{'init': 1,   'min': 1e-12,     'max': np.inf},
        },
        'latex': r"$J_{\mathrm{SE}} = \frac{4\pi q m^*(kT)^2}{h^3} \exp\left( \frac{-q\left( \Phi_B - \sqrt{\frac{qE}{4\pi \varepsilon}} \right)}{kT} \right)$"
    },
    'Fowler–Nordheim': {
        'func': J_fowler_nordheim,
        'params': {
            'K_1':   {'init': 1,  'min': -np.inf,  'max': np.inf},
            'K_2':   {'init': 1,  'min': -np.inf,  'max': np.inf},
        },
        'latex': r"$J_{\mathrm{FN}} = \frac{q^2}{8\pi h \Phi_B} E^2 \exp\left( \frac{-8\pi \sqrt{2qm^*} \, \Phi_B^{3/2}}{3hE} \right)$"
    },
    'Direct tunneling': {
        'func': J_direct_tunneling,
        'params': {
            'm_eff': {'init': 1,   'min': 0,        'max': np.inf},
            'phi_B': {'init': 1,   'min': 0,        'max': np.inf},
            'kappa': {'init': 1,   'min': 0,        'max': np.inf},
            't_ox':  {'init': 1,   'min': 0,        'max': np.inf},
        },
        'latex': r"$J_{\mathrm{DT}} \approx \exp\left\{ \frac{-8\pi \sqrt{2q}}{3h} (m^* \Phi_B)^{1/2} \kappa \cdot t_{\mathrm{ox,eq}} \right\}$"
    },
    'Ohmic conduction': {
        'func': J_ohmic,
        'params': {
            'mu':   {'init': 1,   'min': -np.inf,  'max': np.inf},
            'N_C':  {'init': 1,   'min': -np.inf,  'max': np.inf},
            'E_C':  {'init': 1,   'min': -np.inf,  'max': np.inf},
            'E_F':  {'init': 1,   'min': -np.inf,  'max': np.inf},
        },
        'latex': r"$J_{\mathrm{ohmic}} = q \mu N_C E \exp\left[ \frac{-(E_C - E_F)}{kT} \right]$"
    },
    'Poole–Frenkel': {
        'func': J_poole_frenkel,
        'params': {
            'mu':        {'init': 1,   'min': -np.inf,  'max': np.inf},
            'N_C':       {'init': 1,   'min': -np.inf,  'max': np.inf},
            'phi_T':     {'init': 1,   'min': 0,        'max': np.inf},
            'epsilon_r': {'init': 1,   'min': 0,        'max': np.inf},
        },
        'latex': r"$J_{\mathrm{PFE}} = q \mu N_C E \exp\left[ \frac{-q\left( \Phi_T - \sqrt{\frac{qE}{\pi \varepsilon}} \right)}{kT} \right]$"
    },
    'SCLC': {
        'func': J_space_charge_limited,
        'params': {
            'mu':        {'init': 1,   'min': -np.inf,  'max': np.inf},
            'epsilon_r': {'init': 1,   'min': 0,        'max': np.inf},
            'theta':     {'init': 1,   'min': -np.inf,  'max': np.inf},
            'd':         {'init': 1,   'min': 1e-15,    'max': np.inf},
        },
        'latex': r"$J_{\mathrm{SCLC}} = \frac{9}{8} \varepsilon_i \mu \theta \frac{V^2}{d^3}$"
    },
    'Ionic conduction': {
        'func': J_ionic,
        'params': {
            'sigma_0': {'init': 1,   'min': -np.inf,  'max': np.inf},
            'dG':      {'init': 1,   'min': 0,        'max': np.inf},
        },
        'latex': r"$J_{\mathrm{ionic}} \propto \frac{E}{T} \exp\left( \frac{-\Delta G^{\neq}}{kT} \right)$"
    },
    'Nearest-neighbor hopping': {
        'func': J_nearest_neighbor_hopping,
        'params': {
            'sigma_0': {'init': 1,   'min': -np.inf,  'max': np.inf},
            'T0':      {'init': 1,   'min': -np.inf,  'max': np.inf},
        },
        'latex': r"$J_{\mathrm{NNH}} = \sigma_0 \exp\left( \frac{-T_0}{T} \right) \cdot E$"
    },
    'Variable-range hopping': {
        'func': J_variable_range_hopping,
        'params': {
            'sigma_0': {'init': 1,   'min': -np.inf,  'max': np.inf},
            'T0':      {'init': 1,   'min': 0,        'max': np.inf},
        },
        'latex': r"$J_{\mathrm{VRH}} = \sigma_0 \exp \left( \frac{-T_0}{T} \right)^{1/4} \cdot E$"
    },
    'Trap-assisted tunneling': {
        'func': J_trap_assisted_tunneling,
        'params': {
            'A0':     {'init': 1,   'min': -np.inf,  'max': np.inf},
            'm_eff':  {'init': 1,   'min': 0,        'max': np.inf},
            'phi_T':  {'init': 1,   'min': 0,        'max': np.inf},
        },
        'latex': r"$J_{\mathrm{TAT}} = A \exp\left( \frac{-8\pi\sqrt{2qm^*} \, \Phi_T^{3/2}}{3hE} \right)$"
    },
    'Linear test': {
        'func': linear_test,
        'params': {
            'm': {'init': 1, 'min': -np.inf, 'max': np.inf},
            'b': {'init': 1, 'min': -np.inf, 'max': np.inf},
        },
        'latex': r"$J = mx + b$"
    }
}

