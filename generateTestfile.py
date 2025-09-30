import numpy as np
import models

# =============================================================================
# Definition der Testparameter für jedes Modell
# =============================================================================

# Gemeinsame Parameter
T = 300  # Temperatur in Kelvin
epsilon_r = 6  # Relative Dielektrizitätskonstante (z.B. für SiO2)
A = 625e-12  # Fläche in m^2
d = 10e-9  # Schichtdicke in m


params_schottky = {
    "A": 120e4,         
    "phi_B": 0.8,       
    "epsilon_r": epsilon_r,
    "T": T
}

params_fowler_nordheim = {
    "K_1": 1.54e-6,      
    "K_2": 6.83e3,      
    "phi_B": 0.9       
}

params_direct_tunneling = {
    "m_eff": 0.5,        
    "phi_B": 0.7,       
    "kappa": 1.0,        
    "t_ox": 2e-9         
}

params_direct_tunneling_alt = {
    "m_eff": 0.4,        
    "phi_B": 0.8        
}

params_ohmic = {
    "mu": 0.1,           
    "N_C": 2.8e25,       
    "E_C": 1.12,        
    "E_F": 0.5,         
    "T": T
}

params_poole_frenkel = {
    "mu": 0.05,
    "N_C": 3e25,
    "phi_T": 0.5,       
    "epsilon_r": epsilon_r,
    "T": T
}

params_space_charge_limited = {
    "mu": 0.01,
    "epsilon_r": epsilon_r,
    "theta": 1.0,        
    "d": 100e-9         
}

params_ionic = {
    "dG": 0.3,          
    "sigma_0": 1e-12,    
    "T": T
}

params_hopping = {
    "sigma_0": 1e-10,
    "T0": 1000,          
    "T": T
}

params_trap_assisted_tunneling = {
    "A": 1e-3,           
    "m_eff": 0.5,
    "phi_T": 0.6       
}

params_linear_test = {
    "m": 2.5e-9,        
    "b": 1e-12          
}


# =============================================================================
# Datengenerierung und Dateierstellung
# =============================================================================

# Erstelle einen gemeinsamen E-Wertebereich (elektrisches Feld in V/m)
# Logarithmische Skala, da viele Effekte stark feldabhängig sind
E_values = np.linspace(1e-9, 12e7, 1000)[1::]
print(E_values)

def create_test_file(model_name, E, J):
    """
    Erstellt eine Textdatei für ein gegebenes Modell mit E- und J-Werten.
    """
    filename = f"./Testdaten/{model_name}_test.txt"
    try:
        with open(filename, 'w') as f:
            # Schreibe die ersten drei Zeilen als Header/Info
            f.write(f"Testfile_{model_name}\n")
            f.write("# Mode: U-Sweep; Generated Test Data\n")
            f.write("U / (V)\tI / (A)\n")

            for e_val, j_val in zip(E, J):
                f.write(f"{e_val:.6e}\t{j_val:.6e}\n")
        print(f"Datei '{filename}' erfolgreich erstellt.")
    except Exception as e:
        print(f"Fehler beim Erstellen der Datei '{filename}': {e}")


def combinationModel(x):
    return models.exponential_test(x, A=-1, x0=1) + models.exponential_test(x, A=1, x0=3) 


# Liste aller Modelle und ihrer Parameter
model_list = [
    (models.J_schottky, params_schottky),
    (models.J_fowler_nordheim, params_fowler_nordheim),
    (models.J_direct_tunneling, params_direct_tunneling),
    (models.J_direct_tunneling_alt, params_direct_tunneling_alt),
    (models.J_ohmic, params_ohmic),
    (models.J_poole_frenkel, params_poole_frenkel),
    (models.J_space_charge_limited, params_space_charge_limited),
    (models.J_ionic, params_ionic),
    (models.J_nearest_neighbor_hopping, params_hopping),
    (models.J_variable_range_hopping, params_hopping),
    (models.J_trap_assisted_tunneling, params_trap_assisted_tunneling),
    (models.linear_test, params_linear_test),
]

# Hauptlogik: Iteriere durch die Modelle und erstelle die Dateien
for model_func, params in model_list:
    model_name = model_func.__name__
    try:
        # Berechne J-Werte für das aktuelle Modell
        J_values = model_func(E_values, **params)

        # Erstelle die Testdatei
        create_test_file(model_name, E_values * d, J_values * A)
    except Exception as e:
        print(f"Konnte Datei für '{model_name}' nicht erstellen. Fehler: {e}")

x = np.linspace(0, 5, 70)
create_test_file("Combination_model", x * d, combinationModel(x) * A)

print("\nAlle Testdateien wurden generiert.")