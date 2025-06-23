import numpy as np
import matplotlib.pyplot as plt
import os
import models
from lmfit import Model


# Funktion zum Einlesen der Messdaten aus einer Datei
def lade_messdaten(pfad, delimiter=None, skiprows=3, usecols=(0, 1)):
    daten = np.loadtxt(pfad, delimiter=delimiter, skiprows=skiprows, usecols=usecols)
    x_daten = daten[:, 0]
    y_daten = daten[:, 1]
    return x_daten, y_daten

# Beispiel: Passe den Pfad ggf. an dein System an
ordner = "Testdaten"
dateiname = "2023-10-26_12-32-14_u230119a__1610234_ copy.qtj"
pfad = os.path.join(ordner, dateiname)

# Passe ggf. delimiter, skiprows, usecols an das Dateiformat an!
x_daten, y_daten = lade_messdaten(pfad, delimiter=None, skiprows=3, usecols=(0, 1))


# Funktion zum Fitten mit least_squares
# Die Funktion wird aktuell nicht verwendet und alle Parameter sind ungenutzt, daher entfernen wir sie.

# lmfit-Modell erstellen
model = Model(models.J_trap_assisted_tunneling)
# Anfangswerte setzen und Grenzen definieren
params = model.make_params(A_0=1.2, phi_T=0.7, m_eff=1)
params['phi_T'].set(min=0, max=1)
params['m_eff'].set(min=0) 

# Fit durchführen
result = model.fit(y_daten, params, E=x_daten)

# Ergebnisse ausgeben
# Print fit report, replacing non-ASCII characters to avoid UnicodeEncodeError
print(result.fit_report().encode('ascii', errors='replace').decode())

# Plot
plt.figure(figsize=(8, 5))
plt.plot(x_daten, y_daten, 'bo', label='Daten')
plt.plot(x_daten, result.best_fit, 'r-', label='Fit')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.title('Linearer Fit mit lmfit')
plt.show()
