import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from models import models
import tkinter as tk
from tkinter import simpledialog

# Simulierte Daten (lineares Modell mit Rauschen)
np.random.seed(42)
x = np.linspace(0, 10, 10)
true_slope = 2.5
true_intercept = 1.0
y = true_slope * x + true_intercept + np.random.normal(scale=1.0, size=len(x))

# lmfit-Modell erstellen basierend auf dem Dictionary
selected_model = models['Schottky emission']  # Beispielmodell auswählen
model = Model(selected_model['func'], independent_vars=['E'])

# Parameter aus dem Dictionary erstellen und Grenzen abfragen
params = model.make_params(**{param: 1.0 for param in selected_model['params']})

root = tk.Tk()
root.withdraw()  # Hauptfenster ausblenden

# Dialog für alle Parameter in einem Fenster
class BoundsDialog(simpledialog.Dialog):
    def __init__(self, *args, **kwargs):
        self.result = None
        super().__init__(*args, **kwargs)

    def body(self, master):
        self.entries = {}
        tk.Label(master, text="Parameter").grid(row=0, column=0)
        tk.Label(master, text="Lower bound").grid(row=0, column=1)
        tk.Label(master, text="Upper bound").grid(row=0, column=2)
        for i, param in enumerate(selected_model['params']):
            tk.Label(master, text=param).grid(row=i+1, column=0)
            e1 = tk.Entry(master)
            e2 = tk.Entry(master)
            e1.grid(row=i+1, column=1)
            e2.grid(row=i+1, column=2)
            self.entries[param] = (e1, e2)
        return list(self.entries.values())[0][0]  # focus

    def apply(self):
        self.result = {}
        for param, (e1, e2) in self.entries.items():
            lower = e1.get()
            upper = e2.get()
            try:
                lower_val = float(lower) if lower not in (None, "") else -np.inf
            except ValueError:
                lower_val = -np.inf
            try:
                upper_val = float(upper) if upper not in (None, "") else np.inf
            except ValueError:
                upper_val = np.inf
            self.result[param] = (lower_val, upper_val)

dialog = BoundsDialog(root, title="Set Parameter Bounds")
if dialog.result:
    for param, (lower_val, upper_val) in dialog.result.items():
        params[param].set(min=lower_val, max=upper_val)

root.destroy()

# Fit durchführen
result = model.fit(y, params, E=x, T=300, method='leastsq', nan_policy='omit')

# Ergebnisse ausgeben
print(result.fit_report())

# Plot
plt.figure(figsize=(8, 5))
plt.plot(x, y, 'bo', label='Daten')
plt.plot(x, result.best_fit, 'r-', label='Fit')
plt.xlabel('E')
plt.ylabel('J')
plt.legend()
plt.grid(True)
plt.title('Linearer Fit mit lmfit')
plt.show()
