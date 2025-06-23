import numpy as np
import matplotlib.pyplot as plt

# Beispiel-Funktionen
def f1(x):
    return np.sin(x)

def f2(x):
    return np.sin(x)**2 + 1

# Übergangsbereich definieren
transition_start = 2
transition_end = 4

# Glatte Übergangsfunktion (cosine-based)
def smoothstep(x, a, b):
    """Erzeugt eine glatte Gewichtung zwischen 0 und 1 im Bereich [a, b]."""
    x = np.clip((x - a) / (b - a), 0, 1)
    return 0.5 * (1 - np.cos(np.pi * x))  # Cosine interpolation

# Zusammengesetzte Funktion
def smooth_combined_function(x):
    w = smoothstep(x, transition_start, transition_end)
    return (1 - w) * f1(x) + w * f2(x)

# Plotbereich
x = np.linspace(0, 6, 500)
y = smooth_combined_function(x)

# Plotten
plt.figure(figsize=(10, 5))
plt.plot(x, y, label='Glatter Übergang')
plt.plot(x, f1(x), '--', label='f1(x)', alpha=0.5)
plt.plot(x, f2(x), '--', label='f2(x)', alpha=0.5)
plt.axvline(transition_start, color='gray', linestyle=':', label='Übergang beginnt')
plt.axvline(transition_end, color='gray', linestyle=':', label='Übergang endet')
plt.legend()
plt.title("Glatte Verbindung zweier Funktionen")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.tight_layout()
plt.show()
