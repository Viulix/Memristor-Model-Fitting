import numpy as np
import models

x_values = np.linspace(1, 1e11, 100)
y = models.J_trap_assisted_tunneling(x_values, A=1e-1, m_eff=0.5, phi_T=0.6)
print(y)