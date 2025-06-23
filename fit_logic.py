# fit_logic.py
"""
# Fit Logic Module
## Description
A module for performing fits on data using various electronic conduction-mechanism models.
It provides functions to load data from text files, perform fits using lmfit, and interactively set parameter bounds through a Tkinter dialog.
### Author: Nira
"""
import tkinter as tk
import numpy as np
from lmfit import Model
from models import models
from tkinter import simpledialog
# As overview of fitting methods and the used libary, see:
# https://lmfit.github.io/lmfit-py/fitting.html

def load_txtfile(pfad, delimiter=None, skiprows=3, usecols=(0, 1)):
    """
    Reads a text file with two columns of data and returns two arrays.
    Parameters:
    - pfad: Path to the text file.
    - delimiter: Delimiter used in the file, e.g. ',' or '\t'. If None, whitespace is used.
    - skiprows: Number of rows to skip at the beginning of the file.
    - usecols: Tuple of column indices to read (0-based). Default is (0, 1) for the first two columns.
    """
    data = np.loadtxt(pfad, delimiter=delimiter, skiprows=skiprows, usecols=usecols)
    x = data[:, 0]
    y = data[:, 1]
    return x, y

def perform_fit(x, y, model_key, method='leastsq'):
    """
    Performs a fit on the provided data using the specified model and method with lmfit.
    Parameters:
    - x: Independent variable data (e.g., voltage).
    - y: Dependent variable data (e.g., current density).
    - model_key: Key to select the model from the models dictionary.
    - method: Fitting method to use (default is 'leastsq'). See link above.
    Returns:
    - result: The fit result object containing fit parameters and statistics.
    Raises:
    - KeyError: If the model_key is not found in the models dictionary.
    - ValueError: If the input data is not valid for fitting.
    """

    # Setup the model based on the selected key for fit. Intial values are taken from the model's params.
    if model_key not in models:
        raise KeyError(f"Model '{model_key}' not found in models dictionary.")
    selected_model = models[model_key]
    init_dict = {
        name: float(data.get("init", 1.0)) if data.get("init") is not None else 1.0
        for name, data in selected_model['params'].items()
    }
    model = Model(selected_model['func'], independent_vars=['E'])
    params = model.make_params(**init_dict)

    # Dialogue for setting parameter bounds
    # This dialog allows the user to set lower and upper bounds for each parameter interactively.
    root = tk.Tk()
    root.withdraw()

    class BoundsDialog(simpledialog.Dialog):
        # This is the main body of the dialog where the user can input bounds for each parameter.
        def body(self, master):
            toplevel = self.winfo_toplevel()
            min_width = 600
            min_height = 60 + 30 * (len(selected_model['params']) + 2)
            toplevel.minsize(min_width, min_height)
            hint_text = (
                "With tighter bounds the algorithm may fail to find a good fit - "
                "even if the final parameter is in bounds. So it's recommended to make them "
                "as wide as numerically possible. Avoid negative bounds for parameters that could return complex values."
            )
            lbl_hint = tk.Label(master, text=hint_text, wraplength=min_width-20,
                                justify='left', fg='blue')
            lbl_hint.grid(row=0, column=0, columnspan=4, padx=5, pady=(5, 15), sticky='w')

            # Header-Zeile
            tk.Label(master, text="Parameter", font=('TkDefaultFont', 10, 'bold')).grid(row=1, column=0, padx=5, sticky='w')
            tk.Label(master, text="Lower bound", font=('TkDefaultFont', 10, 'bold')).grid(row=1, column=1, padx=5, sticky='w')
            tk.Label(master, text="Initial Value", font=('TkDefaultFont', 10, 'bold')).grid(row=1, column=2, padx=5, sticky='w')
            tk.Label(master, text="Upper bound", font=('TkDefaultFont', 10, 'bold')).grid(row=1, column=3, padx=5, sticky='w')

            self.entries = {}
            # Parameters from row=2. Dynically create entries for each parameter.
            for i, param_name in enumerate(selected_model['params']):
                row_idx = i + 2
                tk.Label(master, text=param_name).grid(row=row_idx, column=0, padx=5, pady=2, sticky='w')

                e1 = tk.Entry(master, width=15)  # Lower bound
                e2 = tk.Entry(master, width=15)  # Initial value
                e3 = tk.Entry(master, width=15)  # Upper bound

                # Set default values for entries based on the model's params (if available)
                param_data = selected_model['params'][param_name]
                lo = param_data.get("min", -np.inf)
                hi = param_data.get("max", np.inf)

                # Convert to string for display, handling infinities
                # If the value is a finite number, convert it to string; otherwise use "-inf" or "inf"    
                lo_str = str(lo) if isinstance(lo, (float, int)) and np.isfinite(lo) else "-inf"
                hi_str = str(hi) if isinstance(hi, (float, int)) and np.isfinite(hi) else "inf"

                init_val = init_dict.get(param_name, 1.0)
                e1.insert(0, lo_str)
                e2.insert(0, str(init_val))
                e3.insert(0, hi_str)

                e1.grid(row=row_idx, column=1, padx=5, pady=2)
                e2.grid(row=row_idx, column=2, padx=5, pady=2)
                e3.grid(row=row_idx, column=3, padx=5, pady=2)

                self.entries[param_name] = (e1, e2, e3)

            # Set focus on the first entry
            if self.entries:
                first_param = next(iter(self.entries))
                return self.entries[first_param][0]
            return None

        def apply(self):
            """
            This method is called when the user clicks 'OK' in the dialog.
            It collects the values from the entries and parses them into a dictionary.
            """
            self.result = {}
            for param, (e1, e2, e3) in self.entries.items():
                lower = e1.get().strip()
                init = e2.get().strip()
                upper = e3.get().strip()
                # Parsen: falls "-inf" oder "" oder unparseable => -inf; falls "inf" => +inf
                # Lower
                try:
                    if lower.lower() in ('', 'none'):
                        lower_val = -np.inf
                    elif lower.lower() in ('-inf', '-infinity', '-inf.'):
                        lower_val = -np.inf
                    else:
                        lower_val = float(lower)
                except Exception:
                    lower_val = -np.inf
                # Initial
                try:
                    if init.lower() in ('', 'none'):
                        init_val = init_dict.get(param, 1.0)
                    else:
                        init_val = float(init)
                except Exception:
                    init_val = init_dict.get(param, 1.0)
                # Upper
                try:
                    if upper.lower() in ('', 'none'):
                        upper_val = np.inf
                    elif upper.lower() in ('inf', 'infinity', '+inf', 'inf.'):
                        upper_val = np.inf
                    else:
                        upper_val = float(upper)
                except Exception:
                    upper_val = np.inf

                self.result[param] = (lower_val, init_val, upper_val)

    dialog = BoundsDialog(root, title="Set Parameter Bounds")
    # When the dialog is closed, check if the result is valid.
    # If the user clicked 'OK', dialog.result will contain the bounds
    if hasattr(dialog, 'result') and dialog.result:
        for param, (lower_val, init_val, upper_val) in dialog.result.items():
            try:
                params[param].set(value=init_val, min=lower_val, max=upper_val)
            except Exception as e:
                print(f"Warning: Parameter '{param}' couldn't be set. Exception: {e}")

    root.destroy()

    # Performs the fit using the specified method.
    result = model.fit(y, params, E=x, method=method)
    return result
