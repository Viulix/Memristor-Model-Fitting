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

def perform_fit(x, y, model_key, method='leastsq', T=None, primFont=None, secFont=None):
    """
    Performs a fit on the provided data using the specified model and method with lmfit.
    Allows setting of custom primary and secondary fonts for dialog widgets.
    """
    if model_key not in models:
        raise KeyError(f"Model '{model_key}' not found in models dictionary.")
    selected_model = models[model_key]
    init_dict = {name: float(data.get('init', 1.0)) for name, data in selected_model['params'].items()}
    model = Model(selected_model['func'], independent_vars=['E'])
    params = model.make_params(**init_dict)

    # Dialog for parameter bounds
    root = tk.Tk()
    root.withdraw()

    class BoundsDialog(simpledialog.Dialog):
        def __init__(self, *args, **kwargs):
            self.primFont = primFont
            self.secFont = secFont
            super().__init__(*args, **kwargs)

        def body(self, master):
            toplevel = self.winfo_toplevel()
            toplevel.minsize(600, 60 + 30 * (len(selected_model['params']) + 2))

            hint_text = (
                "With tighter bounds the algorithm may fail to find a good fit - "
                "even if the final parameter is in bounds. So it's recommended to make them "
                "as wide as numerically possible. Avoid negative bounds for parameters that could return complex values."
            )
            lbl_hint = tk.Label(
                master,
                text=hint_text,
                wraplength=580,
                justify='left',
                fg='blue'
            )
            lbl_hint.grid(row=0, column=0, columnspan=4, padx=5, pady=(5, 15), sticky='w')

            # Header row
            headers = ['Parameter', 'Lower bound', 'Initial Value', 'Upper bound']
            for col, text in enumerate(headers):
                tk.Label(master, text=text, font=self.secFont).grid(row=1, column=col, padx=5, sticky='w')

            self.entries = {}
            for i, name in enumerate(selected_model['params'], start=2):
                tk.Label(master, text=name, font=self.secFont).grid(row=i, column=0, padx=5, pady=2, sticky='w')

                e_lo = tk.Entry(master, width=15, font=self.secFont)
                e_init = tk.Entry(master, width=15, font=self.secFont)
                e_hi = tk.Entry(master, width=15, font=self.secFont)

                pd = selected_model['params'][name]
                lo = pd.get('min', -np.inf)
                hi = pd.get('max', np.inf)
                lo_str = str(lo) if np.isfinite(lo) else '-inf'
                hi_str = str(hi) if np.isfinite(hi) else 'inf'
                e_lo.insert(0, lo_str)
                e_init.insert(0, str(init_dict[name]))
                e_hi.insert(0, hi_str)

                e_lo.grid(row=i, column=1, padx=5, pady=2)
                e_init.grid(row=i, column=2, padx=5, pady=2)
                e_hi.grid(row=i, column=3, padx=5, pady=2)

                self.entries[name] = (e_lo, e_init, e_hi)

            # Focus first entry
            first = next(iter(self.entries.values()), (None,))[0]
            return first

        def apply(self):
            self.result = {}
            for name, (e_lo, e_init, e_hi) in self.entries.items():
                def parse(val, default, infinities):
                    v = val.strip().lower()
                    if not v or v in infinities:
                        return default
                    try:
                        return float(v)
                    except ValueError:
                        return default

                lower = parse(e_lo.get(), -np.inf, ('-inf', '-infinity'))
                init = parse(e_init.get(), init_dict[name], ())
                upper = parse(e_hi.get(), np.inf, ('inf', 'infinity', '+inf'))
                self.result[name] = (lower, init, upper)

    # Invoke dialog
    dialog = BoundsDialog(root, title='Set Parameter Bounds')
    if dialog.result is None:
        root.destroy()
        return None  # User cancelled the dialog
    if hasattr(dialog, 'result'):
        for name, (lo, init, hi) in dialog.result.items():
            try:
                params[name].set(value=init, min=lo, max=hi)
            except Exception as e:
                print(f"Warning: Could not set '{name}': {e}")
    root.destroy()
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if T is not None:
        params.add('T', value=float(T), vary=False)
    return model.fit(y, params, E=x, method=method)

