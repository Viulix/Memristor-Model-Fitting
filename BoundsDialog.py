"""
BoundsDialog Module

A Tkinter dialog for setting parameter bounds in fitting applications.
Provides an interface for users to set lower bounds, initial values, and upper bounds for model parameters.

Author: Nira
"""
import tkinter as tk
import numpy as np
from tkinter import simpledialog


class BoundsDialog(simpledialog.Dialog):
    """
    A dialog for setting parameter bounds for model fitting.
    
    Allows users to specify lower bounds, initial values, and upper bounds
    for each parameter in a model.
    """
    
    def __init__(self, parent, title, params_def, initial_vals, primFont=None, secFont=None):
        """
        Initialize the BoundsDialog.
        
        Parameters:
        - parent: Parent window
        - title: Dialog window title
        - params_def: Dictionary of parameter definitions with 'min', 'max', etc.
        - initial_vals: Dictionary of initial parameter values
        - primFont: Primary font (currently unused)
        - secFont: Secondary font for dialog elements
        """
        self.params_def = params_def
        self.initial_vals = initial_vals
        self.primFont = primFont
        self.secFont = secFont
        super().__init__(parent, title)

    def body(self, master):
        """
        Create the dialog body with parameter input fields.
        
        Returns the first entry widget for initial focus.
        """
        toplevel = self.winfo_toplevel()
        toplevel.minsize(600, 60 + 30 * (len(self.params_def) + 2))

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
        for i, name in enumerate(self.params_def, start=2):
            tk.Label(master, text=name, font=self.secFont).grid(row=i, column=0, padx=5, pady=2, sticky='w')

            e_lo = tk.Entry(master, width=15, font=self.secFont)
            e_init = tk.Entry(master, width=15, font=self.secFont)
            e_hi = tk.Entry(master, width=15, font=self.secFont)

            pd = self.params_def[name]
            lo = pd.get('min', -np.inf)
            hi = pd.get('max', np.inf)
            lo_str = str(lo) if np.isfinite(lo) else '-inf'
            hi_str = str(hi) if np.isfinite(hi) else 'inf'
            e_lo.insert(0, lo_str)
            e_init.insert(0, str(self.initial_vals[name]))
            e_hi.insert(0, hi_str)

            e_lo.grid(row=i, column=1, padx=5, pady=2)
            e_init.grid(row=i, column=2, padx=5, pady=2)
            e_hi.grid(row=i, column=3, padx=5, pady=2)

            self.entries[name] = (e_lo, e_init, e_hi)

        # Focus first entry
        first = next(iter(self.entries.values()), (None,))[0]
        return first

    def apply(self):
        """
        Apply the dialog results by parsing all input values.
        
        Sets self.result to a dictionary mapping parameter names to 
        (lower_bound, initial_value, upper_bound) tuples.
        """
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
            init = parse(e_init.get(), self.initial_vals[name], ())
            upper = parse(e_hi.get(), np.inf, ('inf', 'infinity', '+inf'))
            self.result[name] = (lower, init, upper)
