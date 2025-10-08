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
    
    def __init__(self, parent, title, params_def, initial_vals, saved_bounds=None, primFont=None, secFont=None):
        """
        Initialize the BoundsDialog.
        
        Parameters:
        - parent: Parent window
        - title: Dialog window title
        - params_def: Dictionary of parameter definitions with 'min', 'max', etc.
        - initial_vals: Dictionary of initial parameter values
        - saved_bounds: Dictionary of previously saved bounds (optional)
        - primFont: Primary font (currently unused)
        - secFont: Secondary font for dialog elements
        """
        self.params_def = params_def
        self.initial_vals = initial_vals
        self.saved_bounds = saved_bounds or {}
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

        # Header row with 'Fixed' column
        headers = ['Parameter', 'Lower bound', 'Initial Value', 'Upper bound', 'Fixed']
        for col, text in enumerate(headers):
            tk.Label(master, text=text, font=self.secFont).grid(row=1, column=col, padx=5, sticky='w')

        self.entries = {}
        for i, name in enumerate(self.params_def.keys(), start=2):
            tk.Label(master, text=name, font=self.secFont).grid(row=i, column=0, padx=5, pady=2, sticky='w')

            e_lo = tk.Entry(master, width=15, font=self.secFont)
            e_init = tk.Entry(master, width=15, font=self.secFont)
            e_hi = tk.Entry(master, width=15, font=self.secFont)

            pd = self.params_def[name]
            
            # Check if we have saved bounds for this parameter
            if name in self.saved_bounds:
                saved = self.saved_bounds[name]
                lo_str = saved.get('lo', str(pd.get('min', -np.inf)))
                init_str = saved.get('init', str(self.initial_vals[name]))
                hi_str = saved.get('hi', str(pd.get('max', np.inf)))
                fixed_val = saved.get('fixed', False)
            else:
                # Use default values from params_def
                lo = pd.get('min', -np.inf)
                hi = pd.get('max', np.inf)
                lo_str = str(lo) if np.isfinite(lo) else '-inf'
                hi_str = str(hi) if np.isfinite(hi) else 'inf'
                init_str = str(self.initial_vals[name])
                fixed_val = False

            e_lo.insert(0, lo_str)
            e_init.insert(0, init_str)
            e_hi.insert(0, hi_str)

            e_lo.grid(row=i, column=1, padx=5, pady=2)
            e_init.grid(row=i, column=2, padx=5, pady=2)
            e_hi.grid(row=i, column=3, padx=5, pady=2)

            # Add 'Fixed' checkbox
            fixed_var = tk.BooleanVar(master=master, value=fixed_val)
            fixed_cb = tk.Checkbutton(master, variable=fixed_var)
            fixed_cb.grid(row=i, column=4, padx=5, pady=2)

            self.entries[name] = {'lo': e_lo, 'init': e_init, 'hi': e_hi, 'fixed_var': fixed_var}
        
        # Focus first entry
        first_entry_dict = next(iter(self.entries.values()), None)
        first = first_entry_dict['lo'] if first_entry_dict else None
        return first

    def buttonbox(self):
        """Add standard button box with additional Reset button."""
        box = tk.Frame(self)

        reset_btn = tk.Button(box, text="Reset to Defaults", width=15, command=self.reset_to_defaults)
        reset_btn.pack(side=tk.LEFT, padx=5, pady=5)

        ok_btn = tk.Button(box, text="OK", width=10, command=self.ok, default=tk.ACTIVE)
        ok_btn.pack(side=tk.LEFT, padx=5, pady=5)

        cancel_btn = tk.Button(box, text="Cancel", width=10, command=self.cancel)
        cancel_btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

        box.pack()

    def reset_to_defaults(self):
        """Reset all fields to hardcoded default values from params_def."""
        for name, entry_dict in self.entries.items():
            pd = self.params_def[name]
            
            # Get default values from params_def
            lo = pd.get('min', -np.inf)
            hi = pd.get('max', np.inf)
            lo_str = str(lo) if np.isfinite(lo) else '-inf'
            hi_str = str(hi) if np.isfinite(hi) else 'inf'
            init_str = str(self.initial_vals[name])
            
            # Clear and update entries
            entry_dict['lo'].delete(0, tk.END)
            entry_dict['lo'].insert(0, lo_str)
            
            entry_dict['init'].delete(0, tk.END)
            entry_dict['init'].insert(0, init_str)
            
            entry_dict['hi'].delete(0, tk.END)
            entry_dict['hi'].insert(0, hi_str)
            
            entry_dict['fixed_var'].set(False)

    def apply(self):
        """
        Apply the dialog results by parsing all input values.
        
        Sets self.result to a dictionary mapping parameter names to 
        (lower_bound, initial_value, upper_bound) tuples.
        """
        self.result = {}
        for name, entry_dict in self.entries.items():
            e_lo = entry_dict['lo']
            e_init = entry_dict['init']
            e_hi = entry_dict['hi']
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
            is_fixed = entry_dict['fixed_var'].get()
            
            self.result[name] = (lower, init, upper, is_fixed)
