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
from BoundsDialog import BoundsDialog
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

def performFit(x, y, model_key_a, model_key_b=None, method='leastsq', T=None, primFont=None, secFont=None):
    """
    Performs a fit on the provided data using one or two specified models.
    If two models are provided, they are summed together.
    Allows setting of custom primary and secondary fonts for dialog widgets.
    """
    if model_key_a not in models:
        raise KeyError(f"Model '{model_key_a}' not found in models dictionary.")
    
    is_combination = model_key_b is not None
    dialog_params = {}
    init_dict = {}

    # Determining whether we have two models to combine or just one. Combined models are summed.
    if is_combination:
        if model_key_b not in models:
            raise KeyError(f"Model '{model_key_b}' not found in models dictionary.")
        
        # Always use prefixes to avoid parameter name conflicts
        model1 = Model(models[model_key_a]['func'], independent_vars=['E'], prefix='m1_')
        model2 = Model(models[model_key_b]['func'], independent_vars=['E'], prefix='m2_')
        
        model = model1 + model2
        params = model.make_params()

        # Find shared parameter names (without prefixes)
        params_a = set(models[model_key_a]['params'].keys())
        params_b = set(models[model_key_b]['params'].keys())
        if model_key_a == model_key_b:
            # If both models are the same, all parameters need to be distinguished, so no shared parameters
            shared_params = set()
        else:
            shared_params = params_a & params_b
        
        # Collect parameters for dialog - only show shared parameters once
        for original_name in params_a | params_b:
            if original_name in shared_params:
                # For shared parameters, only show once (use m1_ prefix)
                dialog_params[original_name] = models[model_key_a]['params'][original_name]
                init_dict[original_name] = float(dialog_params[original_name].get('init', 1.0))
            else:
                # For unique parameters, show with prefix
                if original_name in params_a:
                    prefixed_name = 'm1_' + original_name
                    dialog_params[prefixed_name] = models[model_key_a]['params'][original_name]
                    init_dict[prefixed_name] = float(dialog_params[prefixed_name].get('init', 1.0))
                if original_name in params_b:
                    prefixed_name = 'm2_' + original_name
                    dialog_params[prefixed_name] = models[model_key_b]['params'][original_name]
                    init_dict[prefixed_name] = float(dialog_params[prefixed_name].get('init', 1.0))
            
    else:
        # Single model fit
        selected_model = models[model_key_a]
        model = Model(selected_model['func'], independent_vars=['E'])
        dialog_params = selected_model['params']
        init_dict = {name: float(data.get('init', 1.0)) for name, data in dialog_params.items()}
        params = model.make_params(**init_dict)

    # Dialog for parameter bounds
    root = tk.Tk()
    root.withdraw()

    # Invoke dialog
    dialog = BoundsDialog(root, 'Set Parameter Bounds', dialog_params, init_dict, primFont, secFont)
    if dialog.result is None:
        root.destroy()
        return None  # User cancelled the dialog
    if hasattr(dialog, 'result'):
        if is_combination:
            # Find shared parameters again
            params_a = set(models[model_key_a]['params'].keys())
            params_b = set(models[model_key_b]['params'].keys())
            shared_params = params_a & params_b
            
            for name, (lo, init, hi) in dialog.result.items():
                try:
                    # Check if this is a shared parameter (no prefix in dialog)
                    if name in shared_params:
                        # Set m1_ version with bounds
                        params['m1_' + name].set(value=init, min=lo, max=hi)
                        # Constrain m2_ version to equal m1_ version
                        params['m2_' + name].set(expr='m1_' + name)
                    else:
                        # Non-shared parameter, set directly
                        params[name].set(value=init, min=lo, max=hi)
                except Exception as e:
                    print(f"Warning: Could not set '{name}': {e}")
        else:
            for name, (lo, init, hi) in dialog.result.items():
                try:
                    params[name].set(value=init, min=lo, max=hi)
                except Exception as e:
                    print(f"Warning: Could not set '{name}': {e}")
    root.destroy()
    if T is not None:
        if is_combination:
            # Add T as a fixed parameter to both sub-models if they use it.
            if 'm1_T' in params:
                params['m1_T'].set(value=float(T), vary=False)
            if 'm2_T' in params:
                params['m2_T'].set(value=float(T), vary=False)
        else:
            # Add T as a fixed parameter if the model uses it.
            params.add('T', value=float(T), vary=False)

    return model.fit(y, params, E=x, method=method)

