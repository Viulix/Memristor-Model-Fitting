# --- Standard Library Imports ---
import io                          # For byte stream handling (e.g. image buffers) --> for LaTeX rendering
import inspect                     # For introspection (e.g. argument inspection)
import warnings                    # To control or suppress warnings
import datetime                    # For handling timestamps and time formatting
import sys                         # System-specific parameters and functions

# --- Third-Party Imports ---
import numpy as np                 # Numerical computing
from PIL import Image, ImageTk     # Image handling for Tkinter --> for rendering LaTeX equations

import matplotlib.pyplot as plt                                  # Plotting
from matplotlib.widgets import SpanSelector                      # Interactive span selector
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Embedding Matplotlib in Tkinter

from scipy.interpolate import make_interp_spline           # Smooth curve interpolation

# --- GUI: Tkinter Modules ---
import tkinter as tk
from tkinter import (
    filedialog,     # File dialogs (open/save)
    ttk,            # Themed widgets
    messagebox,     # Dialog popups (info/warning/error)
    simpledialog,   # Input dialog
    font as tkfont  # Font handling
)
import tooltips

# --- Local Application Imports ---
from models import models           # Model definitions. Contains the functions and parameters for fitting models.
from fit_logic import load_txtfile, perform_fit  # File parsing and fitting logic

# --- Matplotlib Configuration ---
# Enable Latex in Matplotlib. Requires a LaTeX installation. Disable if not needed.
plt.rcParams.update({
    "text.usetex": True, # Switch this for LaTeX rendering on/off
})


class FitApp(tk.Tk):
    def __init__(self):
        # --- Initialize the Tkinter root window ---
        super().__init__()
        self.title("Data Fitting App")
        self.state('zoomed')
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        # --- Set up the main window layout and style ---
        font_size = 12
        default_font = tkfont.Font(family="Helvetica", size=font_size)
        style = ttk.Style(self)
        style.configure("TButton", font=default_font)
        style.configure("TLabel", font=default_font)
        style.configure("TEntry", font=default_font)
        style.configure("TCombobox", font=default_font)
        style.configure("TFrame", font=default_font)
        style.configure("TText", font=default_font)

        # --- Data storage ---
        self.x_all = None
        self.y_all = None
        self.current_x = None
        self.current_y = None
        self.selected_range = None
        self.temp_fit = None
        self.fits = []  # list of dicts: {model_key, range, popt, pcov, func, method}
        # Peak and return indices for positive and negative sweeps
        self.pos_peak_index = None
        self.pos_return_index = None
        self.neg_peak_index = None
        self.neg_return_index = None
        self.zero_index = None
        
        # LaTeX rendering temporary variable
        self.latex_label = None

        # --- File selection ---
        file_frame = ttk.Frame(self)
        file_frame.pack(padx=10, pady=5, fill='x')
        ttk.Label(file_frame, text="Data file:").pack(side='left')
        self.file_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.file_var, width=60)
        file_entry.pack(side='left', padx=5)
        ttk.Button(file_frame, text="Browse...", command=self.browse_file).pack(side='left')

        # --- Subset selection ---
        subset_frame = ttk.Frame(self)
        subset_frame.pack(padx=10, pady=5, fill='x')
        ttk.Label(subset_frame, text="Subset:").pack(side='left')
        self.subset_var = tk.StringVar(value="All")
        subset_options = ["All", "Positive x", "Negative x", "Positive LRS", "Positive HRS", "Negative LRS", "Negative HRS"]
        subset_menu = ttk.OptionMenu(subset_frame, self.subset_var, subset_options[0], *subset_options, command=lambda _: self.apply_subset())
        subset_menu.pack(side='left', padx=5)

        # --- Model selection and fit actions ---
        action_frame = ttk.Frame(self)
        action_frame.pack(padx=10, pady=5, fill='x')
        ttk.Label(action_frame, text="Model:").pack(side='left')
        self.model_var = tk.StringVar()
        model_names = list(models.keys())
        if model_names:
            self.model_var.set(model_names[0])
        model_menu = ttk.OptionMenu(action_frame, self.model_var, model_names[0] if           model_names else "", *model_names, command= lambda _: self.update_latex_display())
        model_menu.pack(side='left', padx=5)

        # Fit method selection
        ttk.Label(action_frame, text="Fit Method:").pack(side='left', padx=(20, 0))
        self.fitmethod_var = tk.StringVar()
        method_options = ["leastsq", "least_squares", "ampgo"]
        self.fitmethod_var.set(method_options[0])
        fitmethod_menu = ttk.OptionMenu(action_frame, self.fitmethod_var, method_options[0], *method_options)
        fitmethod_menu.pack(side='left', padx=5)

        fitButton = ttk.Button(action_frame, text="Fit Selection", command=self.fit_selection)
        fitButton.pack(side='left', padx=5)

        addFitButton = ttk.Button(action_frame, text="Add Fit", command=self.add_fit)
        addFitButton.pack(side='left', padx=5)
        
        removeFitButton = ttk.Button(action_frame, text="Remove Fit", command=self.remove_fit)
        removeFitButton.pack(side='left', padx=5)
        
        saveFitButton = ttk.Button(action_frame, text="Save Fits", command=self.save_fits)
        saveFitButton.pack(side='left', padx=5)

        # --- Selected range display ---
        range_frame = ttk.Frame(self)
        range_frame.pack(padx=10, pady=5, fill='x')
        ttk.Label(range_frame, text="Selected Range:").pack(side='left')
        self.range_label = ttk.Label(range_frame, text="None")
        self.range_label.pack(side='left', padx=5)

        # Input for manual range setting
        ttk.Label(range_frame, text="From:").pack(side='left', padx=(10,0))
        self.range_min_var = tk.StringVar()
        ttk.Entry(range_frame, textvariable=self.range_min_var, width=10).pack(side='left')

        ttk.Label(range_frame, text="To:").pack(side='left', padx=(10,0))
        self.range_max_var = tk.StringVar()
        ttk.Entry(range_frame, textvariable=self.range_max_var, width=10).pack(side='left')

        setRangeButton = ttk.Button(range_frame, text="Set Range", command=self.set_manual_range)
        setRangeButton.pack(side='left', padx=5)

        delTempFitButton = ttk.Button(range_frame, text="Delete Temp. Fit", command=self.apply_subset)
        delTempFitButton.pack(side='left', padx=5)
        
        interpolateButton = ttk.Button(range_frame, text="Interpolate Fits", command=self.interpolate_fits)
        interpolateButton.pack(side='left', padx=5)

        # --- Fit list display ---
        fit_list_frame = ttk.Frame(self)
        fit_list_frame.pack(padx=10, pady=5, fill='x', expand=False)
        ttk.Label(fit_list_frame, text="Fits:").pack(side='top', anchor='w')
        self.fit_listbox = tk.Listbox(fit_list_frame, height=3)
        self.fit_listbox.pack(side='left', fill='x', expand=True)
        scrollbar = ttk.Scrollbar(fit_list_frame, orient='vertical', command=self.fit_listbox.yview)
        scrollbar.pack(side='right', fill='y')
        self.fit_listbox.config(yscrollcommand=scrollbar.set)
        
        # --- Model LaTeX equation display ---
        self.latex_frame = ttk.Frame(self)
        self.latex_frame.pack(padx=10, pady=5, fill='x')
        ttk.Label(self.latex_frame, text="Model Equation:").pack(side='left')

        # Initialanzeige
        self.update_latex_display()

        # --- Plot area ---
        self.fig, (self.ax_log, self.ax_lin) = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
        self.ax_log.set_yscale('log')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(padx=10, pady=10, fill='both', expand=True)

        # Initialize SpanSelectors auf beiden Plots
        sig = inspect.signature(SpanSelector)
        span_args = {'direction': 'horizontal', 'useblit': True}
        if 'rectprops' in sig.parameters:
            span_args['rectprops'] = dict(alpha=0.3, facecolor='orange')
        elif 'props' in sig.parameters:
            span_args['props'] = dict(alpha=0.3, facecolor='orange')

        self.span_log = SpanSelector(self.ax_log, self.on_select, **span_args)
        self.span_lin = SpanSelector(self.ax_lin, self.on_select, **span_args)

        # --- Results text ---
        self.result_text = tk.Text(self, height=16)
        self.result_text.pack(padx=10, pady=5, fill='x')

        # --- Tooltips ---
        tooltips.ToolTip(file_entry, "Select a data file to load.")
        tooltips.ToolTip(model_menu, "Select a model to fit the data.")
        tooltips.ToolTip(fitmethod_menu, "Select the fitting method to use.")
        tooltips.ToolTip(fitButton, "Fit the selected range of data using the selected model and method.")
        tooltips.ToolTip(addFitButton, "Add the temporary fit to the list of fits.")
        tooltips.ToolTip(removeFitButton, "Remove the selected fit from the list.")
        tooltips.ToolTip(delTempFitButton, "Delete the temporary fit and reset the selection.")
        tooltips.ToolTip(interpolateButton, "Interpolate fits for the selected states (high and/or low) using a spline.")
        tooltips.ToolTip(saveFitButton, "Save the current fits to a text file with metadata.")
        tooltips.ToolTip(setRangeButton, "Set the fit range through input fields.")


    def set_manual_range(self):
        """Set a manual range for the selected data."""
        try:
            x0 = float(self.range_min_var.get())
            x1 = float(self.range_max_var.get())
            x0 = max(x0, np.min(self.current_x))
            x1 = min(x1, np.max(self.current_x))

            if x0 == x1:
                raise ValueError("Range must be non-zero.")

            self.selected_range = (min(x0, x1), max(x0, x1))
            self.range_label.config(text=f"[{x0:.3g}, {x1:.3g}]")
            self.temp_fit = None
            self.plot_data()
        except ValueError:
            messagebox.showerror("Error", "Invalid manual range. Please enter valid numeric values.")

    def browse_file(self):
        """Open a file dialog to select a data file for the fitting process."""
        path = filedialog.askopenfilename(title="Select data file",
                                          filetypes=[("TXT/QTJ files","*.txt *.qtj *.csv"), ("All files","*.*")])
        if path:
            self.file_var.set(path)
            try:
                x, y = load_txtfile(path)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data: {e}")
                return
            self.x_all = np.array(x)
            self.y_all = np.array(y)
            n = len(self.x_all)
            indices = np.arange(n)
            # Determine zero index (closest to zero)
            self.zero_index = int(np.argmin(np.abs(self.x_all)))
            # Determine positive peak and return
            pos_indices = np.where(self.x_all >= 0)[0]
            if pos_indices.size > 0:
                max_val = np.max(self.x_all[pos_indices])
                peak_idxs = np.where(self.x_all == max_val)[0]
                self.pos_peak_index = int(peak_idxs[0]) if peak_idxs.size > 0 else None
                # find return to zero after peak
                self.pos_return_index = None
                if self.pos_peak_index is not None and self.pos_peak_index+1 < n:
                    close_zero = np.where(np.isclose(self.x_all[self.pos_peak_index+1:], 0, atol=1e-6))[0]
                    if close_zero.size > 0:
                        self.pos_return_index = self.pos_peak_index + 1 + int(close_zero[0])
            else:
                self.pos_peak_index = None
                self.pos_return_index = None
            # Determine negative peak and return
            neg_indices = np.where(self.x_all <= 0)[0]
            if neg_indices.size > 0:
                min_val = np.min(self.x_all[neg_indices])
                peak_idxs = np.where(self.x_all == min_val)[0]
                self.neg_peak_index = int(peak_idxs[0]) if peak_idxs.size > 0 else None
                self.neg_return_index = None
                if self.neg_peak_index is not None and self.neg_peak_index+1 < n:
                    close_zero = np.where(np.isclose(self.x_all[self.neg_peak_index+1:], 0, atol=1e-6))[0]
                    if close_zero.size > 0:
                        self.neg_return_index = self.neg_peak_index + 1 + int(close_zero[0])
            else:
                self.neg_peak_index = None
                self.neg_return_index = None

            # Reset fits and temp
            self.fits.clear()
            self.temp_fit = None
            self.selected_range = None
            self.range_label.config(text="None")
            self.update_fit_list()
            # Apply subset and plot
            self.apply_subset()

    def apply_subset(self):
        """
        Apply the selected subset to the data and update the plot. Subset options:
        - All: Show all data
        - Positive x: Show data where x >= 0
        - Negative x: Show data where x <= 0
        - Positive LRS: Show data in the range of positive peak to zero
        - Positive HRS: Show data in the range of positive peak to positive return
        - Negative LRS: Show data in the range of negative peak to zero
        - Negative HRS: Show data in the range of negative peak to negative return
        """
        if self.x_all is None:
            return
        subset = self.subset_var.get()
        n = len(self.x_all)
        indices = np.arange(n)
        if subset == "All":
            mask = np.ones(n, dtype=bool)
        elif subset == "Positive x":
            mask = self.x_all >= 0
        elif subset == "Negative x":
            mask = self.x_all <= 0
        elif subset == "Positive LRS":
            if self.pos_peak_index is not None and self.zero_index is not None:
                start = self.zero_index
                end = self.pos_peak_index
                if start > end:
                    start, end = end, start
                mask = (indices >= start) & (indices <= end) & (self.x_all >= 0)
            else:
                messagebox.showwarning("Warning", "Cannot determine positive peak/zero for LRS; showing positive data.")
                mask = self.x_all >= 0
        elif subset == "Positive HRS":
            if self.pos_peak_index is not None and self.pos_return_index is not None:
                start = self.pos_peak_index
                end = self.pos_return_index
                if start > end:
                    start, end = end, start
                mask = (indices > start) & (indices <= end) & (self.x_all >= 0)
            else:
                messagebox.showwarning("Warning", "Cannot determine positive peak/return for HRS; showing positive data.")
                mask = self.x_all >= 0
        elif subset == "Negative HRS":
            if self.neg_peak_index is not None and self.zero_index is not None:
                start = self.zero_index
                end = self.neg_peak_index
                if start > end:
                    start, end = end, start
                mask = (indices >= start) & (indices <= end) & (self.x_all <= 0)
            else:
                messagebox.showwarning("Warning", "Cannot determine negative peak/zero for LRS; showing negative data.")
                mask = self.x_all <= 0
        elif subset == "Negative LRS":
            if self.neg_peak_index is not None and self.neg_return_index is not None:
                start = self.neg_peak_index
                end = self.neg_return_index
                if start > end:
                    start, end = end, start
                mask = (indices > start) & (indices <= end) & (self.x_all <= 0)
            else:
                messagebox.showwarning("Warning", "Cannot determine negative peak/return for HRS; showing negative data.")
                mask = self.x_all <= 0
        else:
            mask = np.ones(n, dtype=bool)
        self.current_x = self.x_all[mask]
        self.current_y = self.y_all[mask]
        # Clear selection and temp fit when subset changes
        self.selected_range = None
        self.temp_fit = None
        self.range_label.config(text="None")
        self.plot_data()

    def plot_data(self):
        """Plot the current data and fits in logarithmic and linear scale. Uses the selected subset and temporary fits. For negative values, the absolute value is plotted in the log plot. Zero values are replaced with a small value to avoid log(0)."""
        self.ax_log.clear()
        self.ax_lin.clear()
        self.ax_log.set_yscale('log')
        self.fig.suptitle("Logarithmic and Linear Visualization", fontsize=14)
        if self.current_x is not None:
            abs_y = np.abs(self.current_y)
            abs_y[abs_y == 0] = 1e-12  # keine Nullwerte im Log-Plot
            self.ax_log.scatter(self.current_x, abs_y, label='|I(U)|', s=20)
            self.ax_lin.scatter(self.current_x, self.current_y, label='I(U)', s=20)

        subset = self.subset_var.get()
        fits_to_plot = self.fits + ([self.temp_fit] if self.temp_fit else [])

        # Filter fits je nach Subset
        if subset.endswith("HRS"):
            fits_to_plot = [fit for fit in fits_to_plot if fit.get('state') == 'high']
        elif subset.endswith("LRS"):
            fits_to_plot = [fit for fit in fits_to_plot if fit.get('state') == 'low']
        # sonst: alle wie gehabt

        for fit in fits_to_plot:
            x_min, x_max = fit['range']

            # 1) Zunächst die x- und y-Werte des Fits bestimmen: entweder gespeicherte Arrays oder neu berechnen
            if 'fit_xs' in fit and 'fit_ys' in fit:
                xs_full = np.array(fit['fit_xs'])
                ys_full = np.array(fit['fit_ys'])
            else:
                xs_full = np.linspace(x_min, x_max, 200)
                try:
                    ys_full = fit['func'](xs_full, *fit['popt'])
                except Exception:
                    continue

            # 2) Subset-Maske auf xs_full anwenden
            if subset.startswith("Positive"):
                mask = xs_full >= 0
            elif subset.startswith("Negative"):
                mask = xs_full <= 0
            else:
                mask = np.ones_like(xs_full, dtype=bool)

            xs = xs_full[mask]
            ys = ys_full[mask]
            if xs.size < 2:
                continue

            # 3) Plotten im Log- und Linear-Plot
            abs_ys = np.abs(ys)
            abs_ys[abs_ys == 0] = 1e-12
            label = f"{fit.get('label','Fit')}: {fit['model']} ({fit.get('method','')}) [{x_min:.2g}, {x_max:.2g}]"
            self.ax_log.plot(xs, abs_ys, label=label)
            self.ax_lin.plot(xs, ys, label=label)
            self.ax_log.set_title("Logarithmic Plot")
            self.ax_lin.set_title("Linear Plot")

        for ax, ylabel in zip([self.ax_log, self.ax_lin], ['|I| (log)', 'I (linear)']):
            ax.set_xlabel('U')
            ax.set_ylabel(ylabel)
            ax.legend(loc='best')
            ax.grid(True)

        self.fig.tight_layout()
        self.canvas.draw()

    def on_select(self, xmin, xmax):
        """Callback for the SpanSelector to update the selected range."""
        if xmin == xmax:
            return
        x0, x1 = sorted([xmin, xmax])
        self.selected_range = (x0, x1)
        self.range_label.config(text=f"[{x0:.3g}, {x1:.3g}]")
        self.range_min_var.set(f"{x0:.5g}")
        self.range_max_var.set(f"{x1:.5g}")
        self.temp_fit = None
        self.plot_data()

    def fit_selection(self):
        """Perform a fit on the selected range of data using the selected model and method."""
        if self.current_x is None:
            messagebox.showerror("Error", "No data loaded.")
            return
        if not self.selected_range:
            messagebox.showerror("Error", "No range selected.")
            return
        model_key = self.model_var.get()
        if model_key not in models:
            messagebox.showerror("Error", "Invalid model selected.")
            return
        x_min, x_max = self.selected_range
        mask = (self.current_x >= x_min) & (self.current_x <= x_max)
        xs = self.current_x[mask]
        ys = self.current_y[mask]
        if len(xs) < 2:
            messagebox.showerror("Error", "Not enough data points in the selected range.")
            return
        func = models[model_key]['func']
        param_names = models[model_key].get('params', None)
        p0 = np.ones(len(param_names)) if param_names else None
        method = self.fitmethod_var.get()
        # If lmfit selected but not available, warn and fallback
        fit_warnings = ""
        try:
            with warnings.catch_warnings(record=True) as wlist:
                warnings.simplefilter("always")
                fit_result = perform_fit(xs, ys, model_key, method=method)
                # Alle Warnungen sammeln
                for w in wlist:
                    fit_warnings += f"Warning: {w.message}\n"
        except Exception as e:
            messagebox.showerror("Fit Error", f"Fitting failed ({method}): {e}")
            return
        
        state = "N/A"
        if self.subset_var.get().endswith("LRS"):
            state = "low"
        elif self.subset_var.get().endswith("HRS"):
            state = "high"

           
        self.temp_fit = {
            'model': model_key,
            'range': (x_min, x_max),
            'popt': fit_result.params,
            'pcov': fit_result.covar,
            'func': func,
            'method': method,
            'resultmessage': fit_result.fit_report(),
            'state': state
        }
        # Display results
        self.result_text.delete('1.0', tk.END)
        self.result_text.insert(tk.END, fit_result.fit_report())

        if fit_warnings:
            self.result_text.insert(tk.END, "\n--- Runtime-Warnings---\n")
            self.result_text.insert(tk.END, fit_warnings)
        fit_xs = np.linspace(x_min, x_max, 200)
        fit_ys = models[model_key]["func"](fit_xs, *fit_result.best_values.values())

        self.temp_fit['fit_xs'] = fit_xs
        self.temp_fit['fit_ys'] = fit_ys
        self.plot_data()

    def add_fit(self):
        if not self.temp_fit:
            messagebox.showerror("Error", "No temporary fit to add. Perform a fit first.")
            return
        self.fits.append(self.temp_fit.copy())
        self.temp_fit = None
        self.result_text.delete('1.0', tk.END)
        messagebox.showinfo("Added", "Fit added to plot.")
        self.update_fit_list()
        self.plot_data()

    def remove_fit(self):
        """Remove the selected fit from the fit listbox."""
        sel = self.fit_listbox.curselection()
        if not sel:
            messagebox.showerror("Error", "No fit selected to remove.")
            return
        idx = sel[0]
        if 0 <= idx < len(self.fits):
            del self.fits[idx]
            self.update_fit_list()
            self.plot_data()
            messagebox.showinfo("Removed", "Selected fit removed.")
        else:
            messagebox.showerror("Error", "Invalid selection.")

    def update_fit_list(self):
        """Update the fit listbox with the current fits."""
        self.fit_listbox.delete(0, tk.END)
        for idx, fit in enumerate(self.fits):
            x_min, x_max = fit['range']
            method = fit.get('method', '')
            desc = f"{idx+1}: {fit['model']} ({method}) [{x_min:.3g}, {x_max:.3g}]"
            self.fit_listbox.insert(tk.END, desc)

    def save_fits(self):
        """Save the current fits to a text file with metadata."""
        if not self.fits:
            messagebox.showerror("Error", "No fits to save.")
            return
        save_path = filedialog.asksaveasfilename(defaultextension='.txt', filetypes=[('Text files','*.txt'), ('All files','*.*')])
        if not save_path:
            return
        try:
            with open(save_path, 'w') as f:
                f.write(f"# Fits saved on {datetime.datetime.now().isoformat()}\n")
                for idx, fit in enumerate(self.fits, start=1):
                    model_key = fit['model']
                    method = fit.get('method', 'curve_fit')
                    x_min, x_max = fit['range']
                    f.write(f"Fit {idx}: Model: {model_key}, Method: {method}, Range: [{x_min:.5g}, {x_max:.5g}]\n")
                    f.write(f"{fit['resultmessage']}\n")
                    f.write("\n")
            messagebox.showinfo("Saved", f"Fits saved to {save_path}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save fits: {e}")
    
    def interpolate_fits(self):
        """Interpolate fits for the selected states (high and low) using a spline. Uses the selected polynomial degree, which is asked from the user.

        If only one state is available, interpolate only that state. If one state is already interpolated, allow the other to be interpolated later.
        """
        if len(self.fits) < 2:
            messagebox.showwarning("Interpolation", "At least two fits are required for interpolation.")
            return

        # Ask for polynomial degree
        k = simpledialog.askinteger("Polynomial Degree", "What degree should the interpolation polynomial have? (2-5)", minvalue=1, maxvalue=5)
        if not k:
            return

        # Helper to interpolate fits by state
        def interpolate_state(state_label):
            state_fits = [f for f in self.fits if f.get('state') == state_label]
            if len(state_fits) < 2:
                return None, None
            sorted_fits = sorted(state_fits, key=lambda f: f['range'][0])
            xs_all = []
            ys_all = []
            for fit in sorted_fits:
                xs = np.array(fit.get('fit_xs'))
                ys = np.array(fit.get('fit_ys'))
                xs_all.append(xs)
                ys_all.append(ys)
            xs_concat = np.concatenate(xs_all)
            ys_concat = np.concatenate(ys_all)
            sort_idx = np.argsort(xs_concat)
            xs_sorted = xs_concat[sort_idx]
            ys_sorted = ys_concat[sort_idx]
            return xs_sorted, ys_sorted

        # Check if already interpolated
        already_high = any(f.get('model') == 'interpolated_high' for f in self.fits)
        already_low = any(f.get('model') == 'interpolated_low' for f in self.fits)

        xs_high, ys_high = interpolate_state("high")
        xs_low, ys_low = interpolate_state("low")

        # If both are missing, nothing to do
        if xs_high is None and xs_low is None:
            messagebox.showwarning("Interpolation", "Not enough fits with state 'high' or 'low' for interpolation.")
            return

        # Interpolate high if possible and not already done
        if xs_high is not None and not already_high:
            xs_interp_high = np.linspace(xs_high[0], xs_high[-1], 500)
            try:
                spline_high = make_interp_spline(xs_high, ys_high, k=k)
                ys_high_interp = spline_high(xs_interp_high)
            except Exception as e:
                messagebox.showerror("Interpolation Error", f"Interpolation failed for state 'high': {e}")
                return
            interp_high = {
                'model': 'interpolated_high',
                'range': (xs_interp_high[0], xs_interp_high[-1]),
                'popt': [],
                'pcov': None,
                'func': None,
                'method': 'spline',
                'resultmessage': 'Int. fit (spline) for high',
                'fit_xs': xs_interp_high,
                'fit_ys': ys_high_interp,
                'label': 'Interpolated (high)',
                'state': 'high',
                'color': 'red'
            }
            self.fits.append(interp_high)

        # Interpolate low if possible and not already done
        if xs_low is not None and not already_low:
            xs_interp_low = np.linspace(xs_low[0], xs_low[-1], 500)
            try:
                spline_low = make_interp_spline(xs_low, ys_low, k=k)
                ys_low_interp = spline_low(xs_interp_low)
            except Exception as e:
                messagebox.showerror("Interpolation Error", f"Interpolation failed for state 'low': {e}")
                return
            interp_low = {
                'model': 'interpolated_low',
                'range': (xs_interp_low[0], xs_interp_low[-1]),
                'popt': [],
                'pcov': None,
                'func': None,
                'method': 'spline',
                'resultmessage': 'Int. fit (spline) for low',
                'fit_xs': xs_interp_low,
                'fit_ys': ys_low_interp,
                'label': 'Interpolated (low)',
                'state': 'low',
                'color': 'blue'
            }
            self.fits.append(interp_low)

        if (xs_high is not None and xs_low is not None) and (not already_high and not already_low):
            # If both are present and both are new, try to connect at endpoints
            # Find the interpolated fits just added
            interp_high = self.fits[-2]
            interp_low = self.fits[-1]
            # Average endpoints
            interp_high['fit_ys'][0] = interp_low['fit_ys'][0] = (interp_high['fit_ys'][0] + interp_low['fit_ys'][0]) / 2
            interp_high['fit_ys'][-1] = interp_low['fit_ys'][-1] = (interp_high['fit_ys'][-1] + interp_low['fit_ys'][-1]) / 2

        self.update_fit_list()
        self.plot_data()

    def update_latex_display(self, font_size=12, dpi=150):
        """Update the LaTeX display based on the selected model."""
        selected = self.model_var.get()
        model = models.get(selected, {})
        latex_string = model.get("latex", r"$\text{Keine Formel}$")

        # Altes Label entfernen
        if self.latex_label:
            self.latex_label.destroy()

        # Neues Label rendern und anzeigen
        fig, ax = plt.subplots(figsize=(0.01, 0.01))
        fig.patch.set_visible(False)
        ax.axis('off')

        # Setze den LaTeX-Text
        ax.text(0.5, 0.5, latex_string, fontsize=font_size, ha='center', va='center')

        # Speichere Bild in BytesIO
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi, transparent=True)
        buf.seek(0)
        plt.close(fig)

        # Lade Bild mit PIL und konvertiere für Tkinter
        image = Image.open(buf)
        photo = ImageTk.PhotoImage(image)

        # Erstelle Tkinter Label
        label = tk.Label(self.latex_frame, image=photo, bg='white')
        label.image = photo  # <- Referenz halten, sonst wird Bild gelöscht!
        self.latex_label = label
        self.latex_label.pack(side='left', padx=5)

    def on_closing(self):
        self.destroy()
        sys.exit()
        

# Start up
if __name__ == "__main__":
    app = FitApp()
    app.mainloop()