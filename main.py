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
    ttk       # Themed widgets
)
import tkinter.font as tkfont  # Font handling for Tkinter

# --- Local Application Imports ---
from models import models           # Model definitions. Contains the functions and parameters for fitting models.
from fit_logic import load_txtfile, perform_fit  # File parsing and fitting logic
from ParamDialog import info_messagebox, error_messagebox, ask_integer  # Dialog for setting parameter bounds interactively

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
        style = ttk.Style(self)
        # Set window background to white
        self.configure(bg="white")
        style.configure(".", background="white")  # Set ttk default background

        try:
            self.default_font = tkfont.Font(family="Segoe UI", size=font_size, weight="bold")
            self.out_fit_sec = tkfont.Font(family="Segoe UI", size=font_size)

        except Exception as e:
            print(f"Error: {e}")
            # Fallback
            self.default_font = tkfont.Font(family="Helvetica", size=font_size, weight="bold")
            self.out_fit_sec = tkfont.Font(family="Helvetica", size=font_size)

        style.configure("TButton", font=self.default_font)
        style.configure("TLabel", font=self.default_font)
        style.configure("TEntry", font=self.default_font)
        style.configure("TCombobox", font=self.default_font)
        style.configure("TFrame", font=self.default_font)
        style.configure("TText", font=self.default_font)
        style.configure("Option", font=self.out_fit_sec)

        style.configure("TLabel", foreground="black", background="white", font=self.default_font)
        # Remove border from Listbox and Scrollbar for a cleaner look
        self.option_add("*Listbox.borderWidth", 0)
        self.option_add("*Listbox.highlightThickness", 0)
        self.option_add("*Scrollbar.borderWidth", 0)
        self.option_add("*Scrollbar.highlightThickness", 0)

        # Remove border from result_text (Text widget)
        self.option_add("*Text.borderWidth", 0)
        self.option_add("*Text.highlightThickness", 0)


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

        # Rohdaten speichern
        self.raw_x = None
        self.raw_y = None
        
        # LaTeX rendering temporary variable
        self.latex_label = None

        # --- File selection ---
        file_frame = ttk.Frame(self)
        file_frame.pack(padx=10, pady=5, fill='x')
        ttk.Label(file_frame, text="Data file:").pack(side='left')
        self.file_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.file_var, width=60, font=self.out_fit_sec)
        file_entry.pack(side='left', padx=5)
        ttk.Button(file_frame, text="Browse...", command=self.browse_file).pack(side='left')

        # --- Area, thickness and temperature inputfields ---
        ttk.Label(file_frame, text="Area in um²:").pack(side='left', padx=(10, 5))
        self.new_area = tk.StringVar(value="625")
        ttk.Entry(file_frame, textvariable=self.new_area, width=5, font=self.out_fit_sec).pack(side='left', padx=5)
        ttk.Label(file_frame, text="Thickness in nm:").pack(side='left')
        self.new_thickness = tk.StringVar(value="10")
        ttk.Entry(file_frame, textvariable=self.new_thickness, width=5, font=self.out_fit_sec).pack(side='left', padx=5)
        ttk.Label(file_frame, text="Temperature in K:").pack(side='left')
        self.new_temperature = tk.StringVar(value="300")
        ttk.Entry(file_frame, textvariable=self.new_temperature, width=5, font=self.out_fit_sec).pack(side='left', padx=5)

        # Änderungen an Area/Thickness triggern Update
        self.new_area.trace_add("write", lambda *args: self.on_area_thickness_change())
        self.new_thickness.trace_add("write", lambda *args: self.on_area_thickness_change())

        # --- Subset selection ---
        subset_frame = ttk.Frame(self)
        subset_frame.pack(padx=10, pady=5, fill='x')
        ttk.Label(subset_frame, text="Subset:").pack(side='left')
        self.subset_var = tk.StringVar(value="All")
        subset_options = ["All", "Positive", "Negative","Positive Forward", "Positive Reverse", "Negative Forward", "Negative Reverse"]
        subset_menu = ttk.OptionMenu(subset_frame, self.subset_var, subset_options[0], *subset_options, command=lambda _: self.apply_subset())
        subset_menu["menu"].config(font=self.out_fit_sec)
        style.configure("Custom.TMenubutton", font=self.out_fit_sec)
        subset_menu.configure(style="Custom.TMenubutton")
        subset_menu.pack(side='left', padx=5)

        # --- Model selection and fit actions ---
        action_frame = ttk.Frame(self)
        action_frame.pack(padx=10, pady=5, fill='x')
        ttk.Label(action_frame, text="Model:").pack(side='left')
        self.model_var = tk.StringVar()
        model_names = list(models.keys())
        if model_names:
            self.model_var.set(model_names[0])
        model_menu = ttk.OptionMenu(action_frame, self.model_var, model_names[0] if model_names else "", *model_names, command= lambda _: self.update_latex_display())
        model_menu["menu"].config(font=self.out_fit_sec)
        style.configure("Custom.TMenubutton", font=self.out_fit_sec)
        model_menu.configure(style="Custom.TMenubutton")
        model_menu.pack(side='left', padx=5)

        # Fit method selection
        ttk.Label(action_frame, text="Fit Method:").pack(side='left', padx=(20, 0))
        self.fitmethod_var = tk.StringVar()
        method_options = ["leastsq", "least_squares", "ampgo", "nelder", "powell", "differential_evolution", "basinhopping"]
        self.fitmethod_var.set(method_options[0])
        fitmethod_menu = ttk.OptionMenu(action_frame, self.fitmethod_var, method_options[0], *method_options)
        fitmethod_menu["menu"].config(font=self.out_fit_sec)
        style.configure("Custom.TMenubutton", font=self.out_fit_sec)
        fitmethod_menu.configure(style="Custom.TMenubutton")
        fitmethod_menu.pack(side='left', padx=5)

        fitButton = ttk.Button(action_frame, text="Fit Selection", command=self.fit_selection)
        fitButton.pack(side='left', padx=5)

        addFitButton = ttk.Button(action_frame, text="Add Fit", command=self.add_fit)
        addFitButton.pack(side='left', padx=5)
        
        removeFitButton = ttk.Button(action_frame, text="Remove Fit", command=self.remove_fit)
        removeFitButton.pack(side='left', padx=5)
        
        saveFitButton = ttk.Button(action_frame, text="Save Fits", command=self.save_fits)
        saveFitButton.pack(side='left', padx=5)

        extrapolateButton = ttk.Button(action_frame, text="Extrapolate Fit", command=self.extrapolate_fit)
        extrapolateButton.pack(side='left', padx=5)

        # --- Selected range display ---
        range_frame = ttk.Frame(self)
        range_frame.pack(padx=10, pady=5, fill='x')
        ttk.Label(range_frame, text="Selected Range:").pack(side='left')
        self.range_label = ttk.Label(range_frame, text="None")
        self.range_label.pack(side='left', padx=5)

        # Input for manual range setting
        ttk.Label(range_frame, text="From:").pack(side='left', padx=(10,0))
        self.range_min_var = tk.StringVar()
        ttk.Entry(range_frame, textvariable=self.range_min_var, width=10, font=self.out_fit_sec).pack(side='left')

        ttk.Label(range_frame, text="To:").pack(side='left', padx=(10,0))
        self.range_max_var = tk.StringVar()
        ttk.Entry(range_frame, textvariable=self.range_max_var, width=10, font=self.out_fit_sec).pack(side='left')

        setRangeButton = ttk.Button(range_frame, text="Set Range", command=self.set_manual_range)
        setRangeButton.pack(side='left', padx=5)

        delTempFitButton = ttk.Button(range_frame, text="Delete Temp. Fit", command=self.apply_subset)
        delTempFitButton.pack(side='left', padx=5)
        

        # --- Fit list display ---
        fit_list_frame = ttk.Frame(self)
        fit_list_frame.pack(padx=10, pady=5, fill='x', expand=False)
        ttk.Label(fit_list_frame, text="Fits:").pack(side='top', anchor='w')
        self.fit_listbox = tk.Listbox(fit_list_frame, height=3, font=self.out_fit_sec, selectmode='single')
        self.fit_listbox.pack(side='left', fill='x', expand=True)
        scrollbar = ttk.Scrollbar(fit_list_frame, orient='vertical', command=self.fit_listbox.yview)
        scrollbar.pack(side='right', fill='y')
        self.fit_listbox.config(yscrollcommand=scrollbar.set)
        
        # --- Horizontal separator ---
        separator = ttk.Separator(self, orient='horizontal')
        separator.pack(fill='x', padx=10, pady=5)
        
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
            span_args['rectprops'] = dict(alpha=0.3, facecolor='blue')
        elif 'props' in sig.parameters:
            span_args['props'] = dict(alpha=0.3, facecolor='blue')

        self.span_log = SpanSelector(self.ax_log, self.on_select, **span_args)
        self.span_lin = SpanSelector(self.ax_lin, self.on_select, **span_args)
        
        # --- Horizontal separator ---
        separator = ttk.Separator(self, orient='horizontal')
        separator.pack(fill='x', padx=10, pady=5)
        
        # --- Fit Result Header ---
        header_font = tkfont.Font(family="Segoe UI", size=font_size + 4, weight="bold")
        header_label = tk.Label(self, text="Fit Result", font=header_font, bg="white")
        header_label.pack(padx=10, pady=(5, 0), anchor='w')
        
        # --- Results text (side-by-side) ---
        result_frame = ttk.Frame(self)
        result_frame.pack(padx=10, pady=5, fill='x')

        self.result_text = tk.Text(result_frame, height=16, wrap='word', font=self.out_fit_sec, width=40)
        self.result_text.pack(side='left')
        self.result_text.bind("<Key>", lambda e: "break")  # Alle Tastatureingaben blockieren
        self.result_text.config(cursor="arrow")  # Cursor auf Pfeil setzen, um Eingabe zu verhindern

        self.result_text2 = tk.Text(result_frame, height=16, wrap='word', font=self.out_fit_sec, width=40)
        self.result_text2.pack(side='left', fill='x', expand=True)
        self.result_text2.bind("<Key>", lambda e: "break")
        self.result_text2.config(cursor="arrow")


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
            error_messagebox("Invalid Range", "Please enter valid numeric values for the range.", font=self.out_fit_sec)
            return

    def browse_file(self):
        """Open a file dialog to select a data file for the fitting process."""
        path = filedialog.askopenfilename(title="Select data file",
                                          filetypes=[("TXT/QTJ files","*.txt *.qtj *.csv"), ("All files","*.*")])
        if path:
            self.file_var.set(path)
            try:
                x, y = load_txtfile(path)
                self.raw_x = np.array(x)
                self.raw_y = np.array(y)
                self.update_scaled_data()
            except Exception as e:
                error_messagebox("File Error", f"Could not load file: {e}", font=self.out_fit_sec)
                return
            # Reset fits and temp
            self.fits.clear()
            self.temp_fit = None
            self.selected_range = None
            self.range_label.config(text="None")
            self.update_fit_list()
            # Apply subset and plot
            self.apply_subset()

    def on_area_thickness_change(self):
        """Callback wenn Area oder Thickness geändert werden."""
        if self.raw_x is not None and self.raw_y is not None:
            self.update_scaled_data()
            self.apply_subset()

    def update_scaled_data(self):
        """Berechnet x_all und y_all aus raw_x/raw_y und aktuellen Area/Thickness."""
        try:
            area_in_um_squared = float(self.new_area.get())
            thickness_in_nm = float(self.new_thickness.get())
            x = np.array(self.raw_x) / (thickness_in_nm * 1e-9)
            y = np.array(self.raw_y) / (area_in_um_squared * 1e-12)
            self.x_all = x
            self.y_all = y
            # Peak/Zero-Indices neu berechnen
            n = len(self.x_all)
            self.zero_index = int(np.argmin(np.abs(self.x_all)))
            pos_indices = np.where(self.x_all >= 0)[0]
            if pos_indices.size > 0:
                max_val = np.max(self.x_all[pos_indices])
                peak_idxs = np.where(self.x_all == max_val)[0]
                self.pos_peak_index = int(peak_idxs[0]) if peak_idxs.size > 0 else None
                self.pos_return_index = None
                if self.pos_peak_index is not None and self.pos_peak_index+1 < n:
                    close_zero = np.where(np.isclose(self.x_all[self.pos_peak_index+1:], 0, atol=1e-6))[0]
                    if close_zero.size > 0:
                        self.pos_return_index = self.pos_peak_index + 1 + int(close_zero[0])
            else:
                self.pos_peak_index = None
                self.pos_return_index = None
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
        except Exception:
            pass

    def apply_subset(self):
        """
        Apply the selected subset to the data and update the plot. Subset options:
        - All: Show all data
        - Positive: Show data where x >= 0
        - Negative: Show data where x <= 0
        - Positive Forward: Data from zero to positive peak (increasing x >= 0)
        - Positive Reverse: Data from positive peak to positive return (decreasing x >= 0)
        - Negative Forward: Data from zero to negative peak (decreasing x <= 0)
        - Negative Reverse: Data from negative peak to negative return (increasing x <= 0)
        """
        if self.x_all is None:
            return
        subset = self.subset_var.get()
        n = len(self.x_all)
        indices = np.arange(n)
        if subset == "All":
            mask = np.ones(n, dtype=bool)
        elif subset == "Positive":
            mask = self.x_all >= 0
        elif subset == "Negative":
            mask = self.x_all <= 0
        elif subset == "Positive Forward":
            if self.zero_index is not None and self.pos_peak_index is not None:
                start = self.zero_index
                end = self.pos_peak_index
                if start > end:
                    start, end = end, start
                mask = (indices >= start) & (indices <= end) & (self.x_all >= 0)
            else:
                error_messagebox("Warning", "Cannot determine zero/positive peak for Positive Forward; showing positive data.", font=self.out_fit_sec)
                mask = self.x_all >= 0
        elif subset == "Positive Reverse":
            if self.pos_peak_index is not None and self.pos_return_index is not None:
                start = self.pos_peak_index
                end = self.pos_return_index
                if start > end:
                    start, end = end, start
                mask = (indices > start) & (indices <= end) & (self.x_all >= 0)
            else:
                error_messagebox("Warning", "Cannot determine positive peak/return for Positive Reverse; showing positive data.", font=self.out_fit_sec)
                mask = self.x_all >= 0
        elif subset == "Negative Forward":
            if self.zero_index is not None and self.neg_peak_index is not None:
                start = self.zero_index
                end = self.neg_peak_index
                if start > end:
                    start, end = end, start
                mask = (indices >= start) & (indices <= end) & (self.x_all <= 0)
            else:
                error_messagebox("Warning", "Cannot determine zero/negative peak for Negative Forward; showing negative data.", font=self.out_fit_sec)
                mask = self.x_all <= 0
        elif subset == "Negative Reverse":
            if self.neg_peak_index is not None and self.neg_return_index is not None:
                start = self.neg_peak_index
                end = self.neg_return_index
                if start > end:
                    start, end = end, start
                mask = (indices > start) & (indices <= end) & (self.x_all <= 0)
            else:
                error_messagebox("Warning", "Cannot determine negative peak/return for Negative Reverse; showing negative data.", font=self.out_fit_sec)
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
            self.ax_log.scatter(self.current_x, abs_y, label='J-Data', s=20)
            self.ax_lin.scatter(self.current_x, self.current_y, label='J-Data', s=20)

        subset = self.subset_var.get()
        fits_to_plot = self.fits + ([self.temp_fit] if self.temp_fit else [])

        # Filter fits je nach Subset
        if subset.endswith("Reverse"):
            fits_to_plot = [fit for fit in fits_to_plot if fit.get('state') == 'high']
        elif subset.endswith("Forward"):
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

        # Achsenbeschriftungen und Legende in LaTeX, falls aktiviert
        use_latex = plt.rcParams.get("text.usetex", False)
        # Raw-TeX (kein \mathrm), Nicht-Latex-Variante als Klartext
        ylabels_latex = [r'$|J|~[A/m^2]$', r'$J~[A/m^2]$']
        ylabels_plain = ['|J| [A/m^2]', 'J [A/m^2]']
        xlabels_latex = r'$E~[V/m]$'
        xlabel_plain = 'E [V/m]'
        for ax, ylabel_latex, ylabel_plain in zip([self.ax_log, self.ax_lin], ylabels_latex, ylabels_plain):
            ax.set_xlabel(xlabels_latex if use_latex else xlabel_plain, fontsize=15)
            ax.set_ylabel(ylabel_latex if use_latex else ylabel_plain, fontsize=15)
            ax.tick_params(axis='both', labelsize=14)
            ax.legend(loc='best', fontsize=13)
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
            error_messagebox("Error", "No data loaded. Please load a data file first.", font=self.out_fit_sec)
            return
        if not self.selected_range:
            error_messagebox("Error", "No range selected.", font=self.out_fit_sec)
            return
        model_key = self.model_var.get()
        if model_key not in models:
            error_messagebox("Error", "Invalid model selected.", font=self.out_fit_sec)
            return
        x_min, x_max = self.selected_range
        mask = (self.current_x >= x_min) & (self.current_x <= x_max)
        xs = self.current_x[mask]
        ys = self.current_y[mask]
        if len(xs) < 2:
            error_messagebox("Error", "Not enough data points in the selected range.", font=self.out_fit_sec)
            return
        func = models[model_key]['func']
        method = self.fitmethod_var.get()
        # If lmfit selected but not available, warn and fallback
        fit_warnings = ""
        try:
            with warnings.catch_warnings(record=True) as wlist:
                warnings.simplefilter("always")
                # Perform the fit using the selected method
                fit_result = perform_fit(xs, ys, model_key, method=method, T=self.new_temperature.get(), secFont=self.out_fit_sec)
                if fit_result is None:
                    return
                for w in wlist:
                    fit_warnings += f"Warning: {w.message}\n"
        except Exception as e:
            error_messagebox("Fit Error", f"Fitting failed ({method}): {e}", font=self.out_fit_sec)
            return
        
        state = "N/A"
        if self.subset_var.get().endswith("Forward"):
            state = "low"
        elif self.subset_var.get().endswith("Reverse"):
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
        self.display_fit_result(fit_result)

        if fit_warnings:
            self.result_text2.insert(tk.END, "\n--- Runtime-Warnings---\n")
            self.result_text2.insert(tk.END, fit_warnings)
        fit_xs = np.linspace(x_min, x_max, 200)
        fit_ys = models[model_key]["func"](fit_xs, *fit_result.best_values.values())

        self.temp_fit['fit_xs'] = fit_xs
        self.temp_fit['fit_ys'] = fit_ys
        self.plot_data()

    def add_fit(self):
        if not self.temp_fit:
            error_messagebox("Error", "No temporary fit to add. Perform a fit first.", font=self.out_fit_sec)
            return
        self.fits.append(self.temp_fit.copy())
        self.temp_fit = None
        self.result_text.delete('1.0', tk.END)
        info_messagebox("Success", "Fit added to plot.", font=self.out_fit_sec, width=350, height=100)
        self.update_fit_list()
        self.plot_data()

    def remove_fit(self):
        """Remove the selected fit from the fit listbox."""
        sel = self.fit_listbox.curselection()
        if not sel:
            error_messagebox("Error", "No fit selected to remove.", font=self.out_fit_sec, width=300, height=120)
            return
        idx = sel[0]
        if 0 <= idx < len(self.fits):
            del self.fits[idx]
            self.update_fit_list()
            self.plot_data()
            info_messagebox("Removed", "Selected fit removed.", font=self.out_fit_sec, width=300, height=120)
        else:
            error_messagebox("Error", "Invalid selection.", font=self.out_fit_sec, width=300, height=120)

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
            error_messagebox("Error", "No fits to save.")
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
            info_messagebox("Saved", f"Fits saved to {save_path}")
        except Exception as e:
            error_messagebox("Save Error", f"Failed to save fits: {e}")

    def display_fit_result(self, fit_result):
        
        self.result_text.configure(state='normal')
        self.result_text2.configure(state='normal')

        self.result_text.delete('1.0', tk.END)
        self.result_text2.delete('1.0', tk.END)

        self.result_text.configure(font=self.out_fit_sec)
        self.result_text2.configure(font=self.out_fit_sec)

        # Split fit report into header and variables
        report = fit_result.fit_report()
        report = report.replace("[[", "[")
        report = report.replace("]]", "]") 
        lines = report.splitlines()

        header_lines = []
        variable_lines = []
        writing_variables = False

        for line in lines:
            if line.strip().startswith('[Variables]'):
                writing_variables = True
            if writing_variables:
                variable_lines.append(line)
            else:
                header_lines.append(line)

        # Add model and method info to header
        self.result_text.insert(tk.END, '\n'.join(header_lines))
        self.result_text2.insert(tk.END, '\n'.join(variable_lines))

        # Optional: Make headers bold
        for text_widget in [self.result_text, self.result_text2]:
            text_widget.tag_configure('header', font=self.default_font)

            for i, line in enumerate(text_widget.get("1.0", tk.END).splitlines(), 1):
                if line.strip().startswith('['):
                    text_widget.tag_add('header', f"{i}.0", f"{i}.end")

            text_widget.configure(state='disabled')

    def extrapolate_fit(self):
        """Extrapolate the selected fit over the entire data range and add it as a new fit."""
        sel = self.fit_listbox.curselection()
        if not sel:
            error_messagebox("Error", "Please select a fit from the list first.", font=self.out_fit_sec)
            return

        idx = sel[0]
        if idx >= len(self.fits):
            error_messagebox("Error", "Invalid fit index.", font=self.out_fit_sec)
            return

        fit = self.fits[idx]
        func = fit.get('func')
        popt = fit.get('popt')

        if func is None or popt is None:
            error_messagebox("Error", "The selected fit does not contain a function or parameters.", font=self.out_fit_sec)
            return

        # New x over the entire data range
        x_min = np.min(self.x_all)
        x_max = np.max(self.x_all)
        xs = np.linspace(x_min, x_max, 500)

        try:
            ys = func(xs, *[p.value for p in popt.values()])
        except Exception as e:
                error_messagebox("Error", f"Extrapolation failed: {e}", font=self.out_fit_sec)
                return

        extrap_fit = {
           'model': fit['model'],
           'range': (x_min, x_max),
           'popt': popt,
           'pcov': fit.get('pcov'),
           'func': func,
           'method': fit.get('method'),
           'resultmessage': fit.get('resultmessage', 'Extrapolated fit'),
           'fit_xs': xs,
           'fit_ys': ys,
           'label': f"Extrapolated ({fit['model']})",
           'state': fit.get('state', 'N/A')
           }

        self.fits.append(extrap_fit)
        self.update_fit_list()
        self.plot_data()
        info_messagebox("Success", "Fit has been extrapolated over the entire range.", font=self.out_fit_sec)

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
