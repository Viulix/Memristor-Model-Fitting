# --- Standard Library Imports ---
import io                          # For byte stream handling (e.g. image buffers) --> for LaTeX rendering
import inspect                     # For introspection (e.g. argument inspection)
import warnings                    # To control or suppress warnings
import datetime                    # For handling timestamps and time formatting
import sys                         # System-specific parameters and functions
import os                          # For file path operations
import json                        # For saving and loading configuration

# --- Third-Party Imports ---
import numpy as np                 # Numerical computing
from PIL import Image, ImageTk     # Image handling for Tkinter --> for rendering LaTeX equations
import matplotlib.pyplot as plt            # Plotting
from matplotlib.widgets import SpanSelector                      # Interactive span selector
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Embedding Matplotlib in Tkinter

# --- GUI: Tkinter Modules ---
import tkinter as tk
import re
from tkinter import (
    filedialog,     # File dialogs (open/save)
    ttk       # Themed widgets
)
import tkinter.font as tkfont  # Font handling for Tkinter

# --- Local Application Imports ---
from models import models           # Model definitions. Contains the functions and parameters for fitting models.
from fit_logic import load_txtfile, performFit  # File parsing and fitting logic
from ParamDialog import info_messagebox, error_messagebox  # Dialog for setting parameter bounds interactively

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
        self.config_file = "config.json"
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

        style.configure("TButton", font=self.default_font, padding=4)
        style.configure("TLabel", font=self.default_font)
        style.configure("TEntry", font=self.default_font, padding=4)
        style.configure("TCombobox", font=self.default_font, padding=4)
        style.configure("TFrame", font=self.default_font, padding=4)
        style.configure("TText", font=self.default_font, padding=4)
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

        # Initialize icons
        self.icons = {}
        self.load_icons()

        # Plot scale variables
        self.left_scale_var = tk.StringVar(value="log")
        self.right_scale_var = tk.StringVar(value="log")

        # --- File selection ---
        file_frame = ttk.Frame(self)
        file_frame.pack(padx=10, pady=5, fill='x')
        ttk.Label(file_frame, text="Data file:").pack(side='left')
        self.file_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.file_var, width=60, font=self.out_fit_sec)
        file_entry.pack(side='left', padx=5)
        ttk.Button(file_frame, text="Browse...", command=self.browse_file,
                  image=self.icons.get('browse'), compound='left').pack(side='left')

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

        # Update Area/Thickness triggers
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

        # --- Combination fit selection (top right) ---
        combo_frame = ttk.Frame(action_frame)
        combo_frame.pack(side='right', padx=5)
        

        # Checkbox to enable combination fitting
        self.enable_combo_var = tk.BooleanVar(value=False)
        combo_label_frame = ttk.Frame(combo_frame)
        combo_label_frame.pack(side='top', anchor='e')
        
        ttk.Label(combo_label_frame, text="Combination Fit:", font=self.default_font).pack(side='left')
        combo_checkbox = ttk.Checkbutton(combo_label_frame, text="Active", variable=self.enable_combo_var, command=self.update_latex_display)
        combo_checkbox.pack(side='left', padx=(5, 240))
        
        # Frame for the two model selectors
        combo_selectors_frame = ttk.Frame(combo_frame)
        combo_selectors_frame.pack(side='top', anchor='e', pady=2)
        
        # First model selector
        ttk.Label(combo_selectors_frame, text="Model 1:").pack(side='left')
        self.combo_model1_var = tk.StringVar()
        if model_names:
            self.combo_model1_var.set(model_names[0])
        combo1_menu = ttk.OptionMenu(combo_selectors_frame, self.combo_model1_var, model_names[0] if model_names else "", *model_names, command=lambda _: self.update_latex_display())
        combo1_menu["menu"].config(font=self.out_fit_sec)
        combo1_menu.configure(style="Custom.TMenubutton")
        combo1_menu.pack(side='left', padx=2)
        
        # Second model selector
        ttk.Label(combo_selectors_frame, text="Model 2:").pack(side='left')
        self.combo_model2_var = tk.StringVar()
        if len(model_names) > 1:
            self.combo_model2_var.set(model_names[1])
        elif model_names:
            self.combo_model2_var.set(model_names[0])
        combo2_menu = ttk.OptionMenu(combo_selectors_frame, self.combo_model2_var, model_names[0] if model_names else "", *model_names, command=lambda _: self.update_latex_display())
        combo2_menu["menu"].config(font=self.out_fit_sec)
        combo2_menu.configure(style="Custom.TMenubutton")
        combo2_menu.pack(side='left', padx=2)

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

        fitButton = ttk.Button(action_frame, text="Fit Selection", command=self.fit_selection, 
                              image=self.icons.get('fit'), compound='left')
        fitButton.pack(side='left', padx=5)

        addFitButton = ttk.Button(action_frame, text="Add Fit", command=self.add_fit,
                                 image=self.icons.get('add'), compound='left')
        addFitButton.pack(side='left', padx=5)
        
        removeFitButton = ttk.Button(action_frame, text="Remove Fit", command=self.remove_fit,
                                   image=self.icons.get('remove'), compound='left')
        removeFitButton.pack(side='left', padx=5)

        exportDataButton = ttk.Button(action_frame, text="Export Fitdata", command=self.export_fit_data,
                                    image=self.icons.get('export'), compound='left')
        exportDataButton.pack(side='left', padx=5)

        extrapolateButton = ttk.Button(action_frame, text="Extrapolate Fit", command=self.extrapolate_fit,
                                     image=self.icons.get('extrapolate'), compound='left')
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

        setRangeButton = ttk.Button(range_frame, text="Set Range", command=self.set_manual_range,
                                  image=self.icons.get('range'), compound='left')
        setRangeButton.pack(side='left', padx=5)

        delTempFitButton = ttk.Button(range_frame, text="Delete Temp. Fit", command=self.apply_subset,
                                    image=self.icons.get('delete'), compound='left')
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
        self.fig, (self.ax_left, self.ax_right) = plt.subplots(1, 2, figsize=(12, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(padx=10, pady=10, fill='both', expand=True)

        # Initialize SpanSelectors auf beiden Plots
        sig = inspect.signature(SpanSelector)
        span_args = {'direction': 'horizontal', 'useblit': True}
        if 'rectprops' in sig.parameters:
            span_args['rectprops'] = dict(alpha=0.3, facecolor='blue')
        elif 'props' in sig.parameters:
            span_args['props'] = dict(alpha=0.3, facecolor='blue')

        # Use separate callbacks for each plot to handle different x-axis units
        self.span_left = SpanSelector(self.ax_left, self.on_select_E, **span_args)
        self.span_right = SpanSelector(self.ax_right, self.on_select_U, **span_args)
        
        # --- Plot scale controls ---
        scale_frame = ttk.Frame(self)
        scale_frame.pack(padx=10, pady=5, fill='x')
        
        ttk.Label(scale_frame, text="Left Plot Scale:").pack(side='left')
        left_scale_menu = ttk.OptionMenu(scale_frame, self.left_scale_var, "log", "linear", "log", command=lambda _: self.update_plot_scales())
        left_scale_menu["menu"].config(font=self.out_fit_sec)
        left_scale_menu.configure(style="Custom.TMenubutton")
        left_scale_menu.pack(side='left', padx=5)
        
        ttk.Label(scale_frame, text="Right Plot Scale:").pack(side='left', padx=(20, 0))
        right_scale_menu = ttk.OptionMenu(scale_frame, self.right_scale_var, "log", "linear", "log", command=lambda _: self.update_plot_scales())
        right_scale_menu["menu"].config(font=self.out_fit_sec)
        right_scale_menu.configure(style="Custom.TMenubutton")
        right_scale_menu.pack(side='left', padx=5)
        self.update_plot_scales()
        
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

        # Load configuration at the end of initialization
        self.load_config()

    def update_plot_scales(self):
        """Update the scales of both plots based on the selected options."""
        left_scale = self.left_scale_var.get()
        right_scale = self.right_scale_var.get()
        
        self.ax_left.set_yscale(left_scale)
        self.ax_right.set_yscale(right_scale)
        
        self.plot_data()

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

    def load_data_from_path(self, path):
        """Loads data from a given file path and updates the application state."""
        if not path or not os.path.exists(path):
            if path: # Only show error if path was provided but not found
                error_messagebox("File Error", f"Could not find file: {path}", font=self.out_fit_sec)
            self.raw_x, self.raw_y = None, None
            self.file_var.set("")
            return False

        self.file_var.set(path)
        try:
            x, y = load_txtfile(path)
            self.raw_x = np.array(x)
            self.raw_y = np.array(y)
            self.update_scaled_data()
        except Exception as e:
            error_messagebox("File Error", f"Could not load file: {e}", font=self.out_fit_sec)
            self.raw_x, self.raw_y = None, None
            return False

        # Reset fits and temp
        self.fits.clear()
        self.temp_fit = None
        self.selected_range = None
        self.range_label.config(text="None")
        self.update_fit_list()
        # Apply subset and plot
        self.apply_subset()
        return True

    def browse_file(self):
        """Open a file dialog to select a data file for the fitting process."""
        path = filedialog.askopenfilename(title="Select data file",
                                          filetypes=[("TXT/QTJ files","*.txt *.qtj *.csv"), ("All files","*.*")])
        if path:
            self.load_data_from_path(path)

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
            """Plot the current data and fits in both plots with configurable scales."""
            self.ax_left.clear()
            self.ax_right.clear()

            # Skalierungsfaktoren für die Umrechnung einmalig am Anfang berechnen
            try:
                # Umrechnung von E [V/m] -> U [V] durch Multiplikation mit der Dicke in Metern
                thickness_m = float(self.new_thickness.get()) * 1e-9  # Annahme: Eingabe in nm
                # Umrechnung von J [A/m^2] -> I [A] durch Multiplikation mit der Fläche in m^2
                area_m2 = float(self.new_area.get()) * 1e-12      # Annahme: Eingabe in µm^2
            except (ValueError, TypeError):
                # Fallback, falls die Eingabefelder leer oder ungültig sind
                thickness_m = 1e-9  # Default 1 nm
                area_m2 = 1e-12     # Default 1 µm²
            
            # Skalierung der y-Achsen (log/linear) anwenden
            left_scale = self.left_scale_var.get()
            right_scale = self.right_scale_var.get()
            self.ax_left.set_yscale(left_scale)
            self.ax_right.set_yscale(right_scale)
            
            # Titel setzen
            left_title = f"J/E ({left_scale.capitalize()} Scale)"
            right_title = f"I/U ({right_scale.capitalize()} Scale)"
            self.fig.suptitle(f"{left_title} and {right_title}", fontsize=14)
            
            if self.current_x is not None:
                y_left = self.current_y.copy()
                y_right = self.current_y.copy()
                
                # Für log-Skala Absolutwerte verwenden und Nullen behandeln
                if left_scale == 'log':
                    y_left = np.abs(y_left)
                    y_left[y_left == 0] = 1e-12
                if right_scale == 'log':
                    y_right = np.abs(y_right)
                    y_right[y_right == 0] = 1e-12
                
                # Linker Plot (J vs. E)
                self.ax_left.scatter(self.current_x, y_left, label='J-Data', s=20)

                # Rechter Plot (I vs. U) mit korrigierter Umrechnung
                x_volts = self.current_x * thickness_m
                y_amps = y_right * area_m2
                self.ax_right.scatter(x_volts, y_amps, label='I-Data', s=20)

            # Plotten der Fits
            subset = self.subset_var.get()
            fits_to_plot = self.fits + ([self.temp_fit] if self.temp_fit else [])

            if subset.endswith("Reverse"):
                fits_to_plot = [fit for fit in fits_to_plot if fit.get('state') == 'high']
            elif subset.endswith("Forward"):
                fits_to_plot = [fit for fit in fits_to_plot if fit.get('state') == 'low']

            for fit in fits_to_plot:
                x_min, x_max = fit['range']

                if 'fit_xs' in fit and 'fit_ys' in fit:
                    xs_full = np.array(fit['fit_xs'])
                    ys_full = np.array(fit['fit_ys'])
                else:
                    xs_full = np.linspace(x_min, x_max, 200)
                    try:
                        ys_full = fit['func'](xs_full, *fit['popt'])
                    except Exception:
                        continue

                # Subset-Maske anwenden
                mask = np.ones_like(xs_full, dtype=bool)
                if subset.startswith("Positive"):
                    mask = xs_full >= 0
                elif subset.startswith("Negative"):
                    mask = xs_full <= 0
                
                xs = xs_full[mask]
                ys = ys_full[mask]
                if xs.size < 2:
                    continue

                # y-Werte für die Plots basierend auf der Skala vorbereiten
                ys_left = ys.copy()
                ys_right = ys.copy()
                
                if left_scale == 'log':
                    ys_left = np.abs(ys_left)
                    ys_left[ys_left == 0] = 1e-12
                if right_scale == 'log':
                    ys_right = np.abs(ys_right)
                    ys_right[ys_right == 0] = 1e-12

                label = f"{fit.get('label','Fit')}: {fit['model']} ({fit.get('method','')}) [{x_min:.2g}, {x_max:.2g}]"

                # Linken Fit plotten (J vs. E)
                self.ax_left.plot(xs, ys_left, label=label)
                
                # Rechten Fit plotten (I vs. U) mit korrigierter Umrechnung
                xs_volts = xs * thickness_m
                ys_amps = ys_right * area_m2
                self.ax_right.plot(xs_volts, ys_amps, label=label)

            # Achsenbeschriftungen und Layout
            use_latex = plt.rcParams.get("text.usetex", False)
            
            ylabel_left = r'$|J|~[A/m^2]$' if left_scale == 'log' else r'$J~[A/m^2]$'
            ylabel_left_plain = '|J| [A/m^2]' if left_scale == 'log' else 'J [A/m^2]'
            
            ylabel_right = r'$|I|~[A]$' if right_scale == 'log' else r'$I~[A]$'
            ylabel_right_plain = '|I| [A]' if right_scale == 'log' else 'I [A]'
            
            xlabel_left = r'$E~[V/m]$'
            xlabel_left_plain = 'E [V/m]'
            xlabel_right = r'$U~[V]$'
            xlabel_right_plain = 'U [V]'
            
            self.ax_left.set_xlabel(xlabel_left if use_latex else xlabel_left_plain, fontsize=15)
            self.ax_left.set_ylabel(ylabel_left if use_latex else ylabel_left_plain, fontsize=15)
            
            self.ax_right.set_xlabel(xlabel_right if use_latex else xlabel_right_plain, fontsize=15)
            self.ax_right.set_ylabel(ylabel_right if use_latex else ylabel_right_plain, fontsize=15)
            
            for ax in [self.ax_left, self.ax_right]:
                ax.tick_params(axis='both', labelsize=14)
                ax.legend(loc='best', fontsize=13)
                ax.grid(True)

            # Achsenbereiche explizit setzen
            if self.current_x is not None:
                # Left Plot: E-Field Area
                self.ax_left.set_xlim(np.min(self.current_x) * 1.1, np.max(self.current_x) * 1.1)

                # Right Plot: Voltage Area
                x_volts_range = self.current_x * thickness_m
                self.ax_right.set_xlim(np.min(x_volts_range) * 1.1, np.max(x_volts_range) * 1.1)

            self.fig.tight_layout()
            self.canvas.draw()

    def on_select_E(self, xmin, xmax):
        """Callback for the SpanSelector on the E-field plot (left)."""
        self.update_selection(xmin, xmax)

    def on_select_U(self, xmin, xmax):
        """Callback for the SpanSelector on the Voltage plot (right)."""
        try:
            thickness_nm = float(self.new_thickness.get())
            if thickness_nm == 0:
                error_messagebox("Error", "Thickness cannot be zero.", font=self.out_fit_sec)
                return
            thickness_m = thickness_nm * 1e-9
            
            # Convert selected Voltage range (U) back to Electric Field range (E)
            e_min = xmin / thickness_m
            e_max = xmax / thickness_m
            self.update_selection(e_min, e_max)

        except (ValueError, ZeroDivisionError) as e:
            error_messagebox("Error", f"Could not convert voltage to E-field. Invalid thickness? Error: {e}", font=self.out_fit_sec)

    def update_selection(self, xmin, xmax):
        """Unified method to update the application state with a new selected range."""
        if xmin == xmax:
            return
        x0, x1 = sorted([xmin, xmax])
        self.selected_range = (x0, x1)
        self.range_label.config(text=f"[{x0:.3g}, {x1:.3g}]")
        self.range_min_var.set(f"{x0:.5g}")
        self.range_max_var.set(f"{x1:.5g}")
        self.temp_fit = None
        self.plot_data()

    def add_fit(self):
        if not self.temp_fit:
            error_messagebox("Error", "No temporary fit to add. Perform a fit first.", font=self.out_fit_sec)
            return
        self.fits.append(self.temp_fit.copy())
        self.temp_fit = None
        self.result_text.delete('1.0', tk.END)
        info_messagebox("Success", "Fit added to plot.", font=self.out_fit_sec, width=350, height=120)
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
        
        # Handle units for combination fits differently
        try:
            units_map = {}
            
            if self.enable_combo_var.get():
                # For combination fits, get units from both models
                model1_key = self.combo_model1_var.get()
                model2_key = self.combo_model2_var.get()
                
                model1 = models.get(model1_key, {})
                model2 = models.get(model2_key, {})
                
                # Add units from model 1 with _1 suffix
                for param_name, param_config in model1.get('params', {}).items():
                    if isinstance(param_config, dict) and 'unit' in param_config:
                        if param_name == 'T':
                            units_map['T'] = str(param_config['unit'])
                        else:
                            units_map[f"{param_name}_1"] = str(param_config['unit'])
                
                # Add units from model 2 with _2 suffix
                for param_name, param_config in model2.get('params', {}).items():
                    if isinstance(param_config, dict) and 'unit' in param_config:
                        if param_name != 'T':  # Skip T as it's shared
                            units_map[f"{param_name}_2"] = str(param_config['unit'])
            else:
                # Single model fit
                model_key = self.model_var.get()
                model_def = models.get(model_key, {}) if isinstance(models, dict) else {}
                
                # Extract units from parameter definitions
                if isinstance(model_def.get('params'), dict):
                    for param_name, param_config in model_def['params'].items():
                        if isinstance(param_config, dict) and 'unit' in param_config:
                            units_map[str(param_name)] = str(param_config['unit'])
                            
            units_map['T'] = 'K'  # Temperature unit is always K

            if units_map:
                new_lines = []
                in_vars = False
                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith('[Variables]'):
                        in_vars = True
                        new_lines.append(line)
                        continue
                    if in_vars:
                        m = re.match(r'^(\s*)([A-Za-z0-9_]+)\s*:(.*)$', line)
                        if m:
                            indent, name, rest = m.groups()
                            unit = units_map.get(name)
                            if unit and f'({unit})' not in line:
                                line = f"{indent}{name} ({unit}):{rest}"
                    new_lines.append(line)
                lines = new_lines
        except Exception:
            pass
            
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

    def fit_selection(self):
        """Perform a fit on the selected range using either single or combination model."""
        if self.current_x is None:
            error_messagebox("Error", "No data loaded. Please load a data file first.", font=self.out_fit_sec)
            return
        if not self.selected_range:
            error_messagebox("Error", "No range selected.", font=self.out_fit_sec, width=300, height=120)
            return
        
        x_min, x_max = self.selected_range
        mask = (self.current_x >= x_min) & (self.current_x <= x_max)
        xs = self.current_x[mask]
        ys = self.current_y[mask]
        if len(xs) < 2:
            error_messagebox("Error", "Not enough data points in the selected range.", font=self.out_fit_sec)
            return
        
        method = self.fitmethod_var.get()
        fit_warnings = ""
        
        # Check if combination fit is enabled
        func = None # Initialize func
        if self.enable_combo_var.get():
            # Combination fit
            model1_key = self.combo_model1_var.get()
            model2_key = self.combo_model2_var.get()
            model_key = f"{model1_key}+{model2_key}"
            if model1_key not in models or model2_key not in models:
                error_messagebox("Error", "Invalid combination models selected.", font=self.out_fit_sec)
                return

            try:
                with warnings.catch_warnings(record=True) as wlist:
                    warnings.simplefilter("always")
                    # Perform combination fit using the temporary model
                    fit_result = performFit(xs, ys, model1_key, model_key_b=model2_key, method=method, T=self.new_temperature.get(), secFont=self.out_fit_sec)
                    if fit_result is None:
                        return
                    func = fit_result.model.eval # Use the model's eval method
                    for w in wlist:
                        fit_warnings += f"Warning: {w.message}\n"
            except Exception as e:
                error_messagebox("Fit Error", f"Combination fitting failed ({method}): {e}", font=self.out_fit_sec)
                return

        else:
            # Single model fit
            model_key = self.model_var.get()
            if model_key not in models:
                error_messagebox("Error", "Invalid model selected.", font=self.out_fit_sec)
                return
            
            try:
                with warnings.catch_warnings(record=True) as wlist:
                    warnings.simplefilter("always")
                    # Perform single model fit
                    fit_result = performFit(xs, ys, model_key, method=method, T=self.new_temperature.get(), secFont=self.out_fit_sec)
                    if fit_result is None:
                        return
                    for w in wlist:
                        fit_warnings += f"Warning: {w.message}\n"
            except Exception as e:
                error_messagebox("Fit Error", f"Fitting failed ({method}): {e}", font=self.out_fit_sec)
                return
            func = fit_result.model.eval
        
        # Determine state based on subset
        state = "N/A"
        if self.subset_var.get().endswith("Forward"):
            state = "low"
        elif self.subset_var.get().endswith("Reverse"):
            state = "high"

        # Store fit result
        self.temp_fit = {
            'model': model_key,
            'range': (x_min, x_max),
            'popt': fit_result.params,
            'pcov': fit_result.covar,
            'func': func,
            'method': method,
            'resultmessage': fit_result.fit_report(),
            'state': state,
            'is_combination': self.enable_combo_var.get()
        }
        
        # Display results
        self.display_fit_result(fit_result)

        if fit_warnings:
            self.result_text2.insert(tk.END, "\n--- Runtime-Warnings---\n")
            self.result_text2.insert(tk.END, fit_warnings)
        
        # Generate fit curve
        fit_xs = np.linspace(x_min, x_max, 200)
        try:
            fit_ys = fit_result.eval(E=fit_xs)

            self.temp_fit['fit_xs'] = fit_xs
            self.temp_fit['fit_ys'] = fit_ys
            self.plot_data()
        except Exception as e:
            error_messagebox("Error", f"Failed to generate fit curve: {e}", font=self.out_fit_sec)

    def update_latex_display(self, font_size=12, dpi=150):
        """Update the LaTeX display based on the selected model(s)."""
        if self.enable_combo_var.get():
            # Show combination model equation
            model1_key = self.combo_model1_var.get()
            model2_key = self.combo_model2_var.get()
            
            if model1_key in models and model2_key in models:
                model1_latex = models[model1_key].get('latex', '')
                model2_latex = models[model2_key].get('latex', '')
                
                # Clean up individual LaTeX expressions by removing outer $ symbols
                if model1_latex.startswith('$') and model1_latex.endswith('$'):
                    model1_latex = model1_latex[1:-1]
                if model2_latex.startswith('$') and model2_latex.endswith('$'):
                    model2_latex = model2_latex[1:-1]

                # Remove any leading "J =" or similar to avoid duplication
                model1_latex = model1_latex.replace("\\approx", "=")
                model1_latex = model1_latex.replace("\\propto", "=")

                model2_latex = model2_latex.replace("\\approx", "=")
                model2_latex = model2_latex.replace("\\propto", "=")

                model1_latex = model1_latex.split("=")[1] if "=" in model1_latex else model1_latex
                model2_latex = model2_latex.split("=")[1] if "=" in model2_latex else model2_latex


                # Combine the LaTeX expressions
                latex_string = f"$J = {model1_latex} + {model2_latex}$"
            else:
                latex_string = r"$\text{Select valid models for combination}$"
        else:
            # Show single model equation
            selected = self.model_var.get()
            model = models.get(selected, {})
            latex_string = model.get("latex", r"$\text{Keine Formel}$")

        # Remove old label
        if self.latex_label:
            self.latex_label.destroy()

        # Render and display new label
        fig, ax = plt.subplots(figsize=(0.01, 0.01))
        fig.patch.set_visible(False)
        ax.axis('off')

        # Set LaTeX text
        ax.text(0.5, 0.5, latex_string, fontsize=font_size, ha='center', va='center')

        # Save image to BytesIO
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=dpi, transparent=True)
        buf.seek(0)
        plt.close(fig)

        # Load image with PIL and convert for Tkinter
        image = Image.open(buf)
        photo = ImageTk.PhotoImage(image)

        # Create Tkinter label
        label = tk.Label(self.latex_frame, image=photo, bg='white')
        label.image = photo  # Keep reference so image doesn't get deleted
        self.latex_label = label
        self.latex_label.pack(side='left', padx=5)

    def export_fit_data(self):
        """Export the fit data to a text file with calculated I and U values."""
        if not self.fits:
            error_messagebox("Error", "No fits available for export.", font=self.out_fit_sec)
            return
    
        save_path = filedialog.asksaveasfilename(
            defaultextension='.txt',
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if not save_path:
            return
    
        try:
            area_um2 = float(self.new_area.get())
            thickness_nm = float(self.new_thickness.get())
            area_m2 = area_um2 * 1e-12
            thickness_m = thickness_nm * 1e-9
    
            with open(save_path, 'w') as f:
                f.write(f"# Exported IV-Fit data on {datetime.datetime.now().isoformat()}\n")
                for idx, fit in enumerate(self.fits, start=1):
                    model = fit.get('model', 'unknown')
                    method = fit.get('method', 'unknown')
                    state = fit.get('state', 'N/A')
                    label = fit.get('label', f'Fit {idx}')
                    popt = fit.get('popt', {})
                    xs = np.array(fit.get('fit_xs', []))
                    ys = np.array(fit.get('fit_ys', []))
                    if xs.size != ys.size:
                        continue
    
                    # Kopfzeile für diesen Fit
                    f.write(f"\nFit {idx}: {model}, Method: {method}, Subset: {state}\n")
                    
                    # Fitparameter mit Einheiten exportieren
                    f.write("# Fit Parameters:\n")
                    model_def = models.get(model, {})
                    units_map = {}
                    
                    # Extract units from parameter definitions
                    if isinstance(model_def.get('params'), dict):
                        for param_name, param_config in model_def['params'].items():
                            if isinstance(param_config, dict) and 'unit' in param_config:
                                units_map[str(param_name)] = str(param_config['unit'])
                    units_map['T'] = 'K'  # Temperature unit
                    
                    for param_name, param_obj in popt.items():
                        unit = units_map.get(param_name, '')
                        unit_str = f" ({unit})" if unit else ""
                        value = param_obj.value if hasattr(param_obj, 'value') else param_obj
                        stderr = param_obj.stderr if hasattr(param_obj, 'stderr') and param_obj.stderr is not None else 'N/A'
                        f.write(f"# {param_name}{unit_str}: {value:.6e} ± {stderr}\n")
                    
                    f.write("E (V/m)\tJ (A/m²)\tV (V)\tI (A)\n")
    
                    # Datenzeilen (ohne Leerzeilen)
                    for E, J in zip(xs, ys):
                        V = E * thickness_m
                        I = J * area_m2
                        f.write(f"{E:.6e}\t{J:.6e}\t{V:.6e}\t{I:.6e}\n")

            info_messagebox("Export successful", f"Data saved to:\n{save_path}", font=self.out_fit_sec)

        except Exception as e:
            error_messagebox("Error during export", f"Error:\n{str(e)}", font=self.out_fit_sec)
    
    def on_closing(self):
        self.destroy()
        sys.exit()
        
    def load_icons(self):
        """Load icons for buttons. Automatically loads all PNG files from icons folder."""
        icon_size = 16
        
        # Icon mapping: button name -> possible icon filenames (without extension)
        icon_mapping = {
            'fit': ['paint-brush'],
            'add': ['add', 'plus', 'create', 'new'],
            'remove': ['remove', 'delete', 'minus', 'trash', 'cross'],
            'export': ['upload'],
            'extrapolate': ['code-compare'],
            'browse': ['folder-upload'],
            'range': ['arrows-h'],
            'delete': ['time-quarter-to']
        }
        
        # Fallback colors and symbols if no icon found
        fallback_configs = {
            'fit': {'color': '#4CAF50', 'symbol': '⚡'},
            'add': {'color': '#2196F3', 'symbol': '+'},
            'remove': {'color': '#F44336', 'symbol': '−'},
            'export': {'color': '#FF9800', 'symbol': '💾'},
            'extrapolate': {'color': '#9C27B0', 'symbol': '↗'},
            'browse': {'color': '#795548', 'symbol': '📁'},
            'range': {'color': '#607D8B', 'symbol': '📏'},
            'delete': {'color': '#E91E63', 'symbol': '🗑'}
        }
        
        # Get all PNG files in icons directory
        icons_dir = os.path.join(os.path.dirname(__file__), 'icons')
        available_icons = {}
        
        if os.path.exists(icons_dir):
            for filename in os.listdir(icons_dir):
                if filename.lower().endswith('.png'):
                    icon_name = os.path.splitext(filename)[0].lower()
                    available_icons[icon_name] = os.path.join(icons_dir, filename)
        
        # Load icons for each button
        for button_name, possible_names in icon_mapping.items():
            icon_loaded = False
            
            # Try to find matching icon
            for possible_name in possible_names:
                if possible_name.lower() in available_icons:
                    try:
                        image = Image.open(available_icons[possible_name.lower()])
                        image = image.resize((icon_size, icon_size), Image.Resampling.LANCZOS)
                        self.icons[button_name] = ImageTk.PhotoImage(image)
                        icon_loaded = True
                        break
                    except Exception as e:
                        print(f"Failed to load {possible_name}.png for {button_name}: {e}")
                        continue
            
            # If no icon found, create fallback
            if not icon_loaded:
                fallback = fallback_configs.get(button_name, {'color': '#808080', 'symbol': '?'})
                self.icons[button_name] = self.create_simple_icon(
                    icon_size, fallback['color'], fallback['symbol']
                )
                print(f"Created fallback icon for {button_name}")

    def create_simple_icon(self, size, color, symbol):
        """Create a simple colored icon with a symbol."""
        try:
            # Create colored background
            img = Image.new('RGBA', (size, size), (255, 255, 255, 0))
            
            # Convert hex color to RGB
            color_rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
            
            # Create a simple colored circle or square
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            
            # Draw colored circle
            margin = 2
            draw.ellipse([margin, margin, size-margin, size-margin], 
                        fill=color_rgb + (200,), outline=color_rgb + (255,))
            
            # Try to add symbol (fallback if font issues)
            try:
                # Try to get a reasonable font size
                font_size = max(8, size // 2)
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                try:
                    font = ImageFont.load_default()
                except:
                    font = None
            
            if font and len(symbol) == 1:
                # Get text size and center it
                bbox = draw.textbbox((0, 0), symbol, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                x = (size - text_width) // 2
                y = (size - text_height) // 2
                draw.text((x, y), symbol, fill='white', font=font)
            
            return ImageTk.PhotoImage(img)
        except Exception:
            # Ultimate fallback: empty PhotoImage
            fallback_img = Image.new('RGBA', (size, size), (128, 128, 128, 100))
            return ImageTk.PhotoImage(fallback_img)

    def save_config(self):
        """Saves the current settings to a configuration file."""
        config = {
            'file_path': self.file_var.get(),
            'area': self.new_area.get(),
            'thickness': self.new_thickness.get(),
            'temperature': self.new_temperature.get(),
            'subset': self.subset_var.get(),
            'model': self.model_var.get(),
            'fit_method': self.fitmethod_var.get(),
            'enable_combo': self.enable_combo_var.get(),
            'combo_model1': self.combo_model1_var.get(),
            'combo_model2': self.combo_model2_var.get(),
            'left_plot_scale': self.left_scale_var.get(),
            'right_plot_scale': self.right_scale_var.get(),
        }
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"Error saving configuration: {e}")

    def load_config(self):
        """Loads settings from a configuration file if it exists."""
        if not os.path.exists(self.config_file):
            return
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)

            # Set all the variables
            self.new_area.set(config.get('area', '625'))
            self.new_thickness.set(config.get('thickness', '10'))
            self.new_temperature.set(config.get('temperature', '300'))
            self.subset_var.set(config.get('subset', 'All'))
            self.model_var.set(config.get('model', list(models.keys())[0]))
            self.fitmethod_var.set(config.get('fit_method', 'leastsq'))
            self.enable_combo_var.set(config.get('enable_combo', False))
            self.combo_model1_var.set(config.get('combo_model1', list(models.keys())[0]))
            self.combo_model2_var.set(config.get('combo_model2', list(models.keys())[1] if len(models.keys()) > 1 else list(models.keys())[0]))
            self.left_scale_var.set(config.get('left_plot_scale', 'log'))
            self.right_scale_var.set(config.get('right_plot_scale', 'log'))

            # Load data file if it exists
            file_path = config.get('file_path')
            if file_path and os.path.exists(file_path):
                self.load_data_from_path(file_path)
            
            # Update the LaTeX display to reflect the loaded configuration
            self.update_latex_display()

        except Exception as e:
            print(f"Error loading configuration: {e}")

# Start up
    def on_closing(self):
        self.save_config()
        self.destroy()
        sys.exit()
        
if __name__ == "__main__":
    app = FitApp()
    app.mainloop()
