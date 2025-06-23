import tkinter as tk
from tkinter import ttk, messagebox


class ParamDialog(tk.Toplevel):
    def __init__(self, parent, param_names, default_p0=None, default_bounds=None):
        """
        param_names: Liste von Strings.
        default_p0: Dict param_name -> float oder None.
        default_bounds: Dict param_name -> (min, max) oder None.
        """
        super().__init__(parent)
        self.title("Parameter Einstellungen")
        self.transient(parent)
        self.grab_set()
        self.result = None  # wird später dict mit 'p0' und 'bounds' oder None
        self.entries = {}   # param_name -> dict mit 'p0', 'min', 'max' entries
        frm = ttk.Frame(self)
        frm.pack(padx=10, pady=10, fill='both', expand=True)
        for name in param_names:
            row = ttk.Frame(frm)
            row.pack(fill='x', pady=2)
            ttk.Label(row, text=name).pack(side='left')
            # Initialwert
            p0_val = None
            if default_p0 and name in default_p0:
                p0_val = default_p0[name]
            p0_str = "" if p0_val is None else str(p0_val)
            ent_p0 = ttk.Entry(row, width=10)
            ent_p0.insert(0, p0_str)
            ent_p0.pack(side='left', padx=(5,0))
            # Min
            min_val = None
            if default_bounds and name in default_bounds:
                min_val = default_bounds[name][0]
            min_str = "" if min_val is None else str(min_val)
            ent_min = ttk.Entry(row, width=10)
            ent_min.insert(0, min_str)
            ent_min.pack(side='left', padx=(5,0))
            # Max
            max_val = None
            if default_bounds and name in default_bounds:
                max_val = default_bounds[name][1]
            max_str = "" if max_val is None else str(max_val)
            ent_max = ttk.Entry(row, width=10)
            ent_max.insert(0, max_str)
            ent_max.pack(side='left', padx=(5,0))
            # Speichern
            self.entries[name] = {'p0': ent_p0, 'min': ent_min, 'max': ent_max}
        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="OK", command=self.on_ok).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.on_cancel).pack(side='left', padx=5)
        # Centerieren etc.
        self.protocol("WM_DELETE_WINDOW", self.on_cancel)
        self.wait_window()

    def on_ok(self):
        p0_dict = {}
        bounds_dict = {}
        try:
            for name, ents in self.entries.items():
                # Initialwert
                p0_text = ents['p0'].get().strip()
                if p0_text:
                    p0_val = float(p0_text)
                else:
                    p0_val = None
                p0_dict[name] = p0_val
                # Min
                min_text = ents['min'].get().strip()
                if min_text:
                    min_val = float(min_text)
                else:
                    min_val = None
                # Max
                max_text = ents['max'].get().strip()
                if max_text:
                    max_val = float(max_text)
                else:
                    max_val = None
                # Prüfen, wenn beide vorhanden: min ≤ max
                if (min_val is not None) and (max_val is not None) and (min_val > max_val):
                    raise ValueError(f"Für Parameter {name}: min > max.")
                if (min_val is not None) or (max_val is not None):
                    bounds_dict[name] = (min_val, max_val)
            # Wenn keine Probleme:
            self.result = {'p0': p0_dict, 'bounds': bounds_dict}
            self.destroy()
        except Exception as e:
            messagebox.showerror("Ungültige Eingabe", str(e), parent=self)

    def on_cancel(self):
        self.result = None
        self.destroy()
