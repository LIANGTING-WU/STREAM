import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import csv
import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Physical Constants ---
kB = 1.380649e-23   # Boltzmann constant (J/K)
h = 6.62607015e-34   # Planck constant (J·s)

# --- Plotting Colors ---
rgb_list = [
    (212/255,  67/255,  70/255 ), (227/255, 132/255, 134/255 ),
    (255/255, 185/255,  63/255 ), (255/255, 215/255, 147/255 ),
    ( 24/255, 126/255, 239/255 ), (173/255, 197/255, 241/255 ),
    (108/255, 193/255, 164/255 ), (163/255, 216/255, 198/255 ),
]
original_cmap = plt.get_cmap("YlGnBu_r")
new_cmap = colors.LinearSegmentedColormap.from_list("new_cmap", original_cmap(np.linspace(0.15, 1, 256)))

# --- Core Computation Functions ---
def parse_float(value, default=0.0):
    try: return float(value)
    except (ValueError, TypeError): return default

def compute_rate_constants(params, T, p_bar, A_in, mobile_ts):
    molecule_type = params.get("molecule_type", "linear")
    m_amu = parse_float(params.get("m", 1.0))
    Edes_eV = parse_float(params.get("Edes", 0.0))
    sigma = parse_float(params.get("sigma", 1.0))
    p, A, m_kg, Edes_J = p_bar*1e5, A_in*1e-20, m_amu*1.660539e-27, Edes_eV*1.602176565e-19
    if mobile_ts:
        k_ads = p * A / np.sqrt(2. * np.pi * m_kg * kB * T)
        k_des = (kB * T**2) / h**3 * (2. * A * np.pi * m_kg * kB) * np.exp(-Edes_J/(kB*T))
    else:
        if molecule_type == "linear":
            theta_rot = parse_float(params.get("theta_rot_linear", 1.0))
            k_ads = p * A / np.sqrt(2. * np.pi * m_kg * kB * T) * (sigma * theta_rot / T)
            k_des = (kB * T**3) / h**3 * (2. * A * np.pi * m_kg * kB/(sigma*theta_rot)) * np.exp(-Edes_J/(kB*T))
        elif molecule_type == "nonlinear":
            theta_rot_A, theta_rot_B, theta_rot_C = [parse_float(params.get(f"theta_rot_{x}", 1.0)) for x in ['A', 'B', 'C']]
            theta_rot_product = theta_rot_A * theta_rot_B * theta_rot_C
            k_ads = p * A / np.sqrt(2. * np.pi * m_kg * kB * T) * (sigma / np.sqrt(np.pi)) * np.sqrt(theta_rot_product/T**3)
            k_des = (kB * T**3.5) / h**3 * (2. * A * np.pi**1.5 * m_kg * kB/(sigma*np.sqrt(theta_rot_product))) * np.exp(-Edes_J/(kB*T))
        else: raise ValueError(f"Unknown molecule type: {molecule_type}")
    return k_ads, k_des

def ode_adsorption(t, y, k_ads_arr, k_des_arr, N):
    c_a = y[:N]; c_star = y[N]; dydt = np.zeros(N + 1)
    for i in range(N): dydt[i] = k_ads_arr[i] * c_star - k_des_arr[i] * c_a[i]
    dydt[N] = np.sum(k_des_arr * c_a) - np.sum(k_ads_arr) * c_star
    return dydt

def calculate_desorption(t, c_a_t0, k_des_arr, t0):
    return c_a_t0 * np.exp(-k_des_arr * (t - t0)[:, None])

# --- Utility Classes and Functions ---
class CheckboxDialog(tk.Toplevel):
    def __init__(self, parent, title, items):
        super().__init__(parent)
        self.title(title); self.transient(parent); self.grab_set()
        self.result = []; self.items = items; self.vars = []
        for item in self.items:
            var = tk.BooleanVar(value=True); self.vars.append(var)
            ttk.Checkbutton(self, text=item, variable=var).pack(anchor='w', padx=10, pady=2)
        btn_frame = ttk.Frame(self); btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="OK", command=self.on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=5)
        self.wait_window(self)

    def on_ok(self):
        self.result = [item for item, var in zip(self.items, self.vars) if var.get()]; self.destroy()

class PlotManager(ttk.Frame):
    """A frame to manage displaying and exporting Matplotlib figures."""
    def __init__(self, parent):
        super().__init__(parent)
        self.notebook = None
        self.active_figure = None
        self.active_filename = None

        self.button_frame = ttk.Frame(self); self.button_frame.pack(fill=tk.X, pady=2)
        self.export_button = ttk.Button(self.button_frame, text="Export Current Plot", command=self.export_plot)
        self.plot_area = ttk.Frame(self); self.plot_area.pack(fill=tk.BOTH, expand=True)
        self.placeholder = None

    def show_plots(self, plot_data):
        self.clear_plots()
        self.export_button.pack(side=tk.RIGHT, padx=5)

        if not plot_data:
            self.set_placeholder("No data to plot. Please check parameters.", "orange"); return

        if len(plot_data) == 1:
            self.active_figure, _, self.active_filename = plot_data[0]
            canvas = FigureCanvasTkAgg(self.active_figure, master=self.plot_area)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, expand=False, fill=tk.NONE)
        else:
            self.notebook = ttk.Notebook(self.plot_area); self.notebook.pack(fill=tk.BOTH, expand=True)
            self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)
            for i, (fig, title, filename) in enumerate(plot_data):
                tab = ttk.Frame(self.notebook); self.notebook.add(tab, text=title)
                canvas = FigureCanvasTkAgg(fig, master=tab); canvas.draw()
                canvas.get_tk_widget().pack(side=tk.TOP, expand=False, fill=tk.NONE)
                tab.figure = fig; tab.filename = filename
                if i == 0: self.active_figure, self.active_filename = fig, filename

    def on_tab_change(self, event):
        if self.notebook:
            selected_tab = self.nametowidget(self.notebook.select())
            self.active_figure = selected_tab.figure
            self.active_filename = selected_tab.filename

    def clear_plots(self):
        for widget in self.plot_area.winfo_children(): widget.destroy()
        self.notebook = None; self.active_figure = None; self.active_filename = None
        self.export_button.pack_forget()

    def set_placeholder(self, text, color="black"):
        self.clear_plots()
        self.placeholder = ttk.Label(self.plot_area, text=text, justify=tk.CENTER, foreground=color)
        self.placeholder.pack(expand=True)

    def export_plot(self):
        if self.active_figure is None or self.active_filename is None:
            messagebox.showwarning("No Plot", "There is no plot to export."); return
        
        file_path = os.path.join(os.path.dirname(__file__), self.active_filename)
        try:
            self.active_figure.savefig(file_path, dpi=600, bbox_inches='tight')
            messagebox.showinfo("Success", f"Plot saved to {self.active_filename}")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save plot: {e}")

# --- Main Application Class ---
class SensingApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("STREAM: Simulation toolkit for Time-Resolved Electronic transport and gas Adsorption Modeling"); self.geometry("1500x950")
        self.csv_file_path = os.path.join(os.path.dirname(__file__), "molecule_parameters.csv")
        self.parameters, self.headers = [], []
        self.negf_results = None; self.keq_results = None
        
        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL); main_pane.pack(fill=tk.BOTH, expand=True)
        left_frame = ttk.Frame(main_pane, width=450); main_pane.add(left_frame, weight=2)
        self.plot_notebook = ttk.Notebook(main_pane); main_pane.add(self.plot_notebook, weight=3)

        self._setup_controls(left_frame); self._setup_plot_tabs(); self.load_csv()

    def _setup_controls(self, left_frame):
        left_frame.grid_rowconfigure(2, weight=1); left_frame.grid_columnconfigure(0, weight=1)
        global_params_frame = ttk.LabelFrame(left_frame, text="Global Parameters"); global_params_frame.grid(row=0, column=0, pady=10, padx=10, sticky="ew")
        self.mobile_ts_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(global_params_frame, text="Enable Mobile TS Assumption", variable=self.mobile_ts_var).grid(row=0, column=0, columnspan=2, sticky='w', padx=5, pady=2)
        self.pressure_var = tk.StringVar(value="1.0"); self.area_var = tk.StringVar(value="24.8378")
        self.ttarget_var = tk.StringVar(value="300"); self.tspanstart_var = tk.StringVar(value="1e-12")
        self.tspanend_var = tk.StringVar(value="1e-2"); self.t0_var = tk.StringVar(value="1e-6")
        self.tmin_var = tk.StringVar(value="100"); self.tmax_var = tk.StringVar(value="600"); self.operating_voltage_var = tk.StringVar(value="1.0")
        params_layout = {
            "Pressure (bar) for Single Gas:": (self.pressure_var, 1), "Area (Å²):": (self.area_var, 2),
            "T_target (K) for ODE:": (self.ttarget_var, 3), "t_span start (s):": (self.tspanstart_var, 4),
            "t_span end (s):": (self.tspanend_var, 5), "t0 (s):": (self.t0_var, 6),
            "T_min (K) for Temp Dep:": (self.tmin_var, 7), "T_max (K) for Temp Dep:": (self.tmax_var, 8),
            "Operating Voltage (V):": (self.operating_voltage_var, 9)
        }
        for text, (var, row) in params_layout.items():
            ttk.Label(global_params_frame, text=text).grid(row=row, column=0, sticky='w', padx=5, pady=2)
            ttk.Entry(global_params_frame, textvariable=var, width=15).grid(row=row, column=1, sticky='ew', padx=5, pady=2)
        global_params_frame.columnconfigure(1, weight=1)
        control_frame = ttk.LabelFrame(left_frame, text="Plotting Controls"); control_frame.grid(row=1, column=0, pady=10, padx=10, sticky="ew")
        ttk.Button(control_frame, text="Generate / Update Plot", command=self.update_current_plot).pack(fill=tk.X, pady=5, padx=5)
        param_frame = ttk.LabelFrame(left_frame, text="Molecule Parameters"); param_frame.grid(row=2, column=0, pady=10, padx=10, sticky="nsew")
        param_frame.grid_rowconfigure(0, weight=1); param_frame.grid_columnconfigure(0, weight=1)
        self.tree = ttk.Treeview(param_frame, show='headings'); self.tree.grid(row=0, column=0, columnspan=2, sticky='nsew')
        vsb = ttk.Scrollbar(param_frame, orient="vertical", command=self.tree.yview); vsb.grid(row=0, column=2, sticky='ns')
        hsb = ttk.Scrollbar(param_frame, orient="horizontal", command=self.tree.xview); hsb.grid(row=1, column=0, columnspan=2, sticky='ew')
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        csv_button_frame = ttk.Frame(param_frame); csv_button_frame.grid(row=2, column=0, columnspan=2, pady=5)
        for text, command in {"Load CSV": self.load_csv_dialog, "Save CSV": self.save_csv, "Add Row": self.add_row, "Delete Selected": self.delete_row}.items():
            ttk.Button(csv_button_frame, text=text, command=command).pack(side=tk.LEFT, padx=5)
        target_species_frame = ttk.Frame(param_frame); target_species_frame.grid(row=3, column=0, columnspan=2, pady=5, sticky='ew')
        ttk.Label(target_species_frame, text="Target for Bar Chart:").pack(side=tk.LEFT, padx=5)
        self.target_species_var = tk.StringVar()
        self.target_species_combo = ttk.Combobox(target_species_frame, textvariable=self.target_species_var, state='readonly')
        self.target_species_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.tree.bind("<Double-1>", self.on_double_click)

    def _setup_plot_tabs(self):
        self.plot_tabs = {}
        plot_configs = {
            "Coverage vs. Time": self.run_gas_sensing_single,
            "3D Coverage (T, t)": self.run_gas_sensing_3d,
            "Multi-Component Coverage": self.run_multi_component_ode,
            "Temperature Dependence": self.run_temperature_dependence,
            "NEGF Analyzer": self.run_iv_analyzer,
            "Selectivity": self.run_advanced_selectivity_analysis,
            "Resistance vs. Time": self.run_total_resistance_simulation
        }
        for name, plot_func in plot_configs.items():
            tab_frame = ttk.Frame(self.plot_notebook); self.plot_notebook.add(tab_frame, text=name)
            plot_manager = PlotManager(tab_frame); plot_manager.pack(fill=tk.BOTH, expand=True)
            plot_manager.set_placeholder(f"This is the '{name}' tab.\nClick 'Generate / Update Plot' to see the results.")
            self.plot_tabs[name] = {"manager": plot_manager, "func": plot_func}

    def update_current_plot(self):
        try: selected_tab_name = self.plot_notebook.tab(self.plot_notebook.select(), "text")
        except tk.TclError: messagebox.showwarning("No Tab", "There is no active tab to plot."); return
        if selected_tab_name in self.plot_tabs:
            plot_info = self.plot_tabs[selected_tab_name]; manager = plot_info["manager"]; plot_function = plot_info["func"]
            try:
                plot_data = plot_function()
                manager.show_plots(plot_data)
            except Exception as e:
                manager.set_placeholder(f"Failed to generate plot for '{selected_tab_name}'.\n\nError: {e}", "red")
                messagebox.showerror("Plotting Error", f"An error occurred while generating the plot: {e}")

    def _get_global_params(self):
        try:
            return (float(self.pressure_var.get()), float(self.area_var.get()), self.mobile_ts_var.get(),
                    float(self.ttarget_var.get()), (float(self.tspanstart_var.get()), float(self.tspanend_var.get())),
                    float(self.t0_var.get()), float(self.tmin_var.get()), float(self.tmax_var.get()),
                    float(self.operating_voltage_var.get()))
        except ValueError: messagebox.showerror("Invalid Input", "Enter valid numbers for all global/simulation parameters."); return (None,)*9

    def run_gas_sensing_single(self):
        params = self.get_selected_molecule_params()
        if not params: messagebox.showwarning("No Selection", "Please select a molecule."); return None
        p_bar, A_in, mobile_ts, T_target, t_span, t0, _, _, _ = self._get_global_params()
        if p_bar is None: return None
        fig, ax = plt.subplots(figsize=(10, 6))
        c_a0 = 1.0; k_ads, k_des = compute_rate_constants(params, T_target, p_bar, A_in, mobile_ts)
        t_ads = np.logspace(np.log10(t_span[0]), np.log10(t0), 2000)
        c_a_ads = (k_ads/(k_ads+k_des))*(1-np.exp(-(k_ads+k_des)*t_ads))*c_a0; c_star_ads = c_a0-c_a_ads
        c_a_t0 = c_a_ads[-1]; t_des = np.logspace(np.log10(t0), np.log10(t_span[1]), 500)
        c_a_des = c_a_t0*np.exp(-k_des*(t_des-t0)); c_star_des = c_a0-c_a_des
        print(f"Recovery Time for {params['name']} at {T_target}K = {1.0/k_des:.4e} s")
        ax.plot(t_ads, c_a_ads, label="Adsorbed Gas", color=rgb_list[0]); ax.plot(t_des, c_a_des, color=rgb_list[0])
        ax.plot(t_ads, c_star_ads, label="Active Sites", linestyle="--", color='grey'); ax.plot(t_des, c_star_des, linestyle="--", color='grey')
        ax.axvline(x=t0, color='slategray', linestyle='-.', label=f't0={t0:.1e}s')
        ax.set(xscale="log", xlabel="Time (s)", ylabel="Coverage", title=f"Coverage vs. Time for {params['name']} at {T_target}K", ylim=(0, None))
        ax.grid(True, which='major', linestyle='-', alpha=0.7); ax.grid(True, which='minor', linestyle='--', alpha=0.4); ax.legend()
        fig.tight_layout()
        filename = f"{params['name']}_{T_target}K_coverage.png"
        return [(fig, f"{params['name']} Coverage", filename)]

    def run_gas_sensing_3d(self):
        params = self.get_selected_molecule_params()
        if not params: messagebox.showwarning("No Selection", "Please select a molecule."); return None
        p_bar, A_in, mobile_ts, _, t_span, t0, T_min, T_max, _ = self._get_global_params()
        if p_bar is None: return None
        t_eval = np.logspace(np.log10(t_span[0]), np.log10(t_span[1]), 200)
        T_eval = np.linspace(T_min, T_max, 50); cA_star_T = np.zeros((len(T_eval), len(t_eval)))
        for i, T in enumerate(T_eval):
            k_ads, k_des = compute_rate_constants(params, T, p_bar, A_in, mobile_ts)
            c_a0=1.; c_a_t0=(k_ads/(k_ads+k_des))*(1-np.exp(-(k_ads+k_des)*t0))*c_a0
            cA_star_T[i,:] = np.where(t_eval<=t0, (k_ads/(k_ads+k_des))*(1-np.exp(-(k_ads+k_des)*t_eval))*c_a0, c_a_t0*np.exp(-k_des*(t_eval-t0)))
        T_grid, t_grid = np.meshgrid(T_eval, t_eval, indexing="ij")
        fig = plt.figure(figsize=(10, 4))
        gs = gridspec.GridSpec(1, 4, width_ratios=[1, 0.05, 1, 0.05], wspace=0.3)
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        ax2 = fig.add_subplot(gs[0, 2], projection='3d')
        cax = fig.add_subplot(gs[0, 3])
        axes = [ax1, ax2]
        for ax in axes:
            surf = ax.plot_surface(np.log10(t_grid), T_grid, cA_star_T, cmap=new_cmap, edgecolor=(.2,.2,.2,.2), vmin=0, vmax=1)
            ax.set(xlabel='t (s)', ylabel='T (K)', zlabel=r'$\theta_{A^*}$', xlim=(np.log10(t_eval.min()), np.log10(t_eval.max())), ylim=(T_eval.min(), T_eval.max()), zlim=(0,1))
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"$10^{{{int(x)}}}$"))
            ax.tick_params(axis='x', labelsize=8), ax.tick_params(axis='y', labelsize=8), ax.tick_params(axis='z', labelsize=8)
        axes[0].view_init(elev=30, azim=105); axes[1].view_init(elev=60, azim=45)
        cb = fig.colorbar(surf, cax=cax)
        cb.set_label("Coverage", fontsize=14)
        fig.suptitle(f"3D Coverage vs. Time and Temperature for {params['name']}", fontsize=16)
        #fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        filename = f"{params['name']}_Coverage_Time_Temp_3D.png"
        return [(fig, "3D Coverage Plot", filename)]

    def run_multi_component_ode(self, return_results=False):
        all_params = self.get_all_molecule_params()
        if not all_params: messagebox.showwarning("No Data", "No molecules in table."); return None
        _, A_in, mobile_ts, T_target, t_span, t0, _, _, _ = self._get_global_params()
        if A_in is None: return None
        N = len(all_params)
        k_ads_array, k_des_array = np.zeros(N), np.zeros(N)
        for i, params in enumerate(all_params):
            p_bar_csv = parse_float(params.get('p', 1.0))
            k_ads, k_des = compute_rate_constants(params, T_target, p_bar_csv, A_in, mobile_ts)
            k_ads_array[i], k_des_array[i] = k_ads, k_des
        y0 = np.zeros(N + 1); y0[-1] = 1.0
        t_eval_ads = np.logspace(np.log10(t_span[0]), np.log10(t0), 2000)
        sol_ads = solve_ivp(ode_adsorption, (t_span[0], t0), y0, t_eval=t_eval_ads, args=(k_ads_array, k_des_array, N), method="BDF")
        if not sol_ads.success: messagebox.showerror("ODE Error", f"Adsorption phase solver failed: {sol_ads.message}"); return None
        c_a_t0 = sol_ads.y[:N, -1]
        t_eval_des = np.logspace(np.log10(t0), np.log10(t_span[1]), 500)
        c_a_des = calculate_desorption(t_eval_des, c_a_t0, k_des_array, t0)
        t_combined = np.concatenate([sol_ads.t, t_eval_des])
        c_a_combined = np.hstack([sol_ads.y[:N, :], c_a_des.T])
        c_star_combined = 1.0 - np.sum(c_a_combined, axis=0)
        if return_results: return t_combined, c_a_combined, c_star_combined, all_params
        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(N):
            ax.plot(t_combined, c_a_combined[i, :], label=all_params[i]['name'], color=rgb_list[i % len(rgb_list)])
        ax.plot(t_combined, c_star_combined, label='Active Sites', linestyle='--', linewidth=2, color='grey')
        ax.set(xscale="log", xlabel="Time (s)", ylabel="Coverage", title=f"Multi-Component Coverage at {T_target}K")
        ax.grid(True, which='major', alpha=0.7); ax.grid(True, which='minor', alpha=0.4); ax.legend()
        fig.tight_layout()
        try:
            header_line = "Time " + " ".join([p['name'] for p in all_params]) + " c_star"
            np.savetxt("coverage_results.dat", np.column_stack([t_combined, *c_a_combined, c_star_combined]), header=header_line, comments='', fmt='%.12e')
            print("Saved coverage_results.dat")
        except Exception as e: messagebox.showerror("File Save Error", f"Could not save .dat files: {e}")
        return [(fig, "Multi-Component Coverage", "Multi_Component_Coverage.png")]

    def run_total_resistance_simulation(self):
        if not self.negf_results: messagebox.showerror("Prerequisite Error", "Please run the NEGF I-V Data Analyzer first."); return None
        results = self.run_multi_component_ode(return_results=True)
        if not results: return None
        t_combined, c_a_combined, c_star_combined, all_params = results
        _, _, _, _, _, _, _, _, operating_voltage = self._get_global_params()
        if operating_voltage is None: return None
        resistance_values = {name: np.interp(operating_voltage, self.negf_results['voltage'], res) for name, res in zip(self.negf_results['species'], self.negf_results['resistances'])}
        resistance_info = [f"{name:<15}: {res_val:.4f} kΩ" for name, res_val in resistance_values.items()]
        print(f"\n--- Resistances at Operating Voltage: {operating_voltage:.4f} V ---\n" + "\n".join(resistance_info) + "\n" + "-"*50 + "\n")
        try:
            with open("resistance_at_operating_voltage.txt", "w") as f: f.write(f"Resistances at Operating Voltage: {operating_voltage:.4f} V\n" + "-"*40 + "\n" + "\n".join(resistance_info))
            print("Resistance values saved to resistance_at_operating_voltage.txt")
        except Exception as e: print(f"Warning: Could not save resistance values to file. Error: {e}")
        total_resistance = np.zeros_like(t_combined)
        pristine_name = self.negf_results['species'][0]
        for i, params in enumerate(all_params):
            mol_name = params['name']
            if mol_name in resistance_values: total_resistance += c_a_combined[i, :] * resistance_values[mol_name]
            else: print(f"Warning: Resistance for {mol_name} not found in NEGF results. Skipping.")
        if pristine_name in resistance_values: total_resistance += c_star_combined * resistance_values[pristine_name]
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(t_combined, total_resistance, color='lightseagreen')
        ax1.set(xscale="log", xlabel="Time (s)", ylabel="Total Resistance (kΩ)", title=f"Total Resistance (Log Time) at {operating_voltage:.2f}V")
        ax1.grid(True, which="both"); fig1.tight_layout()
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        final_resistance = total_resistance[-1]
        unstable_indices = np.where(np.abs(total_resistance - final_resistance) > (final_resistance * 0.01))[0]
        t_stable = t_combined[unstable_indices[-1]] if len(unstable_indices) > 0 else t_combined[-1]
        t_max_plot = t_stable * 1.2
        ax2.plot(t_combined, total_resistance, color='lightseagreen')
        ax2.set(xlabel="Time (s)", ylabel="Total Resistance (kΩ)", title=f"Total Resistance (Linear Time, Auto-Zoom) at {operating_voltage:.2f}V", xlim=(0, t_max_plot))
        ax2.grid(True); fig2.tight_layout()
        try:
            np.savetxt("R_t.dat", np.column_stack([t_combined, total_resistance]), header="Time(s)\tTotal_Resistance(kOhm)", delimiter="\t", comments='')
            print("Saved R_t.dat")
        except Exception as e: messagebox.showerror("Save Error", f"Could not save resistance data: {e}")
        return [(fig1, "Log Time Scale", "R_t_log_scale.png"), (fig2, "Linear Time Scale", "R_t_linear_scale.png")]

    def run_temperature_dependence(self):
        all_params = self.get_all_molecule_params()
        if not all_params: messagebox.showwarning("No Data", "No molecules in table."); return None
        p_bar, A_in, mobile_ts, T_target, _, _, T_min, T_max, _ = self._get_global_params()
        if A_in is None: return None
        T_range = np.linspace(T_min, T_max, 100)
        Keq_vals_at_T_target = np.zeros(len(all_params))
        tau_vals, Keq_vals, c_a_eq_vals = [np.zeros((len(all_params), 100)) for _ in range(3)]
        for i, params in enumerate(all_params):
            k_ads_target, k_des_target = compute_rate_constants(params, T_target, p_bar, A_in, mobile_ts)
            if k_des_target > 0: Keq_vals_at_T_target[i] = k_ads_target / k_des_target
            for j, T in enumerate(T_range):
                k_ads, k_des = compute_rate_constants(params, T, p_bar, A_in, mobile_ts)
                if k_des > 0: tau_vals[i,j]=1./k_des; Keq_vals[i,j]=k_ads/k_des
                if Keq_vals[i,j] > 0: c_a_eq_vals[i,j]=1.0/(1.+1./Keq_vals[i,j])
        self.keq_results = {'species': [p['name'] for p in all_params], 'keq': Keq_vals_at_T_target}
        print("Standardized K_eq results (at 1 bar) stored for subsequent calculations.")
        figs = []
        plot_data = [
            ("Recovery Time, τ (s)", tau_vals, True, "Recovery_Time.png"),
            ("Equilibrium Constant, $K_{eq}$", Keq_vals, True, "Equilibrium_Constant.png"),
            ("Equilibrium Coverage, $θ_{eq}$", c_a_eq_vals, False, "Equilibrium_Coverage.png")
        ]
        for title, data, is_log, fname in plot_data:
            fig, ax = plt.subplots(figsize=(10, 6))
            for i, params in enumerate(all_params):
                ax.plot(T_range, data[i,:], label=params['name'], color=rgb_list[i % len(rgb_list)])
            ax.set(xlabel="Temperature (K)", ylabel=title, title=f"{title} (at 1 bar)")
            if is_log: ax.set_yscale("log")
            ax.grid(True, which="both" if is_log else "major"); ax.legend()
            fig.tight_layout()
            figs.append((fig, title.split(',')[0], fname))
        return figs

    def run_advanced_selectivity_analysis(self):
        if not self.negf_results or not self.keq_results: messagebox.showerror("Prerequisite Error", "Please run NEGF Analyzer and Temp Dep first."); return None
        _, _, _, T_target, _, _, _, _, operating_voltage = self._get_global_params()
        target_species_name = self.target_species_var.get()
        if operating_voltage is None or not target_species_name: messagebox.showwarning("Input Needed", "Set valid Operating Voltage and Target Species."); return None
        keq_map = {s: k for s, k in zip(self.keq_results['species'], self.keq_results['keq'])}
        negf_species_map = {name: i for i, name in enumerate(self.negf_results['species'])}
        common_species = [s for s in keq_map if s in negf_species_map and s != self.negf_results['species'][0]]
        dialog = CheckboxDialog(self, "Select Species for Analysis", common_species)
        selected_species = dialog.result
        if not selected_species or len(selected_species) < 2: messagebox.showwarning("Selection Error", "Please select at least two species."); return None
        sens_at_v = [np.interp(operating_voltage, self.negf_results['voltage'], self.negf_results['sensitivities'][negf_species_map[s]]) for s in selected_species]
        keq_vals = [keq_map[s] for s in selected_species]
        n = len(selected_species)
        S = np.full((n, n), np.nan)
        for i in range(n): 
            for j in range(n):
                if i != j and sens_at_v[i] != 0 and keq_vals[i] != 0: S[i, j] = (sens_at_v[j] / sens_at_v[i]) * (keq_vals[j] / keq_vals[i])
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        logS = np.log10(S); vmin, vmax = np.nanmin(logS), np.nanmax(logS)
        norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        im = ax1.imshow(logS, cmap='RdBu', norm=norm, interpolation='nearest')
        cb = fig1.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        ticks = np.linspace(vmin, vmax, num=7)
        cb.set_ticks(ticks); cb.set_ticklabels([f"$10^{{{int(tick)}}}$" for tick in ticks])
        ax1.set_xticks(np.arange(n)); ax1.set_xticklabels(selected_species, rotation=45, ha="right")
        ax1.set_yticks(np.arange(n)); ax1.set_yticklabels(selected_species)
        ax1.set(xlabel="Species j (Target)", ylabel="Species i (Reference)", title=f"Selectivity Matrix (log scale) at {operating_voltage:.2f}V & {T_target}K")
        fig1.tight_layout()
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        if target_species_name in selected_species:
            target_index = selected_species.index(target_species_name)
            reference_species = [sp for idx, sp in enumerate(selected_species) if idx != target_index]
            selectivity_target = np.delete(S[:, target_index], target_index)
            ax2.bar(reference_species, selectivity_target, color=rgb_list)
            ax2.set(yscale="log", xlabel="Reference Species (i)", ylabel=f"Selectivity of {target_species_name} vs. i", title=f"Selectivity Bar Chart for {target_species_name}")
            ax2.grid(True, which='both', axis='y', linestyle='--')
        else: ax2.text(0.5, 0.5, "Target species for bar chart\nwas not included in the analysis.", ha='center', va='center', fontsize=12)
        fig2.tight_layout()
        return [(fig1, "Selectivity Matrix", "Selectivity_Analysis_Matrix.png"), (fig2, "Bar Chart", f"Selectivity_Bar_Chart_{target_species_name}.png")]

    def run_iv_analyzer(self):
        file_path = filedialog.askopenfilename(title="Select NEGF Result CSV File", filetypes=(("CSV files", "*.csv"),))
        if not file_path: return None
        try:
            df = pd.read_csv(file_path)
            if df.shape[1] < 2: raise ValueError("CSV must have at least Voltage and one Current column.")
        except Exception as e: messagebox.showerror("File Read Error", f"Could not read or parse CSV file.\nError: {e}"); return None
        voltage = df.iloc[:, 0].values; species = df.columns.tolist()[1:]
        if not species: messagebox.showerror("Data Error", "No molecule/species columns found."); return None
        currents = [df.iloc[:, i].values * 1000 for i in range(1, df.shape[1])]
        resistances = [np.where(c == 0, np.nan, voltage / c) for c in currents]
        sensitivities = [np.where(resistances[0] == 0, 0, 100 * abs(R - resistances[0]) / resistances[0]) for R in resistances]
        self.negf_results = {'voltage': voltage, 'species': species, 'resistances': resistances, 'sensitivities': sensitivities}
        print("NEGF results stored for subsequent calculations.")
        figs = []
        plot_configs = [
            ("I-V Curves", currents, "Current (mA)", "NEGF_IV_Curves.png"),
            ("R-V Curves", resistances, "Resistance (kΩ)", "NEGF_RV_Curves.png"),
        ]
        for title, data, ylabel, fname in plot_configs:
            fig, ax = plt.subplots(figsize=(10, 6))
            for i, name in enumerate(species):
                ax.plot(voltage, data[i], label=name, color='k' if i == 0 else rgb_list[(i-1) % len(rgb_list)], marker='o', ms=5, ls='-', zorder=5 if i == 0 else 3)
            ax.set(xlabel="Voltage (V)", ylabel=ylabel, title=title); ax.grid(True); ax.legend(); fig.tight_layout()
            figs.append((fig, title, fname))
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        for i, name in enumerate(species[1:]):
            ax3.plot(voltage, sensitivities[i+1], label=name, color=rgb_list[i % len(rgb_list)], marker='o', ms=5)
        ax3.set(xlabel="Voltage (V)", ylabel="Sensitivity (%)", title="Sensitivity Curves"); ax3.grid(True); ax3.legend(); fig3.tight_layout()
        figs.append((fig3, "Sensitivity Curves", "NEGF_Sensitivity_Curves.png"))
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        selected_mol_name = self.target_species_var.get() or (self.get_selected_molecule_params() or {}).get('name')
        if selected_mol_name and selected_mol_name in species:
            idx = species.index(selected_mol_name); S_target = sensitivities[idx]
            peaks, _ = find_peaks(S_target)
            if len(S_target) >= 5 and S_target[0] >= np.max(S_target[:5]): peaks = np.union1d(peaks, [0])
            ax4.plot(voltage, S_target, label=f"Sensitivity ({selected_mol_name})", color="steelblue", marker='o', ms=5)
            if len(peaks) > 0:
                closest_peaks = peaks[np.argsort(np.abs(voltage[peaks]))[:3]]
                optimal_voltage = voltage[closest_peaks[0]]
                self.operating_voltage_var.set(f"{optimal_voltage:.4f}")
                print(f"Optimal voltage of {optimal_voltage:.4f} V for {selected_mol_name} set as Operating Voltage.")
                ax4.scatter(voltage[closest_peaks], S_target[closest_peaks], color='indianred', zorder=10, label='Local Maxima')
                for i in closest_peaks: ax4.annotate(f'({voltage[i]:.2f}V, {S_target[i]:.2f}%)', (voltage[i], S_target[i]), textcoords="offset points", xytext=(0,5), ha='center')
            else: ax4.text(0.5, 0.5, "No peaks found", ha='center', va='center', fontsize=12)
            ax4.set_title(f"Peak Analysis for {selected_mol_name}")
        else: ax4.text(0.5, 0.5, "Select a molecule from the table\n(must exist in CSV) for peak analysis.", ha='center', va='center', fontsize=12); ax4.set_title("Peak Analysis")
        ax4.set(xlabel="Voltage (V)", ylabel="Sensitivity (%)"); ax4.grid(True); ax4.legend(); fig4.tight_layout()
        figs.append((fig4, "Peak Analysis", f"NEGF_Peak_Analysis_{selected_mol_name}.png"))
        return figs

    def load_csv_dialog(self):
        fp = filedialog.askopenfilename(title="Load Molecule Parameters", filetypes=(("CSV files", "*.csv"),))
        if fp: self.load_csv(fp)

    def load_csv(self, file_path=None):
        if file_path is None: file_path = self.csv_file_path
        if not os.path.exists(file_path):
            if file_path == self.csv_file_path: messagebox.showerror("Error", f"Default CSV not found: {os.path.basename(file_path)}")
            return
        self.csv_file_path = file_path
        try:
            with open(file_path, 'r', newline='', encoding='utf-8') as f: self.headers=next(csv.reader(f)); self.parameters=[row for row in csv.reader(f)]
            self.display_data(); self.update_target_species_combo()
        except Exception as e: messagebox.showerror("Error Loading CSV", f"An error occurred: {e}")

    def display_data(self):
        self.tree.delete(*self.tree.get_children()); self.tree["columns"] = self.headers
        for col in self.headers: self.tree.heading(col, text=col); self.tree.column(col, width=80, anchor='center')
        for i, row in enumerate(self.parameters): self.tree.insert("", "end", iid=i, values=row)

    def update_target_species_combo(self):
        species_list = [row[0] for row in self.parameters if row and row[0]]
        self.target_species_combo['values'] = species_list
        if species_list: self.target_species_var.set(species_list[0])

    def on_double_click(self, event):
        if self.tree.identify("region", event.x, event.y) != "cell": return
        item_id, col_id = self.tree.focus(), self.tree.identify_column(event.x); col_idx = int(col_id.replace('#',''))-1
        x,y,w,h = self.tree.bbox(item_id, col_id); val = self.tree.item(item_id, "values")[col_idx]
        entry = ttk.Entry(self.tree); entry.place(x=x,y=y,width=w,height=h); entry.insert(0, val); entry.focus()
        def save_edit(e):
            new_val = entry.get(); curr = list(self.tree.item(item_id, "values")); curr[col_idx] = new_val
            self.tree.item(item_id, values=curr); self.parameters[int(item_id)][col_idx] = new_val; entry.destroy()
        entry.bind("<FocusOut>", save_edit); entry.bind("<Return>", save_edit)

    def save_csv(self):
        save_path = filedialog.asksaveasfilename(title="Save As", filetypes=(("CSV files", "*.csv"),), defaultextension=".csv", initialfile=os.path.basename(self.csv_file_path))
        if not save_path: return
        self.csv_file_path = save_path
        try:
            with open(self.csv_file_path, 'w', newline='', encoding='utf-8') as f: w = csv.writer(f); w.writerow(self.headers); w.writerows(self.parameters)
            messagebox.showinfo("Success", f"Data saved to {self.csv_file_path}")
        except Exception as e: messagebox.showerror("Error Saving CSV", f"An error occurred: {e}")

    def add_row(self): self.parameters.append(['' for _ in self.headers]); self.display_data(); self.update_target_species_combo()
    def delete_row(self):
        if not self.tree.selection(): messagebox.showwarning("No Selection", "Please select row(s) to delete."); return
        for item_id in sorted(self.tree.selection(), key=int, reverse=True): del self.parameters[int(item_id)]
        self.display_data(); self.update_target_species_combo()

    def get_selected_molecule_params(self):
        if not self.tree.selection(): return None
        return dict(zip(self.headers, self.parameters[int(self.tree.selection()[0])]))
    def get_all_molecule_params(self):
        return [dict(zip(self.headers, row)) for row in self.parameters]

if __name__ == "__main__":
    app = SensingApp()
    app.mainloop()