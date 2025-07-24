import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.scrolledtext import ScrolledText
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- DISCLAIMER ---
# THIS CODE IS FOR EDUCATIONAL AND PRELIMINARY DESIGN PURPOSES ONLY.
# IT SHOULD NOT BE USED FOR REAL ENGINEERING PROJECTS WITHOUT THE SUPERVISION
# AND APPROVAL OF A QUALIFIED STRUCTURAL ENGINEER.

# ==============================================================================
# SECTION 1: CALCULATION ENGINE 
# ==============================================================================
def get_material_properties(concrete_grade, steel_grade):
    fck_map = {"C25": 25, "C30": 30, "C35": 35, "C40": 40}
    # --- MODIFIED: Added new steel grades ---
    fyk_map = {"B420C": 420, "S420": 420, "S500": 500, "S550": 550}
    
    fck = fck_map.get(concrete_grade, 0)
    fyk = fyk_map.get(steel_grade, 0)
    if fck == 0 or fyk == 0: raise ValueError("Unsupported material grade!")
    
    gamma_c, gamma_s = 1.5, 1.15
    fcd, fyd = fck / gamma_c, fyk / gamma_s
    E_s = 200000.0
    fctk = 0.35 * np.sqrt(fck)
    fctd = fctk / gamma_c
    return fcd, fyd, fck, fyk, E_s, fctd

# (Other calculation functions remain the same)
def calculate_mn_interaction(diameter, cover, bar_diameter, num_bars, fcd, fyd, E_s):
    R = diameter / 2.0; epsilon_cu = 0.003
    rebar_circle_diameter = diameter - 2 * cover - bar_diameter; rebar_R = rebar_circle_diameter / 2.0
    angle_step = 2 * np.pi / num_bars; area_single_bar = np.pi * (bar_diameter / 2.0)**2
    total_steel_area = num_bars * area_single_bar
    bar_y_positions = [rebar_R * np.cos(i * angle_step) for i in range(num_bars)]
    c_values = list(np.linspace(0.001, diameter * 2, 200)) + [1e9]
    moment_capacity_list, axial_capacity_list = [], []
    num_strips = 200; dy = diameter / num_strips; y_strips = np.linspace(-R + dy/2, R - dy/2, num_strips)
    for c in c_values:
        Fc_concrete, Mc_concrete = 0, 0; a = 0.85 * c
        for y_strip in y_strips:
            dist_from_top = R - y_strip
            if 0 < dist_from_top <= a:
                strip_width = 2 * np.sqrt(R**2 - y_strip**2)
                dFc = 0.85 * fcd * strip_width * dy; dMc = dFc * y_strip
                Fc_concrete += dFc; Mc_concrete += dMc
        Fs_steel, Ms_steel = 0, 0
        for y_bar in bar_y_positions:
            dist_from_top = R - y_bar
            epsilon_s = epsilon_cu * (c - dist_from_top) / c if c > 0 else 0
            sigma_s = np.clip(epsilon_s * E_s, -fyd, fyd)
            concrete_stress_at_bar = 0.85 * fcd if dist_from_top <= a else 0
            dFs = area_single_bar * (sigma_s - concrete_stress_at_bar); dMs = dFs * y_bar
            Fs_steel += dFs; Ms_steel += dMs
        axial_capacity_list.append((Fc_concrete + Fs_steel) / 1000.0)
        moment_capacity_list.append((Mc_concrete + Ms_steel) / 1e6)
    return moment_capacity_list, axial_capacity_list, total_steel_area

def design_longitudinal_reinforcement(D, Nd, Md, fcd, fyd, E_s, cover, user_selected_diameter, log_func, case_name=""):
    log_func(f"---> [{case_name}] Searching for required number of Ø{int(user_selected_diameter)} bars for Nd={Nd:.0f} kN, Md={Md:.0f} kNm...")
    A_g = np.pi * (D / 2)**2; rho_min, rho_max = 0.01, 0.04
    bar_diameter = user_selected_diameter; area_single_bar = np.pi * (bar_diameter / 2.0)**2
    min_count = int(np.ceil(rho_min * A_g / area_single_bar))
    if min_count % 2 != 0: min_count += 1
    if min_count < 6: min_count = 6
    for num_bars in range(min_count, 200, 2):
        total_steel_area = num_bars * area_single_bar; rho_t = total_steel_area / A_g
        if rho_t > rho_max:
            log_func(f"    - WARNING: Max reinforcement ratio exceeded for Ø{int(bar_diameter)}. No solution found with this diameter.")
            return None
        M_k, N_k, _ = calculate_mn_interaction(D, cover, bar_diameter, num_bars, fcd, fyd, E_s)
        n_unique, idx_unique = np.unique(N_k, return_index=True); m_unique = np.array(M_k)[idx_unique]
        sort_indices = np.argsort(n_unique); n_sorted, m_sorted = n_unique[sort_indices], m_unique[sort_indices]
        if not (min(n_sorted) <= Nd <= max(n_sorted)): continue
        M_capacity_at_Nd = np.interp(Nd, n_sorted, m_sorted)
        if M_capacity_at_Nd >= abs(Md):
            solution = {'Ast': total_steel_area, 'diameter': bar_diameter, 'count': num_bars}
            log_func(f"    - Found solution: {solution['count']} Ø{int(solution['diameter'])} (Ast={solution['Ast']:.0f} mm²)")
            return solution
    log_func(f"    - WARNING: No solution found for [{case_name}] with Ø{int(bar_diameter)} bars.")
    return None

def calculate_shear_reinforcement(D, cover, tie_diameter, fcd, fctd, fck, fyk, Vd, Nd, is_seismic, main_bar_diameter, log_func, case_name=""):
    log_func(f"---> [{case_name}] Calculating transverse reinforcement (Vd={Vd:.0f} kN)...")
    d=0.8*D; Ac=np.pi*(D/2)**2; fydw=fyk/1.15
    V_max=0.22*fcd*Ac/1000
    if Vd>V_max:
        log_func(f"    - ERROR: Design shear Vd={Vd:.1f} kN exceeds max capacity Vmax={V_max:.1f} kN. SECTION IS NOT ADEQUATE!")
        return None,"Section Failed"
    Vc=0.65*fctd*Ac/1000
    Vc_modified = Vc * (1 + 0.07 * (Nd * 1000) / Ac) if Nd > 0 else 0
    if is_seismic: Vc_modified = 0
    Asw_s_ratio_shear = 0
    if Vd > Vc_modified:
        Vw_required = Vd - Vc_modified; Asw_s_ratio_shear = (Vw_required * 1000) / (d * fydw)
    Asw_s_ratio_min = 0.20 * (Ac / 1000) * (fctd / fydw)
    Asw_s_ratio_required = max(Asw_s_ratio_shear, Asw_s_ratio_min)
    Ash = np.pi * (tie_diameter/2)**2; Asw = 2 * Ash
    spacing_from_shear = Asw / Asw_s_ratio_required if Asw_s_ratio_required > 0 else float('inf')
    spacing_from_confinement = float('inf')
    if is_seismic:
        D_k=D-2*cover; A_ck=np.pi*(D_k/2)**2; f_ykh=fyk
        rho_sh_1=0.45*(Ac/A_ck-1)*(fck/f_ykh); rho_sh_2=0.12*(fck/f_ykh)
        rho_sh_required = max(rho_sh_1, rho_sh_2); spacing_from_confinement = (4 * Ash) / (D_k * rho_sh_required)
    if is_seismic:
        D_k=D-2*cover; s_max1,s_max2,s_max3,s_min = D_k/5.0,150.0,6*main_bar_diameter,50.0
        s_max_limit=min(s_max1,s_max2,s_max3)
    else:
        s_max1,s_max2,s_min = 12*main_bar_diameter,200.0,75.0
        s_max_limit=min(s_max1,s_max2)
    final_spacing = min(spacing_from_shear, spacing_from_confinement, s_max_limit)
    final_spacing = max(final_spacing, s_min)
    final_spacing = np.floor(final_spacing/5)*5
    log_func(f"    - Calculated tie spacing: {int(final_spacing)} mm")
    return final_spacing, "OK"

# ==============================================================================
# SECTION 2: PLOTTING FUNCTIONS
# ==============================================================================
def plot_final_design_on_canvas(fig, ax, M_k, N_k, loads, final_reinf_str):
    ax.clear(); ax.plot(M_k, N_k, color="blue"); ax.plot([-m for m in M_k], N_k, color="blue")
    ax.fill_betweenx(N_k, 0, M_k, color='lightblue', alpha=0.5, label=f"Capacity ({final_reinf_str})")
    ax.fill_betweenx(N_k, 0, [-m for m in M_k], color='lightblue', alpha=0.5)
    ax.plot(loads['Md_n'], loads['Nd_n'], 'go', markersize=8, label=f"Normal Case")
    ax.plot(loads['Md_s'], loads['Nd_s'], 'ro', markersize=8, label=f"Seismic Case")
    ax.set_title(f"Final Design M-N Diagram", fontsize=10)
    ax.set_xlabel("Moment (kNm)", fontsize=8); ax.set_ylabel("Axial Load (kN)", fontsize=8)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.grid(True, linestyle='--', alpha=0.6); ax.legend(fontsize=8); fig.tight_layout(); fig.canvas.draw()

def draw_final_cross_section_on_canvas(fig, ax, D, cover, bar_diameter, num_bars, tie_diameter, spacing):
    ax.clear(); ax.add_patch(plt.Circle((0, 0), D/2, facecolor='lightgray', edgecolor='black'))
    ax.add_patch(plt.Circle((0, 0), (D - 2*cover - tie_diameter)/2, facecolor='none', edgecolor='black', linestyle='--'))
    rebar_R = (D - 2*cover - bar_diameter) / 2.0
    for i in range(num_bars):
        angle = i * (2 * np.pi / num_bars); x, y = rebar_R * np.sin(angle), rebar_R * np.cos(angle)
        ax.add_patch(plt.Circle((x, y), bar_diameter/2, facecolor='black'))
    ax.set_aspect('equal', adjustable='box'); ax.set_xlim(-D/2 * 1.1, D/2 * 1.1); ax.set_ylim(-D/2 * 1.1, D/2 * 1.1)
    title_str = (f"Final Cross-Section Design\n"
                 f"{num_bars}Ø{int(bar_diameter)} | Ties Ø{int(tie_diameter)}@{int(spacing)}mm")
    ax.set_title(title_str, fontsize=10); ax.tick_params(axis='both', which='major', labelsize=8)
    fig.tight_layout(); fig.canvas.draw()

# ==============================================================================
# SECTION 3: TKINTER GUI CLASS 
# ==============================================================================

class PileDesignerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Drilled Pile (Bored Pile) Envelope Design Tool")
        self.geometry("1100x800")
        main_frame = ttk.Frame(self, padding="10"); main_frame.pack(fill=tk.BOTH, expand=True)
        left_frame = ttk.Frame(main_frame); left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        right_frame = ttk.Frame(main_frame); right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.create_input_widgets(left_frame)
        self.create_output_widgets(right_frame)

    def create_input_widgets(self, parent):
        general_frame = ttk.LabelFrame(parent, text="General Properties", padding="10")
        general_frame.pack(fill=tk.X, pady=5)
        
        self.inputs = {}
        i = 0 # Row counter

              
        # Pile Diameter and Cover (Entry widgets)
        entries_info = {"diameter": "Pile Diameter (mm)", "cover": "Concrete Cover (mm)"}
        for key, label in entries_info.items():
            ttk.Label(general_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=2)
            self.inputs[key] = tk.StringVar()
            ttk.Entry(general_frame, textvariable=self.inputs[key], width=12).grid(row=i, column=1, sticky=tk.EW, pady=2)
            i += 1
        
        # Material Grades
        ttk.Label(general_frame, text="Concrete Grade").grid(row=i, column=0, sticky=tk.W, pady=2)
        self.inputs["concrete_grade"] = tk.StringVar()
        ttk.Combobox(general_frame, textvariable=self.inputs["concrete_grade"], values=["C25", "C30", "C35", "C40"], width=10, state="readonly").grid(row=i, column=1, sticky=tk.EW)
        i += 1
        
        ttk.Label(general_frame, text="Steel Grade").grid(row=i, column=0, sticky=tk.W, pady=2)
        self.inputs["steel_grade"] = tk.StringVar()
        ttk.Combobox(general_frame, textvariable=self.inputs["steel_grade"], values=["B420C", "S420", "S500", "S550"], width=10, state="readonly").grid(row=i, column=1, sticky=tk.EW)
        i += 1

        # Longitudinal Bar Diameter
        ttk.Label(general_frame, text="Longitudinal Bar Ø (mm)").grid(row=i, column=0, sticky=tk.W, pady=2)
        self.inputs["main_bar_diameter"] = tk.StringVar()
        ttk.Combobox(general_frame, textvariable=self.inputs["main_bar_diameter"], values=[16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38 ,40], width=10, state="readonly").grid(row=i, column=1, sticky=tk.EW)
        i += 1

        
        ttk.Label(general_frame, text="Tie/Spiral Diameter (mm)").grid(row=i, column=0, sticky=tk.W, pady=2)
        self.inputs["tie_diameter"] = tk.StringVar()
        ttk.Combobox(general_frame, textvariable=self.inputs["tie_diameter"], values=[10, 12, 14, 16, 18], width=10, state="readonly").grid(row=i, column=1, sticky=tk.EW)
        i += 1

        # Load Cases
        normal_frame = ttk.LabelFrame(parent, text="Normal Case Design Loads", padding="10"); normal_frame.pack(fill=tk.X, pady=5)
        normal_entries = {"Nd_n": "Nd (kN)", "Md_n": "Md (kNm)", "Vd_n": "Vd (kN)"}
        for i, (key, label) in enumerate(normal_entries.items()):
            ttk.Label(normal_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=2)
            self.inputs[key] = tk.StringVar()
            ttk.Entry(normal_frame, textvariable=self.inputs[key], width=10).grid(row=i, column=1, sticky=tk.EW, pady=2)

        sismic_frame = ttk.LabelFrame(parent, text="Seismic Case Design Loads", padding="10"); sismic_frame.pack(fill=tk.X, pady=5)
        sismic_entries = {"Nd_s": "Nd (kN)", "Md_s": "Md (kNm)", "Vd_s": "Vd (kN)"}
        for i, (key, label) in enumerate(sismic_entries.items()):
            ttk.Label(sismic_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=2)
            self.inputs[key] = tk.StringVar()
            ttk.Entry(sismic_frame, textvariable=self.inputs[key], width=10).grid(row=i, column=1, sticky=tk.EW, pady=2)

        calc_button = ttk.Button(parent, text="Calculate", command=self.run_analysis); calc_button.pack(fill=tk.X, pady=20)

    def create_output_widgets(self, parent):
        plot_frame = ttk.Frame(parent); plot_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.fig = plt.Figure(figsize=(6, 5), dpi=100); self.ax_mn = self.fig.add_subplot(121); self.ax_cs = self.fig.add_subplot(122)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame); self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        report_frame = ttk.LabelFrame(parent, text="Calculation Report", padding="10"); report_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.report_text = ScrolledText(report_frame, wrap=tk.WORD, width=80, height=15); self.report_text.pack(fill=tk.BOTH, expand=True)
    
    def log_to_report(self, message):
        self.report_text.insert(tk.END, message + "\n"); self.report_text.see(tk.END); self.update_idletasks()

    def run_analysis(self):
        self.report_text.delete('1.0', tk.END)
        self.ax_mn.clear(); self.ax_cs.clear(); self.fig.canvas.draw()
        
        try:
            p = {key: var.get() for key, var in self.inputs.items()}
            
            if any(not value for value in p.values()):
                raise ValueError("An input field is empty.")
            
            D, cover, tie_diameter = float(p["diameter"]), float(p["cover"]), float(p["tie_diameter"])
            main_bar_diameter = float(p["main_bar_diameter"])
            loads = {'Nd_n': float(p["Nd_n"]), 'Md_n': float(p["Md_n"]), 'Vd_n': float(p["Vd_n"]),
                     'Nd_s': float(p["Nd_s"]), 'Md_s': float(p["Md_s"]), 'Vd_s': float(p["Vd_s"])}
        except ValueError as e:
            messagebox.showerror("Input Error", "Please ensure all input fields are filled with valid values before running the analysis.")
            return

        self.log_to_report("="*60 + "\nENVELOPE DESIGN ANALYSIS STARTED\n" + "="*60)
        fcd, fyd, fck, fyk, E_s, fctd = get_material_properties(p["concrete_grade"], p["steel_grade"])
        
        self.log_to_report(f"User selected longitudinal bar diameter: Ø{int(main_bar_diameter)} mm")
        self.log_to_report(f"User selected transverse tie diameter: Ø{int(tie_diameter)} mm\n")

        self.log_to_report("[A] CALCULATING REQUIRED REINFORCEMENT FOR EACH CASE...")
        sol_normal = design_longitudinal_reinforcement(D, loads['Nd_n'], loads['Md_n'], fcd, fyd, E_s, cover, main_bar_diameter, self.log_to_report, "Normal")
        sol_seismic = design_longitudinal_reinforcement(D, loads['Nd_s'], loads['Md_s'], fcd, fyd, E_s, cover, main_bar_diameter, self.log_to_report, "Seismic")
        
        if not sol_normal or not sol_seismic:
            messagebox.showerror("Design Error", f"Could not find a valid solution with Ø{int(main_bar_diameter)} bars for one or both cases.\n\nThis usually means the required steel area exceeds the maximum limit (4%).\n\nPlease try a larger bar diameter or increase the pile size.")
            return
            
        self.log_to_report("\n[B] SELECTING FINAL LONGITUDINAL REINFORCEMENT (ENVELOPE)...")
        final_solution = sol_seismic if sol_seismic['Ast'] >= sol_normal['Ast'] else sol_normal
        self.log_to_report(f"    - Normal Case Ast: {sol_normal['Ast']:.0f} mm² ({sol_normal['count']} bars) | Seismic Case Ast: {sol_seismic['Ast']:.0f} mm² ({sol_seismic['count']} bars)")
        self.log_to_report(f"    ==> GOVERNING BARS: {final_solution['count']} Ø{int(final_solution['diameter'])} (Ast={final_solution['Ast']:.0f} mm²)")
        final_bar_count = final_solution['count']
        
        self.log_to_report("\n[C] CALCULATING AND SELECTING FINAL TRANSVERSE REINFORCEMENT (ENVELOPE)...")
        spacing_n, status_n = calculate_shear_reinforcement(D, cover, tie_diameter, fcd, fctd, fck, fyk, loads['Vd_n'], loads['Nd_n'], False, main_bar_diameter, self.log_to_report, "Normal")
        spacing_s, status_s = calculate_shear_reinforcement(D, cover, tie_diameter, fcd, fctd, fck, fyk, loads['Vd_s'], loads['Nd_s'], True, main_bar_diameter, self.log_to_report, "Seismic")
        
        if status_n == "Section Failed" or status_s == "Section Failed":
             messagebox.showerror("Design Error", "Section is not adequate for shear. Consider increasing the section size.")
             return
        final_spacing = min(spacing_n, spacing_s)
        self.log_to_report(f"    ==> GOVERNING SPACING: {int(final_spacing)} mm (min of {int(spacing_n)}mm and {int(spacing_s)}mm)")
        
        self.log_to_report("\n" + "="*23 + " FINAL DESIGN SUMMARY " + "="*23)
        self.log_to_report(f"Longitudinal Bars : {final_bar_count} Ø {int(main_bar_diameter)}")
        self.log_to_report(f"Transverse Ties   : Ø{int(tie_diameter)} @ {int(final_spacing)} mm")
        
        M_k, N_k, _ = calculate_mn_interaction(D, cover, main_bar_diameter, final_bar_count, fcd, fyd, E_s)
        plot_final_design_on_canvas(self.fig, self.ax_mn, M_k, N_k, loads, f"{final_bar_count}Ø{int(main_bar_diameter)}")
        draw_final_cross_section_on_canvas(self.fig, self.ax_cs, D, cover, main_bar_diameter, final_bar_count, tie_diameter, final_spacing)
        self.log_to_report("\n" + "="*60 + "\nANALYSIS COMPLETED\n" + "="*60)

if __name__ == "__main__":
    app = PileDesignerApp()
    app.mainloop()