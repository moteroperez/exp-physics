import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
import matplotlib.colors as mcolors
from scipy.signal import savgol_filter  # For smoothing derivatives

def parse_filename(filename):
    """Improved filename parsing with better error handling"""
    base_name = os.path.splitext(filename)[0]
    parts = base_name.split('.')
    
    params = {'R': None, 'C': None, 'V': None}
    for part in parts:
        for key in params:
            if part.startswith(f"{key}="):
                value = part[2:].replace('M', '').replace(',', '.')
                try:
                    params[key] = float(value)
                except ValueError:
                    print(f"Warning: Couldn't parse value in {part}")
    return params['R'], params['C'], params['V']

def calculate_derivative(t, I, smoothing=True):
    """Calculate derivative with optional adaptive smoothing"""
    if smoothing and len(I) > 5:  # Need enough points for smoothing
        # Adaptive window size: between 5 and 11, and always odd
        window_size = max(5, min(11, len(I) - 1))
        if window_size % 2 == 0:
            window_size += 1  # Ensure it's odd for Savitzky–Golay filter
        I_smooth = savgol_filter(I, window_length=window_size, polyorder=3)
        dIdt = np.gradient(I_smooth, t)
    else:
        dIdt = np.gradient(I, t)
    return dIdt

def read_folder(path, var):
    """Improved data reading with better time handling and derivative calculation"""
    data = []
    constants = {'R': None, 'C': None, 'V': None}
    
    for filename in sorted(os.listdir(path)):  # Sort files for consistent ordering
        # if filename == "R=2,2M.C=4,7.V=12,22.csv":  # maldita medicion
            # continue
        if not filename.endswith('.csv'):
            continue
            
        R, C, V = parse_filename(filename)
        for key in constants:
            if constants[key] is None:
                constants[key] = locals()[key]
        
        param_value = locals()[var]
        
        file_path = os.path.join(path, filename)
        t_values, I_values = [], []
        
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                if not row: continue
                try:
                    t_values.append(float(row[0].replace(',', '.')))
                    I_values.append(float(row[2].replace(',', '.')))
                except (ValueError, IndexError):
                    print(f"Skipping invalid row in {filename}: {row}")
        
        if len(t_values) < 2:  # Need at least 2 points for derivatives
            print(f"Warning: Not enough data in {filename}")
            continue
            
        # Convert to numpy arrays and process
        t = np.array(t_values)
        I = np.array(I_values)
        t -= t[0]  # Remove time offset
        
        # Calculate derivative
        dIdt = calculate_derivative(t, I)
        
        # Store data
        data.append({
            't': t, 'I': I, 'dIdt': dIdt,
            'param': param_value,
            'R': R, 'C': C, 'V': V
        })
    
    return data, constants['R'], constants['C'], constants['V']


def plot_phase_space(data, fixed_params, var):
    """Enhanced phase space plotting with better vector field and labeling"""
    plt.figure(figsize=(10, 10))

    R, C = fixed_params.get('R', None), fixed_params.get('C', None)

    # Determine plot ranges from data
    I_min, I_max = np.inf, -np.inf
    dIdt_min, dIdt_max = np.inf, -np.inf

    for dataset in data:
        I_min = min(I_min, np.min(dataset['I']))
        I_max = max(I_max, np.max(dataset['I']))
        dIdt_min = min(dIdt_min, np.min(dataset['dIdt']))
        dIdt_max = max(dIdt_max, np.max(dataset['dIdt']))

    # Add some padding
    I_range = I_max - I_min
    dIdt_range = dIdt_max - dIdt_min
    I_min, I_max = I_min - 0.1 * I_range, I_max + 0.1 * I_range
    dIdt_min, dIdt_max = dIdt_min - 0.1 * dIdt_range, dIdt_max + 0.1 * dIdt_range

    # Create grid for vector field
    grid_points = 20
    I_grid = np.linspace(I_min, I_max, grid_points)
    dIdt_grid = np.linspace(dIdt_min, dIdt_max, grid_points)
    I_mesh, dIdt_mesh = np.meshgrid(I_grid, dIdt_grid)

    # Corrected vector field calculation: Only X-component is non-zero

    # Corrected vector field calculation: Always pointing towards the origin

    # U = -I_mesh
    # V = -dIdt_mesh

    U = dIdt_mesh
    V = -dIdt_mesh/(R*C)

    # Plot vector field with thinner arrows

    if var == 'V':
        s = 50
    elif var == 'C':
        s = 1
    elif var == 'R':
        s = 0.1

    plt.quiver(I_mesh, dIdt_mesh, U, V,
               color='lightblue', scale=s, angles='xy', width=0.0025,
               alpha=0.8, label='Flujo Teórico')

    # Plot experimental trajectories with better labeling and visibility
    texts = []
    colors = ['blue', 'orange', 'green', 'red']
    for i, dataset in enumerate(data):
        label = rf"{var} = {dataset['param']}{'$M \Omega$' if var=='R' else '$\mu F$' if var=='C' else '$V$'}"
        plt.plot(dataset['I'], dataset['dIdt'], '.-', 
                 markersize=5, label=label, alpha=0.9, color=colors[i % len(colors)])
        
        # Annotate the end of each trajectory without overlap
        texts.append(plt.text(dataset['I'][-1], dataset['dIdt'][-1], label, fontsize=8))

        # Plot individual theoretical prediction for each curve (darker tone of the same color)
        if var == 'C':
            Rc = R * dataset['param']
        elif var == 'R':
            Rc = dataset['param'] * C
        else:
            Rc = R * C  # Default case for var-V

        # Generate the individual theoretical curve
        if Rc:  # Only plot if Rc is valid
            I_null = np.linspace(I_min, I_max, 100)
            dIdt_null = -I_null / Rc

            # Darker color for the theoretical prediction
            # dark_color = np.array(plt.cm.get_cmap('Dark2')(i % 4)) * 0.7

            color = colors[i % len(colors)]
            color = mcolors.to_rgb(color)
            dark_color = tuple(max(0, min(1, c * 0.75)) for c in color)


            if not var == 'V':
                plt.plot(I_null, dIdt_null, linestyle='--', color=dark_color, alpha=0.8,
                         label=f'Teoría {label}')

    # Automatically adjust text to avoid overlap
    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='black'))

    if var == 'V':
        # Plot global theoretical nullcline for reference
        if R and C:
            I_null = np.linspace(I_min, I_max, 100)
            dIdt_null = -I_null / (R * C)
            plt.plot(I_null, dIdt_null, 'k--', 
                     label=r'Teoría General : $\frac{dI}{dt} = -\frac{I}{RC}$', alpha=0.7)

    # Formatting
    plt.xlabel(r'$I$ ($\mu A$)', fontsize=12)
    plt.ylabel(r'$\frac{dI}{dt}$ ($\mu A \cdot s^{-1}$)', fontsize=12)

    title_parts = []
    if var != 'R':
        title_parts.append(rf"R = {fixed_params['R']}$M \Omega$")
    if var != 'C':
        title_parts.append(rf"C = {fixed_params['C']}$\mu F$")
    if var != 'V':
        title_parts.append(f"V = {fixed_params['V']}$V$")

    plt.title(r"Espacio de fase $(I, \frac{dI}{dt})$" + f" ({', '.join(title_parts)})", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(f"media/phasespace-{var}.png", bbox_inches='tight', dpi=600)


def analyze_var(folder_path, var):
    """Automated analysis for var-c, var-r, var-v"""
    data, R, C, V = read_folder(folder_path, var)
    
    # Asegurarse de que las claves siempre estén presentes en el diccionario
    fixed_params = {'R': R, 'C': C, 'V': V}
    
    # Eliminar claves que sean None
    fixed_params = {key: value for key, value in fixed_params.items() if value is not None}

    # Comprobar que haya suficientes parámetros para graficar
    if len(fixed_params) < 2:
        print(f"Error: No hay suficientes parámetros fijos para graficar el caso var-{var}")
        return

    plot_phase_space(data, fixed_params, var)

# Análisis para var-v (Voltaje variable)
folder_path_v = "raw/var-v"
print("Análisis para var-v (Voltaje variable)")
analyze_var(folder_path_v, 'V')

# Análisis para var-c (Capacitancia variable)
folder_path_c = "raw/var-c"
print("Análisis para var-c (Capacitancia variable)")
analyze_var(folder_path_c, 'C')

# Análisis para var-r (Resistencia variable)
folder_path_r = "raw/var-r"
print("Análisis para var-r (Resistencia variable)")
analyze_var(folder_path_r, 'R')

