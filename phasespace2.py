import os
import csv
import numpy as np
import matplotlib.pyplot as plt
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


def plot_phase_space(data, fixed_params, var, ax):
    """Enhanced phase space plotting with better vector field and labeling"""

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
    U = -I_mesh
    V = -dIdt_mesh

    # Plot vector field with thinner arrows

    if var == 'V':
        s = 200
    elif var == 'C':
        s = 100
    elif var == 'R':
        s = 50

    ax.quiver(I_mesh, dIdt_mesh, U, V,
               color='lightblue', scale=s, angles='xy', width=0.0025,
               alpha=0.8, label='Flujo Teórico')

    # Plot experimental trajectories with better labeling and visibility
    # texts = []
    colors = ['blue', 'orange', 'green', 'red']
    for i, dataset in enumerate(data):
        label = rf"{var} = {dataset['param']}{'$M \Omega$' if var=='R' else '$\mu F$' if var=='C' else '$V$'}"
        ax.plot(dataset['I'], dataset['dIdt'], '.-', 
                 markersize=5, label=label, alpha=0.9, color=colors[i % len(colors)])
        
        # Annotate the end of each trajectory without overlap
        # texts.append(ax.text(dataset['I'][-1], dataset['dIdt'][-1], label, fontsize=8))

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
            # dark_color = np.array(ax.cm.get_cmap('Dark2')(i % 4)) * 0.7

            color = colors[i % len(colors)]
            color = mcolors.to_rgb(color)
            dark_color = tuple(max(0, min(1, c * 0.75)) for c in color)


            if not var == 'V':
                ax.plot(I_null, dIdt_null, linestyle='--', color=dark_color, alpha=0.8,
                         label=f'Teoría {label}')


    if var == 'V':
        # Plot global theoretical nullcline for reference
        if R and C:
            I_null = np.linspace(I_min, I_max, 100)
            dIdt_null = -I_null / (R * C)
            ax.plot(I_null, dIdt_null, 'k--', 
                     label=r'Teoría General : $\frac{dI}{dt} = -\frac{I}{RC}$', alpha=0.7)

    # Formatting
    ax.set_xlabel(r'$I$ ($\mu A$)', fontsize=12)
    ax.set_ylabel(r'$\frac{dI}{dt}$ ($\mu A \cdot s^{-1}$)', fontsize=12)

    title_parts = []
    if var != 'R':
        title_parts.append(rf"R = {fixed_params['R']}$M \Omega$")
    if var != 'C':
        title_parts.append(rf"C = {fixed_params['C']}$\mu F$")
    if var != 'V':
        title_parts.append(f"V = {fixed_params['V']}$V$")

    ax.set_title(r"Espacio de fase $(I, \frac{dI}{dt})$" + f" ({', '.join(title_parts)})", fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend()

def analyze_var(folder_path, var, ax):
    """Automated analysis for var-c, var-r, var-v"""
    data, R, C, V = read_folder(folder_path, var)
    
    # Prepare fixed parameters
    fixed_params = {'R': R, 'C': C, 'V': V}
    fixed_params = {key: value for key, value in fixed_params.items() if value is not None}

    # Plot on the given axis
    plot_phase_space(data, fixed_params, var, ax)
    plt.savefig(f"media/phasespace.png", bbox_inches='tight', dpi=600)



# Main

# Create a single figure with three subplots
fig, axs = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.10, wspace=0.25)


# Análisis para var-v (Voltaje variable)
folder_path_v = "raw/var-v"
print("Análisis para var-v (Voltaje variable)")
analyze_var(folder_path_v, 'V', axs[0])

# Análisis para var-c (Capacitancia variable)
folder_path_c = "raw/var-c"
print("Análisis para var-c (Capacitancia variable)")
analyze_var(folder_path_c, 'C', axs[1])

# Análisis para var-r (Resistencia variable)
folder_path_r = "raw/var-r"
print("Análisis para var-r (Resistencia variable)")
analyze_var(folder_path_r, 'R', axs[2])

# Show the combined plot
plt.tight_layout()
plt.show()

