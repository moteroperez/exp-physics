import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Fixed values for R, C, V

def f(t, V, R, C):
    return V * (1 / R) * np.exp(-t * (1 / R) * (1 / C))

def plot(ax, X, Y, Z, points_x, points_y, points_z, var, cte_R, cte_C, cte_V):
    # Create surface plot
    surf = ax.plot_surface(X, Y, Z, cmap='jet', edgecolor='k', alpha=0.6, linewidth=0.2, label='Teoría', zorder=1)

    # Add scatter points
    ax.scatter(points_x, points_y, points_z, color='red', marker='o', s=4, label='Experimental', alpha=1.0, zorder=2)

    # Set axis labels
    ax.set_xlabel('t ($s$)', fontsize=4, labelpad=-9)
    
    # Custom y-axis label based on varied parameter
    if var == 'R':
        ax.set_ylabel(r'R ($M \Omega$)', fontsize=4, labelpad=-9)
        varied_label = 'R'
        constant_values = rf'C = {cte_C}$\mu F$, V = {cte_V}$V$'
    elif var == 'C':
        ax.set_ylabel('C (µF)', fontsize=4, labelpad=-9)
        varied_label = 'C'
        constant_values = rf'R = {cte_R}$M\Omega$, V={cte_V}$V$'
    elif var == 'V':
        ax.set_ylabel('V ($V$)', fontsize=4, labelpad=-9)
        varied_label = 'V'
        constant_values = rf'R = {cte_R}$M\Omega$, C = {cte_C}$\mu F$'

    ax.set_zlabel(rf'$I(t, {varied_label})$ ($\mu A$)', fontsize=4, labelpad=-9)

    # Add legend with constant values
    ax.legend([f'$I(t, {varied_label})$', f'{constant_values}'], loc='upper right', fontsize=4)

    # Add title indicating the varied parameter
    ax.set_title(f'Intensidad en función de t y {varied_label}', fontsize=4)

    ax.set_facecolor('white')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    # Thinner and more minimal grid
    grid_style = {'color': 'lightgray', 'linestyle': '--', 'linewidth': 0.3}
    ax.xaxis._axinfo['grid'].update(grid_style)
    ax.yaxis._axinfo['grid'].update(grid_style)
    ax.zaxis._axinfo['grid'].update(grid_style)

    # Thinner and less intense axis lines
    ax.xaxis.line.set_color('dimgray')
    ax.yaxis.line.set_color('dimgray')
    ax.zaxis.line.set_color('dimgray')

    # Subtle and fine ticks
    ax.tick_params(axis='both', which='major', labelsize=3.5, color='dimgray', width=0.3)
    ax.tick_params(axis='x', pad=-4, color='dimgray', width=0.2)
    ax.tick_params(axis='y', pad=-4, color='dimgray', width=0.2)
    ax.tick_params(axis='z', pad=-4, color='dimgray', width=0.2)

def parse_filename(filename):
    # Remove file extension
    base_name = os.path.splitext(filename)[0]

    # Split the filename into components
    parts = base_name.split('.')

    R, C, V = None, None, None
    for part in parts:
        if part.startswith("R="):
            R = float(part[2:].replace('M', '').replace(',', '.'))
        elif part.startswith("C="):
            C = float(part[2:].replace(',', '.'))
        elif part.startswith("V="):
            V = float(part[2:].replace(',', '.'))

    return R, C, V

def read_folder(path, var):
    data = []

    cte_R = None
    cte_C = None
    cte_V = None

    for filename in os.listdir(path):
        R, C, V = parse_filename(filename)

        if cte_R is None:
            cte_R = R
        if cte_C is None:
            cte_C = C
        if cte_V is None:
            cte_V = V

        y = 0
        if var == 'R':
            y = R
        elif var == 'C':
            y = C
        elif var == 'V':
            y = V

        if filename.endswith('.csv'):
            file_path = os.path.join(path, filename)
            t_values = []
            I_values = []

            # Open and read the CSV file
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                next(reader)    # skip header
                for row in reader:
                    if not row:
                        continue
                    try:
                        t = float(row[0])  # Read t
                        I = float(row[2])  # Read I
                        t_values.append(t)
                        I_values.append(I)
                    except (ValueError, IndexError):
                        print(f"Skipping invalid row in file {filename}: {row}")


            offset = t_values[0]
            for i in range(len(t_values)):
                t_values[i] -= offset

            data.append((t_values, I_values, [y] * len(t_values)))

    return data, cte_R, cte_C, cte_V

def flatten_data(data):
    x, y, z = [], [], []
    for t_values, I_values, y_values in data:
        x.extend(t_values)
        y.extend(y_values)
        z.extend(I_values)
    return x, y, z

def calculate_offset(value, factor=0.1):
    return value * (1 + factor)

# Create a figure with three subplots side by side
fig, axs = plt.subplots(1, 3, subplot_kw={'projection': '3d'}, figsize=(15, 5))

# Plotting for varying R
var_R, cte_R, cte_C, cte_V = read_folder("raw/var-r", 'R')
points_x, points_y, points_z = flatten_data(var_R)
t_min, t_max = min(points_x), calculate_offset(max(points_x))
y_min, y_max = min(points_y), calculate_offset(max(points_y))
t = np.linspace(t_min, t_max, 300)
y = np.linspace(y_min, y_max, 300)
T, Y = np.meshgrid(t, y)
Z = f(T, cte_V, Y, cte_C)
plot(axs[0], T, Y, Z, points_x, points_y, points_z, 'R', cte_R, cte_C, cte_V)

# Plotting for varying C
var_C, cte_R, cte_C, cte_V = read_folder("raw/var-c", 'C')
points_x, points_y, points_z = flatten_data(var_C)
t_min, t_max = min(points_x), calculate_offset(max(points_x))
y_min, y_max = min(points_y), calculate_offset(max(points_y))
t = np.linspace(t_min, t_max, 300)
y = np.linspace(y_min, y_max, 300)
T, Y = np.meshgrid(t, y)
Z = f(T, cte_V, cte_R, Y)
plot(axs[1], T, Y, Z, points_x, points_y, points_z, 'C', cte_R, cte_C, cte_V)

# Plotting for varying V
var_V, cte_R, cte_C, cte_V = read_folder("raw/var-v", 'V')
points_x, points_y, points_z = flatten_data(var_V)
t_min, t_max = min(points_x), calculate_offset(max(points_x))
y_min, y_max = min(points_y), calculate_offset(max(points_y))
t = np.linspace(t_min, t_max, 300)
y = np.linspace(y_min, y_max, 300)
T, Y = np.meshgrid(t, y)
Z = f(T, Y, cte_R, cte_C)
plot(axs[2], T, Y, Z, points_x, points_y, points_z, 'V', cte_R, cte_C, cte_V)

# Adjust layout, save the image, and show it
plt.tight_layout()
plt.savefig("relations.png", dpi=300, bbox_inches='tight')
plt.show()
plt.savefig(f"media/var.png", bbox_inches='tight', dpi=600)
