import os
import re
import sys
import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from scipy.optimize import curve_fit, minimize_scalar

uncertainty_alpha = 0.15

def format_scientific(value):
    """Format a number in scientific notation as LaTeX a \times 10^b"""
    formatted = "{:.2e}".format(value)
    parts = formatted.split("e")
    base = float(parts[0])
    exponent = int(parts[1])
    return rf"{base:.2f} \times 10^{{{exponent}}}"

def exponential(t, I0, tau):
    return I0 * np.exp(-t / tau)

def theoretical_current(t, V, R, C):
    I0 = V / R
    tau = R * C
    return I0 * np.exp(-t / tau) * 1e6

def parse_filename(filename):
    pattern = r"R=([\d,]+)M\.C=([\d,]+)\.V=([\d,\.]+)"
    match = re.search(pattern, filename)
    if match:
        R = float(match.group(1).replace(",", ".")) * 1e6
        C = float(match.group(2).replace(",", ".")) * 1e-6
        V = float(match.group(3).replace(".", "").replace(",", "."))
        print(f"{filename} : R = {R} ; C = {C} ; V = {V}")
        return R, C, V
    return None, None, None

def error_function_shift(k, t, I, V, R, C):
    shifted_t = t + k
    theoretical_values = theoretical_current(shifted_t, V, R, C)
    error = np.sum((I - theoretical_values) ** 2)
    return error

def find_best_shift(t, I, V, R, C):
    result = minimize_scalar(error_function_shift, args=(t, I, V, R, C))
    return result.x


def plot_data(file, log_scale=False):
    R, C, V = parse_filename(file)
    if R is None or C is None or V is None:
        print(f"Could not parse R, C, V from filename: {file}")
        return

    # Load data
    data = np.genfromtxt(file, delimiter=',', skip_header=1)
    t = data[:, 0]
    s_t = data[:, 1]

    # esto es muy sucio, pero funciona despues de mi calculo de la incertidumbre
    # Ver la memoria para entender esto -> ../memoria/main.tex

    s_t = np.full(len(t), 0.5001)

    I = data[:, 2]
    s_I = data[:, 3]

    t -= t[0]

    # Remove zero or negative values (logarithm issues)
    valid_indices = I > 0
    t = t[valid_indices]
    I = I[valid_indices]
    s_t = s_t[valid_indices]
    s_I = s_I[valid_indices]

    # Find the best horizontal shift to align experimental data with the theoretical prediction
    optimal_k = find_best_shift(t, I, V, R, C)
    print(f"Optimal shift for {file}: {optimal_k:.4f} seconds")
    t_shifted = t + optimal_k
    t += optimal_k

    # Generate a distinct color for this dataset
    color = plt.cm.tab10(len(plt.gca().get_lines()) // 3)

    # Transform current to logarithmic scale
    ln_I = np.log(I)

    # Perform linear fit in logarithmic space
    slope, intercept = np.polyfit(t, ln_I, 1)
    I0_fit = np.exp(intercept)
    tau_fit = -1 / slope

    # Calculate the fit uncertainty (standard deviation of parameters)
    popt, pcov = curve_fit(exponential, t, I, sigma=s_I, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))  # Standard deviations of the parameters
    I0_err, tau_err = perr

    # Calculate R² for the fit
    ln_I_fit = intercept + slope * t
    mean = np.mean(ln_I)
    r2_exp = 1 - np.sum((ln_I - ln_I_fit) ** 2) / np.sum((ln_I - mean) ** 2)

    # Calculate theoretical curve
    t_theoretical = np.linspace(t[0], t[-1], 300)
    I_theoretical = theoretical_current(t_theoretical, V, R, C)

    # Prepare legend items depending on the plot scale
    if log_scale:
        # Logarithmic plot notation: log(I) = at + b
        label_data = rf"Datos : R={R/1e6:.1f}$M \Omega$, C={C*1e6:.1f}$\mu F$, V={V}$V$"
        label_fit = rf"Ajuste : $\log(I) = {format_scientific(intercept)} - {format_scientific(1/tau_fit)}t$ ; $R^2$={r2_exp:.4f}"
        label_theor = rf"Teoría : $\log(I) = {format_scientific(np.log(V/R))} - {format_scientific(1/(R*C))}t$ ; $R^2$={r2_exp:.4f}"
    else:
        # Linear plot notation: I = I0 * exp(-t/tau)
        label_data = f"Datos : R={R/1e6:.1f}$M \Omega$, C={C*1e6:.1f}$\mu F$, V={V}$V$"
        label_fit = rf"Ajuste : ${format_scientific(I0_fit)}e^{{-{format_scientific(1/tau_fit)}t}}$ ; $R^2$={r2_exp:.4f}"
        label_theor = rf"Teoría : ${format_scientific(V/R)}e^{{-{format_scientific(1/(R*C))}t}}$ ; $R^2$={r2_exp:.4f}"

    # Datos experimentales con barras de error
    exp_line = plt.errorbar(t, I, xerr=s_t, yerr=s_I, fmt='o', color=color, label=label_data, markersize=4)

    # Plot experimental best fit line (dashed)
    t_fit = np.linspace(t[0], t[-1], 300)
    I_fit = exponential(t_fit, I0_fit, tau_fit)

    # Calculate uncertainty bounds
    I_fit_upper = exponential(t_fit, I0_fit + I0_err, tau_fit + tau_err)
    I_fit_lower = exponential(t_fit, I0_fit - I0_err, tau_fit - tau_err)

    # Plot the shaded uncertainty region
    plt.fill_between(t_fit, I_fit_lower, I_fit_upper, color=color, alpha=uncertainty_alpha)

    # Plot the fit line (dashed)
    fit_line, = plt.plot(t_fit, I_fit, linestyle='dashed', color=color, alpha=0.7, label=label_fit)

    # Plot theoretical prediction line (solid)
    theor_line, = plt.plot(t_theoretical, I_theoretical, linestyle='-', color=color, alpha=0.4, label=label_theor)

    # Use the normal legend to display all elements
    plt.legend(loc='upper right', ncol=1)


from matplotlib.ticker import ScalarFormatter

def generate_plot(log_scale=False, filename="capacitor_discharge.png"):
    plt.figure(figsize=(16, 11))
    for file in files:
        plot_data(file, log_scale)

    plt.xlabel("Tiempo ($s$)")
    plt.ylabel("Intensidad ($\mu A)$")
    plt.title("")

    if log_scale:
        # Logarithmic plot with both major and minor grids
        plt.yscale("log")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.minorticks_on()
        # Ensure scientific notation for logarithmic scale
        plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        plt.gca().yaxis.get_major_formatter().set_powerlimits((0, 0))

        plt.savefig(f"media/figure-log.png", bbox_inches='tight', dpi=600)

    else:
        # No grid lines for the normal plot
        plt.grid(False)

        plt.savefig(f"media/figure.png", bbox_inches='tight', dpi=600)

    # Simple traditional legend in the upper right corner
    plt.legend(loc='upper right', fontsize='medium', frameon=True)

    plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <file or directory>")
        sys.exit(1)

    path = sys.argv[1]

    global files
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".csv")]
    elif os.path.isfile(path):
        files = [path]
    else:
        print("Invalid file or directory")
        sys.exit(1)

    generate_plot(log_scale=False)
    generate_plot(log_scale=True)

if __name__ == "__main__":
    main()

