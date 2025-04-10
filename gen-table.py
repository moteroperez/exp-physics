import os
import re
import sys
import numpy as np


def format_scientific(value):
    formatted = "{:.2e}".format(value)
    parts = formatted.split("e")
    base = float(parts[0])
    exponent = int(parts[1])
    return rf"{base:.2f} \times 10^{{{exponent}}}"

def parse_filename(filename):
    pattern = r"R=([\d,]+)M\.C=([\d,]+)\.V=([\d,\.]+)"
    match = re.search(pattern, filename)
    if match:
        R = float(match.group(1).replace(",", ".")) * 1e6
        C = float(match.group(2).replace(",", ".")) * 1e-6
        V = float(match.group(3).replace(".", "").replace(",", "."))
        return format_scientific(R), format_scientific(C), format_scientific(V)
    return None, None, None

def gen_table(data, filename):
    """Generate a LaTeX table from the given data and save it to a file."""
    x, s_x, y, s_y = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
    s_x = np.full(len(x), 0.50)

    R, C, V = parse_filename(filename)

    caption = (
        r"\caption*{Tabla: "
        + rf"$R = {R}\,\Omega$, $C = {C}$\,F, $V = {V}$\,V"
        + r"}"
    )

    tex = r'''\begin{table}[H]
\centering
\setlength{\arrayrulewidth}{1.2pt}
\begin{tabular}{|c|c|}
\hline
$t \pm s(t) (s)$ & $I \pm s(I) (\mu A)$ \\
\hline
'''

    for i in range(len(x)):
        tex += f"${x[i]:.2f} \\pm {s_x[i]:.2f}$ & ${y[i]:.2f} \\pm {s_y[i]:.2f}$ \\\\\n"

    tex += r'''\hline
\end{tabular}
''' + caption + '\n\\end{table}\n'

    with open(filename, 'w') as fd:
        fd.write(tex)

def process_csv(infile):
    """Load CSV file and handle errors gracefully."""
    try:
        data = np.genfromtxt(infile, delimiter=',', skip_header=1)
        if data.ndim != 2 or data.shape[1] != 4:
            print(f"Error: Expected 4 columns in file '{infile}', got {data.shape[1] if data.ndim == 2 else 'unknown'}")
            return None
        return data
    except Exception as e:
        print(f"Error reading '{infile}': {e}")
        return None

def process_files(infile, outfile):
    """Process input and output paths."""
    if os.path.isfile(infile) and os.path.isfile(outfile):
        data = process_csv(infile)
        if data is not None:
            gen_table(data, outfile)
            print(f"{infile} -> {outfile}")
        else:
            print(f"Error: Could not generate table for '{infile}'.")
    elif os.path.isdir(infile) and os.path.isdir(outfile):
        for f in os.listdir(infile):
            if f.endswith(".csv"):
                input_path = os.path.join(infile, f)
                output_filename = f.replace(".csv", ".tex")
                output_path = os.path.join(outfile, output_filename)

                data = process_csv(input_path)
                if data is not None:
                    gen_table(data, output_path)
                    print(f"{input_path} -> {output_path}")
                else:
                    print(f"Skipping file '{input_path}' due to errors.")
    else:
        print("Error: Both input and output paths must be either files or directories.")
        sys.exit(1)

def main():
    """Main entry point of the script."""
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file_or_dir> <output_file_or_dir>")
        sys.exit(1)

    infile = sys.argv[1]
    outfile = sys.argv[2]

    if not (os.path.isfile(infile) and os.path.isfile(outfile)) and not (os.path.isdir(infile) and os.path.isdir(outfile)):
        print("Error: Either both inputs are files or both are directories.")
        sys.exit(1)

    process_files(infile, outfile)

if __name__ == "__main__":
    main()

