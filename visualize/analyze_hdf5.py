#%%
import h5py, matplotlib.pyplot as plt

filename = '/Users/FO278650/Desktop/FlowWaveSim/output/h5/prova3.h5'

"""
analysis_h5.py
--------------
Inspect and visualize 2D wave simulation results stored in HDF5.
 - Lists all groups, datasets, and attributes
 - Allows quick visualization of n(x,y,t) fields
"""

import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1️⃣ OPEN FILE
# ==========================================
if len(sys.argv) < 2:
    print("Usage: python analysis_h5.py <file.h5>")
    sys.exit(1)

# filename = sys.argv[1]

with h5py.File(filename, "r") as f:
    print(f"File: {filename}")
    print(f"{'-'*40}\n")

    # --------------------------------------
    # List attributes (simulation parameters)
    # --------------------------------------
    print("Attributes:")
    for key, val in f.attrs.items():
        print(f"  {key}: {val}")
    print()

    # --------------------------------------
    # List all groups and datasets
    # --------------------------------------
    print('Structure;')
    def print_h5_structure(name, obj):
        print(f" {name}")
    f.visititems(print_h5_structure)

    print("\n" + "-"*40)

    # --------------------------------------
    # Load data arrays
    # --------------------------------------
    x = f["grid/x"][:]
    y = f["grid/y"][:]
    n_data = f["evolution/n"][:]  # shape = (Nt, Ny, Nx)
    Nt, Ny, Nx = n_data.shape
    print(f"Data shape: n(x,y,t) = {n_data.shape}")
    print(f"Grid: Nx={Nx}, Ny={Ny}, Nt={Nt}")
    print(f"x ∈ [{x.min():.2f}, {x.max():.2f}], y ∈ [{y.min():.2f}, {y.max():.2f}]\n")


#%%
# ==========================================
# 2️⃣ INTERACTIVE FRAME PLOTTING
# ==========================================
def plot_frame(frame):
    """Plot a single frame of n(x,y,t)."""
    plt.figure(figsize=(6,5))
    plt.title(f"Frame {frame}/{Nt-1}")
    plt.xlabel("x (radial)")
    plt.ylabel("y (poloidal)")
    vmax = np.max(np.abs(n_data))
    plt.pcolormesh(x, y, n_data[frame], shading="auto",
                   cmap="seismic", vmin=-vmax, vmax=vmax)
    plt.colorbar(label="n(x,y)")
    plt.tight_layout()
    plt.show()

# ==========================================
# 3️⃣ CHOOSE WHICH FRAMES TO SHOW
# ==========================================
while True:
    try:
        cmd = input(f"\nEnter frame index [0–{Nt-1}] (or 'q' to quit): ")
        if cmd.lower() in ("q", "quit", "exit"):
            break
        frame = int(cmd)
        if 0 <= frame < Nt:
            plot_frame(frame)
        else:
            print("⚠️ Invalid frame index.")
    except ValueError:
        print("⚠️ Please enter an integer index.")

# %%
