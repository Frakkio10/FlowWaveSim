"""
2D wave / turbulence propagation simulator
-----------------------------------------
Models propagation in a rectangular box with:
 - Poloidal advection (U_y)
 - Optional phase velocity (ω = c|k|)
 - Radial shear (U_y(x))                                    --> to implement / improve
 - Optional solid-body or differential rotation (Ω or Ω(x)) --> to implement / improve
 - HDF5 output of all frames using save_data() from Ozgur

"""
#%%
import numpy as np
import h5py
import matplotlib.pyplot as plt
from src.utils import save_data, initialize_field, field_from_nk, U_y_profile, Omega_profile
from config.definitions import HD5_DIR

#%%
# ======================================================
#                    USER CONFIGURATION
# ======================================================
Nx, Ny = 1024, 1024                  # grid resolution
Lx, Ly = 10, 10                      # box dimensions
dt = 0.02
n_steps = 500
save_every = 1

# Physics options
mode = "y_velocity + phase + shear"    # or "y_velocity", "y_velocity + phase", "y_velocity + phase + shear", 
include_phase = "phase" in mode

U0 = 5.0        # base poloidal velocity
S = 0.3           # shear rate (for U_y(x) = U0 + S*(x - Lx/2))
c = 1.0         # phase velocity magnitude

# Rotation options
rotation = False
Omega0 = 0.5
differential_rotation = False
center = (Lx/2, Ly/2)

# Initial condition
init_type = "packet"     # "packet" or "plane"
direction = "horizontal" # direction of propagation

# Output file
output_file = HD5_DIR.joinpath('prova3.h5')

# prevent diffusion 
anti_diffusion = True

# ======================================================
#                   NUMERICAL SETUP
# ======================================================
x = np.linspace(0, Lx, Nx, endpoint=False)
y = np.linspace(0, Ly, Ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing="xy")

kx = 2 * np.pi * np.fft.fftfreq(Nx, d = Lx / Nx)
ky = 2 * np.pi * np.fft.fftfreq(Ny, d = Ly / Ny)
KX, KY = np.meshgrid(kx, ky, indexing="xy")
K_abs = np.sqrt(KX ** 2 + KY ** 2)

# ======================================================
#                   Phase velocity
# ======================================================
omega_k = c * K_abs   




# ======================================================
#           EVOLUTION OPERATOR (stable version)
# ======================================================


def evolve_one_step(nk, t, dt):
    """
    Hybrid split-step evolution:
    - Half-step phase in Fourier space
    - Full-step advection (via integer grid roll)
    - Half-step phase again
    """
    # Half-step phase evolution (Fourier space)
    if include_phase:
        nk *= np.exp(-1j * omega_k * dt / 2)

    # Go to real space
    n_xy = np.fft.ifft2(nk).real

    # Flow velocity field
    Uy = U_y_profile(x, Lx, U0, S, mode)[None, :].repeat(Ny, axis=0)
    Ux = np.zeros_like(Uy)

    if rotation:
        xc, yc = center
        Omega_x = Omega_profile(x, Lx, Omega0, differential_rotation)[None, :]
        Ux -= Omega_x * (Y - yc)
        Uy += Omega_x * (X - xc)

    # Integer grid shifts for advection
    dx = Lx / Nx
    dy = Ly / Ny
    shift_y = np.rint(Uy * dt / dy).astype(int)
    shift_x = np.rint(Ux * dt / dx).astype(int)

    # Apply shifts
    for i in range(Nx):
        n_xy[:, i] = np.roll(n_xy[:, i], shift_y[0, i])
    for j in range(Ny):
        n_xy[j, :] = np.roll(n_xy[j, :], shift_x[j, 0])

    # Transform back to Fourier space
    nk = np.fft.fft2(n_xy)

    # Second half-step phase
    if include_phase:
        nk *= np.exp(-1j * omega_k * dt / 2)

    return nk



# ======================================================
#                    MAIN SIMULATION
# ======================================================
nk   = initialize_field(init_type, Nx, Ny, Lx, Ly, KX, KY)
n_xy = field_from_nk(nk)

with h5py.File(output_file, "w",  libver = "latest") as f:
    save_data(f, "grid", False, x = x, y = y)
    f.attrs.update({
        "mode": mode, "U0": U0, "S": S, "c": c, "dt": dt,
        "rotation": rotation, "Omega0": Omega0,
        "differential_rotation": differential_rotation,
        "init_type": init_type, "direction": direction
    })

    print(f"\n Starting simulation ({mode})...")
    for step in range(n_steps):
        nk = evolve_one_step(nk, step * dt, dt)
        if step % save_every == 0:
            n_xy = field_from_nk(nk)
            
            # --- Optional anti-diffusion correction ---
            if anti_diffusion:
                # Compute total "energy" 
                energy_now = np.sum(np.abs(n_xy)**2)
                if step == 0:
                    energy_ref = energy_now  # store initial energy
                else:
                    # Compute renormalization factor
                    factor = np.sqrt(energy_ref / energy_now)
                    n_xy *= factor
                    # Recompute Fourier transform for next step
                    nk = np.fft.fft2(n_xy)
        
        
            save_data(f, "fields", True, n = n_xy)   
            if step % 10 == 0:
                print(f"Step {step}/{n_steps}: max|n| = {np.abs(n_xy).max():.3e}")

print(f"\n Simulation complete. Results saved to {output_file}\n")

# %%
