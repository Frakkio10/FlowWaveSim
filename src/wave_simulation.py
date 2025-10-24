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
import os
#%%
# ======================================================
#                    USER CONFIGURATION
# ======================================================
nx, ny     = 512, 512                  # grid resolution
dx         = 4e-4 # m
dt         = 1e-4 #s 
n_steps    = 200
save_every = 1

# Physics options
mode = "y_velocity + phase "    # or "y_velocity", "y_velocity + phase", "y_velocity + phase + shear", 
include_phase = "phase" in mode

U0 = 4.0 # 4.0        # base poloidal velocity m/s
S  = 0 # 0.5 * 1e-2          # shear rate (for U_y(x) = U0 + S*(x - Lx/2))
c  = 1e-1        # phase velocity magnitude

# Rotation options
rotation = False
Omega0 = 0.5
differential_rotation = False
center = ((nx * dx)/2, (ny * dx)/2)

# Initial condition
init_type = "tilt"     # "packet" , "plane" or 'tilt' 
direction = "horizontal" # direction of propagation
disp_type = 'complex'      # 'quadratic




#%%
# Output file
output_file = HD5_DIR.joinpath('tilt/advection_linear.h5')

if not output_file.parent.exists():
    os.makedirs(output_file.parent)

# prevent diffusion 
anti_diffusion = True

# ======================================================
#                   NUMERICAL SETUP
# ======================================================

x = np.linspace(0, nx * dx, nx, endpoint=False)
y = np.linspace(0, nx * dx, ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing="xy")

kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)   # range: -1/(2dx) ... +1/(2dx) (in cycles/unit)
ky = 2 * np.pi * np.fft.fftfreq(ny, d=dx)
kx = np.fft.fftshift(kx)   # now -kNyq..+kNyq in order
ky = np.fft.fftshift(ky)
KX, KY = np.meshgrid(kx, ky)

# ======================================================
#                   Phase velocity
# ======================================================

# omega_k = c * KY  + beta_x * KX ** 2 + beta_y * KY ** 2
# omega_k = np.where(KY>=0, c_pos * KY, c_neg * KY)
omega_k = c * KY
# omega_k = c *1e-6 * ky ** 2
#%%
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
        nk *= np.exp(1j * omega_k * dt / 2)

    # Go to real space
    n_xy = np.fft.ifft2(nk).real

    # Flow velocity field
    Uy = U_y_profile(x, nx * dx, U0, S, mode)[None, :].repeat(ny, axis=0)
    Ux = np.zeros_like(Uy)

    if rotation:
        xc, yc = center
        Omega_x = Omega_profile(x, nx * dx, Omega0, differential_rotation)[None, :]
        Ux -= Omega_x * (Y - yc)
        Uy += Omega_x * (X - xc)

    # Integer grid shifts for advection
    shift_y = np.rint(Uy * dt / dx).astype(int)
    shift_x = np.rint(Ux * dt / dx).astype(int)

    # Apply shifts
    for i in range(nx):
        n_xy[:, i] = np.roll(n_xy[:, i], shift_y[0, i])
    for j in range(ny):
        n_xy[j, :] = np.roll(n_xy[j, :], shift_x[j, 0])

    # Transform back to Fourier space
    nk = np.fft.fft2(n_xy)

    # Second half-step phase
    if include_phase:
        nk *= np.exp(1j * omega_k * dt / 2)

    return nk



# ======================================================
#                    MAIN SIMULATION
# ======================================================
nk   = initialize_field(init_type, nx, ny, dx, KX, KY, beta = 45)
n_xy = field_from_nk(nk)

im = plt.pcolormesh(x, y, n_xy, cmap = 'seismic')
plt.colorbar(im)

#%%
with h5py.File(output_file, "w",  libver = "latest") as f:
    save_data(f, "grid", False, x = x, y = y, kx = kx, ky = ky)
    save_data(f, "parameters", False, nx = nx, ny = ny, dx = dx)
    f.attrs.update({
        "mode": mode, "U0": U0, "S": S, "c": c, "dt": dt,
        "rotation": rotation, "Omega0": Omega0,
        "differential_rotation": differential_rotation,
        "init_type": init_type, "direction": direction, 'dispersion': disp_type
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
im = plt.pcolormesh(x, y, n_xy, cmap = 'seismic')
plt.colorbar(im)


# %%
