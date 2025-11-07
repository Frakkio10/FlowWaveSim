#%%
import numpy as np
import h5py
import matplotlib.pyplot as plt
from src.utils import save_data, initialize_field, field_from_nk, U_y_profile, plot_spectrum, dispersion_rel, animation
from config.definitions import HD5_DIR
import os
#%%
nx, ny     = 512, 512                  # grid resolution
dx         = 4e-4 # m
dt         = 1e-4 #s 
n_steps    = 200
save_every = 1

# Physics options
mode = "y_velocity + phase + shear"    # or "y_velocity", "y_velocity + phase", "y_velocity + phase + shear", 
include_phase = "phase" in mode
include_shear = "shear" in mode

U0 = 10 # 4.0        # base poloidal velocity m/s
S  = -50 #0.5 # 0.5 * 1e-2          # shear rate (for U_y(x) = U0 + S*(x - Lx/2))

ky_c   = 0.8e3    # m
omega0 = 12       # s-1
cs     = 5        # phase velocity magnitude m/s (for linear dispersion relation)

# Initial condition
init_type = "tilt"     # "packet" , "plane" or 'tilt' 
direction = "horizontal" # direction of propagation
disp_type = 'linear'      # 'quadratic

# ======================================================
#                   NUMERICAL SETUP
# ======================================================

x = np.linspace(0, nx * dx, nx, endpoint=False)
y = np.linspace(0, ny * dx, ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing="xy")

kx = 2 * np.pi * np.fft.fftfreq(nx, d = dx)   # range: -1/(2dx) ... +1/(2dx) (in m-1)
ky = 2 * np.pi * np.fft.fftfreq(ny, d = dx)
KX, KY = np.meshgrid(kx, ky)


# ======================================================
#                   TIME VECTOR
# ======================================================

t = np.linspace(0, dt * 200, 200)
#%%
# ======================================================
#                   INITIAL FIELD
# ======================================================
 
n_kx_ky   = initialize_field(init_type, nx, ny, dx, KX, KY, beta = 45)
n_xy      = field_from_nk(n_kx_ky)


# ======================================================
#               FIELD, SPECTRUM PLOTs
# ======================================================

# plt.pcolormesh(x, y, n_xy, cmap = 'seismic')
# plot_spectrum(n_xy, kx, ky)
#%%
n_xky0 = np.fft.fft(n_xy.T, axis=1)   # shape (nx, nky)
nx_check, nky = n_xky0.shape
assert nx_check == nx

# ======================================================
#                  VELOCITY SETUP
# ======================================================
omega_ky = dispersion_rel(disp_type, ky)
U_y      = U_y_profile(x, nx * dx, U0, S, include_shear)
# U(x, ky)
U_xky = np.einsum('i,j->ij', U_y, np.ones_like(ky)) #shape (Nx, Nky)

#%%
# ======================================================
#              TIME EVOLUTION
# ======================================================

def propagate(n_xky0, U_xky, omega, t):
    # ky°U(x, ky) + omega(ky)
    exponent = (np.einsum('j,ij->ij', ky, U_xky) + omega[None, :])  #shape (Nx, Nky)

    # exp{-j (ky°U(x, ky) + omega(ky))}
    phase_exponent = np.einsum('t, nm -> tnm', t, exponent) #shape (t, Nx, Nky)
    phase = np.exp(-1j * phase_exponent)

    n_xky = n_xky0[None, :, :] * phase
    return n_xky

n_xkyt = propagate(n_xky0, U_xky, omega_ky, t)


#%%
# ======================================================
#              SAVING
# ======================================================


output_file = HD5_DIR.joinpath('final_test/advection_linear_shear.h5')

if not output_file.parent.exists():
    os.makedirs(output_file.parent)
    
with h5py.File(output_file, "w",  libver = "latest") as f:
    save_data(f, "grid", False, x = x, y = y, kx = kx, ky = ky)
    save_data(f, "parameters", False, nx = nx, ny = ny, dx = dx)
    f.attrs.update({
        "mode": mode, "U0": U0, "S": S, "c": cs, "dt": dt,
        "init_type": init_type, "direction": direction, 'dispersion': disp_type
    })
    save_data(f, "parameters", False, nx = nx, ny = ny, dx = dx)
    f.attrs.update({
        "mode": mode, "U0": U0, "S": S, "c": cs, "dt": dt,
        "init_type": init_type, "direction": direction, 'dispersion': disp_type
    })
    save_data(f, "fields", True, n = n_xkyt)   


#%%
# ======================================================
#              ANIMATION
# ======================================================

animation(n_xkyt, x, y, output_file)
# %%
