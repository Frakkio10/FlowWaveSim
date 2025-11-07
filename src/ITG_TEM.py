#%%
#%%
import numpy as np
import h5py
import matplotlib.pyplot as plt
from src.utils import save_data, initialize_field, field_from_nk, U_y_profile, plot_spectrum, dispersion_rel, animation, isotropic_spectrum
from config.definitions import HD5_DIR
import os

# %%
nx, ny     = 512, 512                  # grid resolution
dx         = 4e-4 # m
dt         = 1e-4 #s 
n_steps    = 200
save_every = 1

# Physics options
mode = "y_velocity + phase "    # or "y_velocity", "y_velocity + phase", "y_velocity + phase + shear", 
include_phase = "phase" in mode
include_shear = "shear" in mode

U0 = 10 # 4.0        # base poloidal velocity m/s
S  = 0 #0.5 # 0.5 * 1e-2          # shear rate (for U_y(x) = U0 + S*(x - Lx/2))

# Initial condition
init_type = "tilt"     # "packet" , "plane" or 'tilt' 
direction = "horizontal" # direction of propagation

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
# %%
# ======================================================
#               parameters for ITG TEM
# ======================================================
# ITG (forward)
params_itg = dict(lmin=0.5e-2, lmax=1e-2, beta_deg=20.0,
                  kc=800.0, omega0=+12.0, amp=1.0)

# TEM (backward, opposite sign)
params_tem = dict(lmin=6.0e-3, lmax=1.5e-2, beta_deg=-10.0,
                  kc=1200.0, omega0=-3, amp=1.2)

# build envelopes (these are *power* envelopes)
S_itg   = initialize_field(init_type, nx, ny, dx, KX, KY, 
                         lmin = params_itg['lmin'], lmax = params_itg['lmax'], 
                         beta = params_itg['beta_deg'])
n_xy_itg = field_from_nk(S_itg)

S_tem    = initialize_field(init_type, nx, ny, dx, KX, KY, 
                         lmin = params_tem['lmin'], lmax = params_tem['lmax'], 
                         beta = params_tem['beta_deg'])
k0_tem = 1000.0  # m^-1 
S_tem *= (1.0 + 0.4 * np.exp(-((KY - k0_tem)**2)/(2*(200.0**2))))  # bump
n_xy_tem = field_from_nk(S_tem)


# %%
# ======================================================
#                  VELOCITY SETUP
# ======================================================
U_y      = U_y_profile(x, nx * dx, U0, S, include_shear)
# U(x, ky)
U_xky = np.einsum('i,j->ij', U_y, np.ones_like(ky)) #shape (Nx, Nky)


omega_itg = dispersion_rel('ITG', ky, omega0 = params_itg['omega0'], kc = params_itg['kc'])
omega_tem = dispersion_rel('TEM', ky, omega0 = params_tem['omega0'], kc = params_tem['kc'])

# %%
def propagate(n_xky0, U_xky, omega, t):
    # ky°U(x, ky) + omega(ky)
    exponent = (np.einsum('j,ij->ij', ky, U_xky) + omega[None, :])  #shape (Nx, Nky)

    # exp{-j (ky°U(x, ky) + omega(ky))}
    phase_exponent = np.einsum('t, nm -> tnm', t, exponent) #shape (t, Nx, Nky)
    phase = np.exp(-1j * phase_exponent)

    n_xky = n_xky0[None, :, :] * phase
    return n_xky

#%%
# ITG propagation 
n_xky_itg = np.fft.fft(n_xy_itg.T, axis=1)   # shape (nx, nky)
n_xkyt_itg = propagate(n_xky_itg, U_xky, omega_itg, t)

# TEM propagation 
n_xky_tem  = np.fft.fft(n_xy_tem.T, axis=1)   # shape (nx, nky)
n_xkyt_tem = propagate(n_xky_tem, U_xky, omega_tem, t)

#%%
output_file = HD5_DIR.joinpath('final_test/ITG.h5')
animation(n_xkyt_itg, x, y, output_file )
# %%
output_file = HD5_DIR.joinpath('final_test/TEM.h5')
animation(n_xkyt_tem, x, y, output_file )
# %%
output_file = HD5_DIR.joinpath('final_test/ITG_TEM.h5')
animation(n_xkyt_itg + n_xkyt_tem,  x, y, output_file )

# %%
