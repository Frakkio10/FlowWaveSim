#%%

import numpy as np 

# ======================================================
#                   SAVE FILE
# ======================================================

def save_data(fl, grpname, ext_flag, **kwargs):
    """Appends or creates datasets in an HDF5 group."""
    if grpname not in fl:
        grp = fl.create_group(grpname)
    else:
        grp = fl[grpname]
    for key, val in kwargs.items():
        if key not in grp:
            if not ext_flag:
                grp[key] = val
            else:
                if np.isscalar(val):
                    grp.create_dataset(key, (1,), maxshape=(None,), dtype=type(val))
                    if not fl.swmr_mode:
                        fl.swmr_mode = True
                else:
                    grp.create_dataset(
                        key, (1,) + val.shape,
                        chunks=(1,) + val.shape,
                        maxshape=(None,) + val.shape,
                        dtype=val.dtype
                    )
                    if not fl.swmr_mode:
                        fl.swmr_mode = True
                dset = grp[key]
                dset[-1,] = val
                dset.flush()
        elif ext_flag:
            dset = grp[key]
            dset.resize((dset.shape[0]+1,) + dset.shape[1:])
            dset[-1,] = val
            dset.flush()
        fl.flush()

# ======================================================
#                   INITIAL CONDITION
# ======================================================

def hermitianize(nk):
    """Ensure real-valued inverse FFT field."""
    Ny_, Nx_ = nk.shape
    nk_sym = np.zeros_like(nk, dtype=np.complex128)
    for iy in range(Ny_):
        for ix in range(Nx_):
            nk_sym[iy, ix] = 0.5*(nk[iy, ix] + np.conj(nk[-iy % Ny_, -ix % Nx_]))
    return nk_sym

def initialize_field(init_type, Nx, Ny, Lx, Ly, KX, KY, direction = 'horizontal'):
    nk = np.zeros((Ny, Nx), dtype=np.complex128)
    if init_type == "plane":
        m = 4 if direction == "horizontal" else 0
        n = 4 if direction == "vertical" else 0
        kx0, ky0 = 2*np.pi*m/Lx, 2*np.pi*n/Ly
        ix = np.argmin(np.abs(kx - kx0))
        iy = np.argmin(np.abs(ky - ky0))
        nk[iy, ix] = 1.0
    elif init_type == "packet":
        sigma_k = 2.0
        kx_c = 2*np.pi*4/Lx if direction == "horizontal" else 0.0
        ky_c = 2*np.pi*4/Ly if direction == "vertical" else 0.0
        _nk = np.exp(-0.5*((KX - kx_c)**2 + (KY - ky_c)**2) / sigma_k**2)
        nk = _nk * np.exp(1j * np.random.uniform(0, 2*np.pi, nk.shape))
    else:
        raise ValueError("Invalid init_type")
    return hermitianize(nk)

# ======================================================
#                       UTILITY
# ======================================================
def rms(variable):
    return np.sqrt(np.mean(variable * np.conjugate(variable)) )


def field_from_nk(nk): 
    return np.real(np.fft.ifft2(nk))


# ======================================================
#    PHYSICS: Dispersion relation -> to implement
# ======================================================

def _dispersion_rel(disp_type, c, K_abs):
    if disp_type == 'linear':
        return c * K_abs
    elif disp_type == 'quadratic':
        return c * K_abs ** 2 - 2 *c * K_abs ** 2


# ======================================================
#            PHYSICS: FLOW PROFILES
# ======================================================
def U_y_profile(x, Lx, U0, S, mode):
    """Radial profile of poloidal flow."""
    if "shear" in mode:
        U0 += S * (x - Lx/2)
    else:
        U0 *= np.ones_like(x)
    return U0


def Omega_profile(x, Lx, Omega0, differential_rotation):
    """Radial profile of angular velocity (for differential rotation)."""
    if differential_rotation:
        return Omega0 * (1 + 0.5*np.sin(2*np.pi*(x - Lx/2)/Lx)) 
    return Omega0 * np.ones_like(x)
# %%
