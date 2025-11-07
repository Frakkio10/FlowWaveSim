#%%
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from config import HD5_DIR, MP4_DIR, GIF_DIR
import numpy as np 
import os 

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
        
def animation(n_xkyt, x, y, filename):
    
    Nt = n_xkyt.shape[0]
    n_xy0 = np.fft.ifft(n_xkyt[0,:,:], axis=1).real.T
    fig, ax = plt.subplots(figsize=(6,5))
    vmax = np.max(np.abs(n_xy0))
    img = ax.pcolormesh(x*1e2, y*1e2, n_xy0, shading="auto", cmap="seismic", vmin=-vmax, vmax=vmax)
    plt.colorbar(img, ax=ax, label="n(x,y,t)")
    ax.set_title(f"2D Field Evolution")
    ax.set_xlabel("x [cm]")
    ax.set_ylabel("y [cm]")

    text_t = ax.text(0.02, 0.95, "", transform=ax.transAxes,
                    color="black", fontsize=10, va="top", ha="left",
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

    def update(frame):
        n = np.fft.ifft(n_xkyt[frame,:,:], axis=1).real.T   # inverse FFT over k_y -> y

        img.set_array(n.ravel())
        text_t.set_text(f"Frame {frame+1}/{Nt}")
        return img, text_t

    anim = FuncAnimation(fig, update, frames=Nt, interval=40, blit=False)

    out_gif = MP4_DIR.joinpath(str(filename.parent.name) + '/').joinpath(filename.name).with_suffix('.gif')
    out_gif = GIF_DIR.joinpath(str(filename.parent.name) + '/').joinpath(filename.name).with_suffix('.gif')

    def ffmpeg_available():
        """Check if ffmpeg is installed and in PATH."""
        return shutil.which("ffmpeg") is not None

    try:
        if ffmpeg_available():
            print(f"\n Using ffmpeg to render MP4: {out_mp4}")
            writer = FFMpegWriter(fps=50, metadata=dict(artist="Tokamak Edge Simulation"))
            
            if not out_mp4.parent.exists():
                os.makedirs(out_mp4.parent)
        
            anim.save(out_mp4, writer=writer, dpi=150)
            print(f"MP4 saved to {out_mp4}\n")
        else:
            raise FileNotFoundError("ffmpeg not found")
    except Exception as e:
        print(f"\n Could not use ffmpeg ({e}). Switching to Pillow (GIF).")
        writer = PillowWriter(fps=15)
        if not out_gif.parent.exists():
                os.makedirs(out_gif.parent)
        
        anim.save(out_gif, writer=writer)
        print(f"GIF saved to {out_gif}\n")

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

def _initialize_field(init_type, Nx, Ny, Lx, Ly, KX, KY, direction = 'horizontal'):
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

def initialize_field(init_type, nx, ny, dx, KX, KY, **kwargs):
    spec_centered = np.zeros((ny, nx), dtype=np.complex128)
    phi           = 2 * np.pi * np.random.random((ny, nx)) - np.pi

    if init_type == "plane":
        m = 4 if kwargs.get('direction', 'horizontal') == "horizontal" else 0
        n = 4 if kwargs.get('direction', 'horizontal') == "vertical" else 0
        kx0, ky0 = 2 * np.pi * m / (nx * dx), 2 * np.pi * n / (ny * dx)
        ix = np.argmin(np.abs(KX[0, :] - kx0))
        iy = np.argmin(np.abs(KY[:, 0] - ky0))
        spec_centered[iy, ix] = 1.0
        
    elif init_type == "packet":
        sigma_k = nx * dx * 5e2
        kx_c = 2 * np.pi *4 / (nx * dx) if kwargs.get('direction', 'horizontal') == "horizontal" else 0.0
        ky_c = 2 * np.pi *4 / (ny * dx) if kwargs.get('direction', 'horizontal') == "vertical" else 0.0
        norm = 1 / (2 * np.pi * sigma_k ** 2)
        spec_centered = np.exp(-0.5*((KX - kx_c)**2 + (KY - ky_c)**2) / sigma_k**2)
        
    elif init_type == 'tilt':
        lmin, lmax = kwargs.get('lmin', 0.5 *1e-2), kwargs.get('lmax', 1.4 *1e-2)
        beta = np.pi / 2 - kwargs.get('beta', 20) * np.pi / 180.0

        A = (KX * np.cos(beta) - KY * np.sin(beta)) ** 2 * lmin ** 2
        B = (KX * np.sin(beta) + KY * np.cos(beta)) ** 2 * lmax ** 2
        norm = lmin * lmax / (8.0 * np.pi)
        spec_centered = norm * np.exp(-(A + B) / 8.0) 
    
    else:
        raise ValueError("Invalid init_type")
    
    nk = hermitianize(spec_centered * np.exp(1j * phi))
    # spec_for_ifft = np.fft.ifftshift(nk)
    # Z             = np.fft.ifft2(spec_for_ifft).real
    # delta_ne = Z.copy()
    # delta_ne /= rms(delta_ne) 
    return nk



# ======================================================
#                       UTILITY
# ======================================================

def rms(v):
    return np.sqrt(np.mean(np.abs(v)**2))

def field_from_nk(nk): 
    # spec_for_ifft = np.fft.ifftshift(nk)
    Z = np.fft.ifft2(nk).real 
    delta_ne = Z.copy()
    delta_ne /= rms(delta_ne) 
    return delta_ne


# ======================================================
#    PHYSICS: Dispersion relation -> to implement
# ======================================================

def dispersion_rel(disp_type, ky, **kwargs):
    if disp_type == 'linear':
        cs = kwargs.get('cs', 5)
        return cs * ky
    elif disp_type == 'ITG':
        omega0 = kwargs.get('omega0', 12)
        ky_c   = kwargs.get('ky_c', 0.8e3)
        return omega0 * ky / (1 + (ky / ky_c)**2)
    elif disp_type == 'TEM':
        omega0 = kwargs.get('omega0', -3)
        ky_c   = kwargs.get('ky_c', 1.2e3)
        return omega0 * ky / (1 + (ky / ky_c)**2)
    
# ======================================================
#            PHYSICS: FLOW PROFILES
# ======================================================
def U_y_profile(x, Lx, U0, S, include_shear):
    """Radial profile of poloidal flow."""
    if include_shear:
        #U0 += S * x #(x - Lx / 2)
        # U0 += S * np.tanh(4 * (x - Lx/2) / Lx)
        U_y = U0 + S * (x)
    else:
        U_y = U0 * np.ones_like(x) 
    return U_y


# ======================================================
#            ISOTROPIC SPECTRUM
# ======================================================

def isotropic_spectrum(F, kx, ky, nbins=100):
    """
    Compute isotropic (angle-averaged) 1D spectrum from 2D Fourier map F(kx, ky).
    Returns: k_center, S(k)
    """
    # Create k-magnitude grid
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    kr = np.sqrt(KX**2 + KY**2)
    
    # Define bins
    k_bins = np.linspace(0, kr.max(), nbins + 1)
    k_center = 0.5 * (k_bins[1:] + k_bins[:-1])
    
    # Flatten arrays
    kr_flat = kr.ravel()
    F_flat = np.abs(F.ravel())**2  # power spectrum density
    
    # Bin by k magnitude
    S = np.zeros(nbins)
    for i in range(nbins):
        mask = (kr_flat >= k_bins[i]) & (kr_flat < k_bins[i+1])
        if np.any(mask):
            S[i] = np.mean(F_flat[mask])
    
    return k_center, S

# ======================================================
#            PLOT SPECTRUM
# ======================================================

def plot_spectrum(delta_ne, kx, ky, ax = None, fig = None):
    nx, ny = delta_ne.shape
    F = np.fft.fftshift(np.fft.fft2(delta_ne))  # shift zero freq to center
    PSD2 = np.abs(F) #(np.abs(F)**2)  / (nx * ny)  # normalization; adjust if you prefer power spectral density per unit k
    KX, KY = np.meshgrid(kx, ky)
    _kx, _ky = np.fft.fftshift(kx), np.fft.fftshift(ky)

    from scipy.stats import binned_statistic

    # Sx(kx): average PSD over ky (axis 0)
    Sx = PSD2.mean(axis=0)   # length Nx, corresponds to kx array
    Sy = PSD2.mean(axis=1)   # length Ny, corresponds to ky array

    if ax is None:
        fig, ax = plt.subplots(1, 4, figsize=(12,4))

    im = ax[0].pcolormesh(_kx * 1e-2, _ky *1e-2, np.log10(PSD2 + 1e-20), cmap = 'seismic')
    ax[0].set_xlabel(r'$k_x$ [$cm^{-1}$]')
    ax[0].set_ylabel(r'$k_y$ [$cm^{-1}$]')
    ax[0].set_xlim(-75, 75)
    ax[0].set_ylim(-75, 75)
    fig.colorbar(im, ax=ax[0])

    ax[1].plot(_kx * 1e-2, Sx, c = 'r', label = r'$S_k^x$')
    ax[1].plot(_ky * 1e-2, Sy, c = 'b', label = r'$S_k^y$')
    ax[1].set_xlabel(r'$k$ [$cm^{-1}$]')
    ax[1].set_ylabel(r'PSD [a.u.]')
    ax[1].set_xlim(-50, 50)
    ax[1].legend()
    
    ax[2].loglog(_kx * 1e-2, Sx ,c = 'r',  label = r'$S_k^x$')
    ax[2].loglog(_ky * 1e-2, Sy ,c = 'b',  label = r'$S_k^y$')
    ax[2].set_xlim(1e-1, 100)
    ax[2].legend()
    ax[2].set_xlabel(r"$k$ [cm$^{-1}$]")
    ax[2].set_ylabel(r'1D directional spectrum [a.u.]')
    
    k, S = isotropic_spectrum(F**0.5, _kx, _ky)
    E = 2 * np.pi * k * S    
    ax[3].loglog(k * 1e-2, S / S.max(),c = 'r', label="S(k)")
    ax[3].loglog(k * 1e-2, E / E.max(),c = 'b', label="E(k)")
    ax[3].set_xlabel(r"$k$ [cm$^{-1}$]")
    ax[3].set_ylabel("Isotropic spectrum [a.u.]")
    ax[3].legend()
    ax[3].set_xlim(1, 20)
    ax[3].set_ylim(1e-2, 2)
    plt.tight_layout()
    plt.show()
# %%
if __name__ == '__main__':
    nx, ny, dx = 512, 512, 4e-4
    x = np.linspace(0, nx * dx, nx, endpoint=False)
    y = np.linspace(0, ny * dx, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="xy")

    kx = 2 * np.pi * np.fft.fftfreq(nx, d = dx)   # range: -1/(2dx) ... +1/(2dx) (in m-1)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d = dx)
    # kx = np.fft.fftshift(kx)   # now -kNyq..+kNyq in order
    # ky = np.fft.fftshift(ky)
    KX, KY = np.meshgrid(kx, ky)

    n_kx_ky   = initialize_field('tilt', nx, ny, dx, KX, KY, beta = 45, lmin = 2e-3, lmax = 4e-3)    
    n_xy = field_from_nk(n_kx_ky)

    plt.pcolormesh(x, y, n_xy, cmap = 'seismic')
    fig, ax = plt.subplots(1, 4, figsize = (10, 3))

    plot_spectrum(n_xy, kx, ky, ax = ax, fig = fig)
# %%
