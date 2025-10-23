"""
2D wave / turbulence propagation simulator
-----------------------------------------
Models propagation in a rectangular box with:
 - Poloidal advection (U_y)
 - Optional phase velocity (ω = c|k|)
 - Radial shear (U_y(x))
 - Optional solid-body or differential rotation (Ω or Ω(x))
 - HDF5 output of all frames using save_data() from Ozgur
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from src.utils import save_data, initialize_field, field_from_nk, U_y_profile, Omega_profile
from config.definitions import HD5_DIR, FIG_DIR
import os
import argparse


def evolve_one_step(nk, t, dt, include_phase, omega_k, x, X, Y,
                    U0, S, rotation, Omega0, differential_rotation, center, mode, nx, ny, dx):
    """Hybrid split-step evolution"""
    if include_phase:
        nk *= np.exp(-1j * omega_k * dt / 2)

    n_xy = np.fft.ifft2(nk).real

    Uy = U_y_profile(x, nx * dx, U0, S, mode)[None, :].repeat(ny, axis=0)
    Ux = np.zeros_like(Uy)

    if rotation:
        xc, yc = center
        Omega_x = Omega_profile(x, nx * dx, Omega0, differential_rotation)[None, :]
        Ux -= Omega_x * (Y - yc)
        Uy += Omega_x * (X - xc)

    shift_y = np.rint(Uy * dt / dx).astype(int)
    shift_x = np.rint(Ux * dt / dx).astype(int)

    for i in range(nx):
        n_xy[:, i] = np.roll(n_xy[:, i], shift_y[0, i])
    for j in range(ny):
        n_xy[j, :] = np.roll(n_xy[j, :], shift_x[j, 0])

    nk = np.fft.fft2(n_xy)

    if include_phase:
        nk *= np.exp(-1j * omega_k * dt / 2)

    return nk


def main(args):
    # ---------------- USER CONFIGURATION ----------------
    nx, ny = args.nx, args.ny
    dx = args.dx
    dt = args.dt
    n_steps = args.n_steps
    save_every = args.save_every

    mode = args.mode
    include_phase = "phase" in mode

    U0 = args.U0
    S = args.S
    c = args.c

    rotation = args.rotation
    Omega0 = args.Omega0
    differential_rotation = args.diff_rot
    center = ((nx * dx)/2, (ny * dx)/2)

    init_type = args.init_type
    direction = args.direction
    disp_type = args.disp_type

    anti_diffusion = args.anti_diffusion

    # ---------------- OUTPUT FILE ----------------
    output_file = HD5_DIR.joinpath(f'{init_type}/advection_only2.h5')
    os.makedirs(output_file.parent, exist_ok=True)

    # ---------------- NUMERICAL SETUP ----------------
    x = np.linspace(0, nx * dx, nx, endpoint=False)
    y = np.linspace(0, nx * dx, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="xy")

    kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dx)
    kx = np.fft.fftshift(kx)
    ky = np.fft.fftshift(ky)
    KX, KY = np.meshgrid(kx, ky)

    omega_k = c * KY

    # ---------------- INITIAL CONDITION ----------------
    nk = initialize_field(init_type, nx, ny, dx, KX, KY, beta=45)
    n_xy = field_from_nk(nk)

    im = plt.pcolormesh(x, y, n_xy, cmap='seismic')
    plt.colorbar(im)
    plt.title("Initial condition")
    figpath = FIG_DIR.joinpath(f'{init_type}_initial_field.png')
    plt.savefig(figpath, dpi=150)
    plt.close()

    # ---------------- MAIN SIMULATION ----------------
    with h5py.File(output_file, "w", libver="latest") as f:
        save_data(f, "grid", False, x=x, y=y)
        f.attrs.update({
            "mode": mode, "U0": U0, "S": S, "c": c, "dt": dt,
            "rotation": rotation, "Omega0": Omega0,
            "differential_rotation": differential_rotation,
            "init_type": init_type, "direction": direction, 'dispersion': disp_type
        })

        print(f"\nStarting simulation ({mode})...")
        energy_ref = None
        for step in range(n_steps):
            nk = evolve_one_step(nk, step * dt, dt, include_phase, omega_k, x, X, Y,
                                 U0, S, rotation, Omega0, differential_rotation, center, mode, nx, ny, dx)
            if step % save_every == 0:
                n_xy = field_from_nk(nk)

                if anti_diffusion:
                    energy_now = np.sum(np.abs(n_xy)**2)
                    if energy_ref is None:
                        energy_ref = energy_now
                    else:
                        factor = np.sqrt(energy_ref / energy_now)
                        n_xy *= factor
                        nk = np.fft.fft2(n_xy)

                save_data(f, "fields", True, n=n_xy)
                if step % 10 == 0:
                    print(f"Step {step}/{n_steps}: max|n| = {np.abs(n_xy).max():.3e}")

    print(f"\nSimulation complete. Results saved to {output_file}\n")

    im = plt.pcolormesh(x, y, n_xy, cmap='seismic')
    plt.colorbar(im)
    plt.title("Final condition")
    figpath = FIG_DIR.joinpath(f'{init_type}_final_field.png')
    plt.savefig(figpath, dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D wave / turbulence propagation simulator")

    # Main numerical parameters
    parser.add_argument("--nx", type=int, default=1024)
    parser.add_argument("--ny", type=int, default=1024)
    parser.add_argument("--dx", type=float, default=2e-4)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--n_steps", type=int, default=200)
    parser.add_argument("--save_every", type=int, default=1)

    # Physical parameters
    parser.add_argument("--mode", type=str, default="y_velocity")
    parser.add_argument("--U0", type=float, default=3.0e-2)
    parser.add_argument("--S", type=float, default=0.0)
    parser.add_argument("--c", type=float, default=0.0)

    parser.add_argument("--rotation", action="store_true")
    parser.add_argument("--Omega0", type=float, default=0.5)
    parser.add_argument("--diff_rot", action="store_true")

    parser.add_argument("--init_type", type=str, default="tilt")
    parser.add_argument("--direction", type=str, default="horizontal")
    parser.add_argument("--disp_type", type=str, default="complex")
    parser.add_argument("--anti_diffusion", action="store_true")

    args = parser.parse_args()
    main(args)
