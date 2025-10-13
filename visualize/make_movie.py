"""
make_movie.py
-------------
Create an animation (MP4 or GIF) from HDF5 simulation output.
Automatically falls back to GIF if ffmpeg is not available.
"""
#%%
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
import shutil
from pathlib import Path
from config import HD5_DIR, MP4_DIR, GIF_DIR

# ------------------ LOAD DATA ------------------
if len(sys.argv) < 2:
    print("Usage: python make_movie.py <file.h5>")
    sys.exit(1)

filename = HD5_DIR.joinpath('prova6.h5')
if not filename.exists():
    print(f"File not found: {filename}")
    sys.exit(1)

with h5py.File(filename, "r") as f:
    n = f["evolution/n"][:]
    x = f["grid/x"][:]
    y = f["grid/y"][:]

X, Y = np.meshgrid(x, y, indexing="xy")
Nt = n.shape[0]

# ------------------ FIGURE SETUP ------------------
fig, ax = plt.subplots(figsize=(6,5))
vmax = np.max(np.abs(n))
img = ax.pcolormesh(X, Y, n[0], shading="auto", cmap="seismic", vmin=-vmax, vmax=vmax)
plt.colorbar(img, ax=ax, label="n(x,y,t)")
ax.set_title("2D Field Evolution")
ax.set_xlabel("x")
ax.set_ylabel("y")

text_t = ax.text(0.02, 0.95, "", transform=ax.transAxes,
                 color="black", fontsize=10, va="top", ha="left",
                 bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

def update(frame):
    img.set_array(n[frame].ravel())
    text_t.set_text(f"Frame {frame+1}/{Nt}")
    return img, text_t

anim = FuncAnimation(fig, update, frames=Nt, interval=40, blit=False)

# ------------------ OUTPUT PATHS ------------------

out_mp4 = MP4_DIR.joinpath(filename.name).with_suffix('.mp4')
out_gif = GIF_DIR.joinpath(filename.name).with_suffix('.gif')

# ------------------ TRY FFMPEG FIRST ------------------
def ffmpeg_available():
    """Check if ffmpeg is installed and in PATH."""
    return shutil.which("ffmpeg") is not None

try:
    if ffmpeg_available():
        print(f"\n Using ffmpeg to render MP4: {out_mp4}")
        writer = FFMpegWriter(fps=25, metadata=dict(artist="Tokamak Edge Simulation"))
        anim.save(out_mp4, writer=writer, dpi=150)
        print(f"MP4 saved to {out_mp4}\n")
    else:
        raise FileNotFoundError("ffmpeg not found")
except Exception as e:
    print(f"\n Could not use ffmpeg ({e}). Switching to Pillow (GIF).")
    writer = PillowWriter(fps=15)
    anim.save(out_gif, writer=writer)
    print(f"GIF saved to {out_gif}\n")

# %%
