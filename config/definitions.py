#%%
from pathlib import Path 

ROOT_DIR = Path(__file__).resolve().parent.parent

OUT_DIR  = ROOT_DIR.joinpath('output/')
MP4_DIR  = OUT_DIR.joinpath('mp4/')
HD5_DIR  = OUT_DIR.joinpath('h5/')
GIF_DIR  = OUT_DIR.joinpath('gif/')

# %%
