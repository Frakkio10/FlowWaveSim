#                   2D FLOW WAVE SIMULATION
A lightweight, modular Python framework for simulating 2D wave and turbulence propagation in a rectangular plasma domain — including poloidal flow, shear, phase velocity, and optional rotation.
Designed for tokamak edge plasma physics studies and fast prototyping of transport phenomena.
        
#                    FEATURES
2D grid-based propagation using FFT and gradient methods

Supports:

  - Poloidal advection U_y(x)
    
  - Shear flow (S ≠ 0)
    
  - Optional phase velocity (for the moment: ω = c|k|, later from GK)
    
  - Solid-body or differential rotation (Ω(x))
    
Real-valued fields via Hermitian symmetrization

Output stored in HDF5 (*.h5) for portability

Visualization and diagnostics included

#                    ISTALLATION
git clone [https://github.com/<your-username>/Flow-wave-sim.git](https://github.com/Frakkio10/FlowWaveSim.git)

cd FlowWaveSim

pip install --user --editable .


If you don’t have a requirements.txt yet, use:

        - numpy

        - matplotlib
        
        - h5py

        - imageio[ffmpeg]

#                   RUNNING A SIMULATION
python wave_simulation.py


This will:

Initialize the field in Fourier space

Evolve it over time (advection + optional phase)

Save results to an HDF5 file (e.g. simulation_run1.h5)

All parameters (grid size, shear, rotation, etc.) can be configured at the top of the script.

#                    CREATING ANIMATION
After simulation, generate an .mp4 movie:

python visualize/make_movie.py simulation_run1.h5

This script:

Reads the field evolution (n(x, y, t))

Builds a time-lapse using Matplotlib

Exports a smooth .mp4 video using FFmpeg

Ensure ffmpeg is available (installed via pip install imageio[ffmpeg] or system package).

#                    EXAMPLE OUTPUT
Quantity	Visualization
n(x, y, t=0)	
n(x, y, t=end)	


#                    USEFUL TIPS

For stability, reduce dt or S if the field “explodes.”

Use Nx, Ny = 512 for faster tests.

The HDF5 structure is designed for easy parallel extensions.

#                    AUTHOR
Author: **FRANCESCO ORLACCHIO** 

Affiliation: **LABORATOIRE DE PHYSIQUE DES PLASMAS (LPPP) ECOLE POLYTECHNIQUE** 

Email: **fracnesco.orlacchio@lpp.polytechnique.fr** 
s
License: MIT 
