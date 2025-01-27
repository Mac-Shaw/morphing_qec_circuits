# morphing_qec_circuits
Repository for morphing quantum error correction circuits.

Contains helper functions to generate stim circuits corresponding to morphing QEC circuits. Contains a "MorphingSpecifications" class that stores the defining features of a CSS morphing QEC circuit (mid-cycle code stabilisers, contraction circuits). This is then used to generate stim circuits corresponding to memory and stability experiments using the morphing specification. Also contains helper functions to calculate the static code distance and an upper-bound on the circuit-level distance. Finally, there are a number of morphing circuits for Bivariate Bicycle codes, as described in `arXiv:2407.16336`.

# Installation

No particular tricks or special moves required to install this repo, just `git clone` the repository and run.

Required packages:
- `numpy`, `copy`, `galois`, `random`, `collections` for general calculations.
- `stim` and `surface_sim` for describing and simulating Clifford quantum circuits. `surface_sim` is available at `https://github.com/MarcSerraPeralta/surface-sim/tree/main`.
- `gurobipy` for calculating the static code distance (requires valid Gurobi license)

# Usage

See `Example.ipynb` for examples of usage.

# Contact

For troubleshooting issues, or general questions, contact me (Mackenzie Shaw) at `m.h.shaw@tudelft.nl`.
