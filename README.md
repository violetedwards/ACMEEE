# ACMEEE

ACMEEE (Avoided Crossings in the Movement of Exact Electronic Excitation)

Python package for tracking electronic excitations as one-dimensional potential wells move.

Made by Violet Edwards. Email me at ie4005@hw.ac.uk if you need anything :)

## Installation

Install the published package with pip:

```bash
python -m pip install ACMEEE
```

To install the current GitHub version instead:

```bash
python -m pip install git+https://github.com/violetedwards/acmeee.git
```

## Quick start

See testing.py for an example

`assemble` saves energy, wavefunction, density, and overlap outputs to the
requested output directory. Set `save_states=True` to save many body state objects and
`save_hf_orbitals=True` to save the Hartree-Fock orbital information.

The potential name parameter selects one of the built in potentials in
`ACMEEE.potential.potential`.

`trackedstate` selects the starting excitation.

See the `assemble` function parameters for controls on step size, overlap
tolerance, and the number of states generated above the tracked state.

