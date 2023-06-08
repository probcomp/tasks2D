# tasks2d: experiments with decision tasks in 2D environments

### Directory

Navigating this repository:
- `data/`: Datasets
- `notebooks_clean/`: Fairly clean Jupyter notebooks.
- `Tasks2D/`: Julia package used to load the Julia environment needed to run the notebooks.
    This package also contains some basic utilities used in the notebooks.
- `GridWorlds/`: Simple Julia package for 2D gridworld environments.
- `LineWorlds/`: Simple Julia package for continuous-space 2D environments, with objects in the environment represented as collections of line segments.
- `notebooks_messy/`: Messy Jupyter notebooks used during development.  No promises that the code in these still runs.

Also relevant:
- [The GenPOMDPs repository.](https://github.com/probcomp/GenPOMDPs.jl)

### Installation
At some point I intend to set up all the dependencies properly so this is easy to install.  I haven't done this yet.  In the interrim --

You will need to activate the `Tasks2D` enviornment, and play a bit of error whack-a-mole to set up the proper development dependencies.  Feel free to message me if you want to use this and need help understanding the errors.