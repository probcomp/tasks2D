# tasks2d: experiments with decision tasks in 2D environments

### Directory

Navigating this repository:
- `data/`: Datasets
- `notebooks_clean/`: Fairly clean Jupyter notebooks.
- `Tasks2D/`: Julia package used to load the Julia environment needed to run the notebooks.
    This package also contains some basic utilities used in the notebooks.
- `GridWorlds/`: Simple Julia package for 2D gridworld environments.
- `LineWorlds/`: Simple Julia package for continuous-space 2D environments, with objects in the environment represented as collections of line segments.  This also contains
code for raycasting in line environments.  (Note that gridworld environments
utilizing raycasting must load in the LineWorlds raycaster. See `GridWorlds/src/raytrace.jl` for etails.)
- `notebooks_messy/`: Messy Jupyter notebooks used during development.  No promises that the code in these still runs.

Also relevant:
- [The GenPOMDPs repository.](https://github.com/probcomp/GenPOMDPs.jl)

### Installation

- I recommend running this on an AWS instance with a GPU.  (The code runs much faster on GPU, and the GPU code is only compatible
  with CUDA.  I also haven't tested on CPU only in a little while.)
  - > To launch the same type of EC2 instance I am using:
        Go to AWS (register on aws.mit.edu if you haven't yet).  Go to EC2.  Click "Launch Instance".  Click "Browse AMIs" and find "Deep Learning AMI GPU PyTorch 1.13.1 (Ubuntu 20.04)".
        Set up the instance with a bit of storage (I didn't need a ton.), and whatever other setup details it asks for.  Then SSH onto the instance.  (It will prompt you to set up an SSH key
        on the way, which you'll need to download to your computer to SSH from your computer's shell.)  Then what I did is download the VSCode extension for editing remotely, and added the SSH URL as a remote host.  (By default, the URL will be different each time the AWS instance launches, so to keep it static, you may also want to set up an "Elastic IP" in AWS, and then associate that with the instance.  Then the URL will be static, so VSCode can more easily connect to it without manual work each time.)  Then I just used the built-in Jupyter notebook viewer in VSCode!  (Had to install Julia in VScode, and in the AWS instance.  For that, I installed juliaup via a curl command I found online, and used it to install Julia 1.8.)
- Use Julia 1.8 [not 1.9; for some reason this causes bugs with the current version of the CUDA raycasting code]
- Git clone this repository.
- From within the `tasks2D/` directory (top level directory of the repo), activate Julia 1.8.
- From within Julia, run `] activate Tasks2D` to activate the `Tasks2D` environment.
- From within Julia, run `julia> include("setup.jl")` to install a few dependencies which the package
  manager has trouble handling the standard way.  (If this doesn't work, try manually installing these packages.
  This can take a little while to run, btw.)
- From within Julia, run `] instantiate`.
- Restart the Julia session.
- From Julia, try running `julia> using Tasks2D`; `julia> using GenSMCP3`, `julia> using GenPOMDPs` `julia> using LineWorlds`, `julia> using GridWorlds` to make sure everything is working.
  (All these packages should be able to load.)
- Then, fire up Jupyter notebook, and navigate to the `notebooks_clean/` directory.  (Or, if you are using the VSCode notebook viewer, open the notebooks in VScode.)  You should be able to run the notebooks there.
  (I have tested `KidnappedRobot.ipynb` and `RoomSearch.ipynb` recently, and they should work.  The `Localization.ipynb` demo is older,
  and uses a different plotting library.  It has interactive demo code, though, which is nice, and may be worth checking out if anyone
  wants to try to develop interactive demos for the ProbProg summer school.  This plotting library works in browser Jupyter notebooks, but not in VSCode Jupyter notebooks.)