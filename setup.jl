import Pkg;
Pkg.add(url="https://github.com/bicycle1885/Fmt.jl")
Pkg.add(url="https://github.com/probcomp/DynamicForwardDiff.jl")
Pkg.add(url="https://github.com/probcomp/GenTraceKernelDSL.jl")
Pkg.add(url="https://github.com/probcomp/GenSMCP3.jl")
Pkg.develop(path="./GridWorlds")
Pkg.develop(path="./LineWorlds")

try
    Pkg.develop("path=../GenPOMDPs.jl")
catch e
    @error "Setup file expected GenPOMDPs to be locally installed at ../.  GenPOMDPs was not found here; manual installation is therefore required."
    throw(e)
end