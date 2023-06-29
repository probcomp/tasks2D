import Pkg;
Pkg.add(url="https://github.com/probcomp/DynamicForwardDiff.jl")
Pkg.add(url="https://github.com/probcomp/GenTraceKernelDSL.jl")
Pkg.add(url="https://github.com/probcomp/GenSMCP3.jl")
Pkg.develop(path="./GridWorlds")
Pkg.develop(path="./LineWorlds")