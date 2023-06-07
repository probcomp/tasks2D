"""
GridTasks2D.Distributions

Some Gen distributions used in the grid tasks.
"""
module Distributions
using Gen

export UniformFromSet, uniform_from_set

struct UniformFromSet <: Gen.Distribution{Any}; end
Gen.random(::UniformFromSet, set) = collect(set)[uniform_discrete(1, length(set))]
Gen.logpdf(::UniformFromSet, x, set) = -log(length(set))
uniform_from_set = UniformFromSet()
(::UniformFromSet)(args...) = random(uniform_from_set, args...)

end