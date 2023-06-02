struct UniformFromSet <: Gen.Distribution{Any}; end
Gen.random(::UniformFromSet, set) = collect(set)[uniform_discrete(1, length(set))]
Gen.logpdf(::UniformFromSet, x, set) = -log(length(set))
uniform_from_set = UniformFromSet()
(::UniformFromSet)(args...) = random(uniform_from_set, args...)