using Gen

struct DiagonalNormal <: Distribution{Vector{Float64}} end
const diagnormal = DiagonalNormal()

function Gen.random(::DiagonalNormal, mus::AbstractVector{U}, stds::AbstractVector{V}) where {U <: Real, V <: Real}
    return [normal(mu, std) for (mu,std) in zip(mus, stds)]
end

function Gen.random(::DiagonalNormal, mus::AbstractVector{U}, std::V) where {U <: Real, V <: Real}
    return [normal(mu, std) for mu in mus]
end

(::DiagonalNormal)(mus::AbstractVector{U}, stds::AbstractVector{V})  where {U <: Real, V <: Real} = Gen.random(DiagonalNormal(), mus, stds)
(::DiagonalNormal)(mus::AbstractVector{U}, std::V)  where {U <: Real, V <: Real} = Gen.random(DiagonalNormal(), mus, std)


function Gen.logpdf(::DiagonalNormal, xs::AbstractVector{T},
                mus::AbstractVector{U}, stds::AbstractVector{V}) where {T <: Real, U <: Real, V <: Real}
    log_p = 0.0
    for (x, mu, std) in zip(xs, mus, stds)
        log_p += Gen.logpdf(normal, x, mu, std)
    end
    return log_p
end

function Gen.logpdf(::DiagonalNormal, xs::AbstractVector{T},
                mus::AbstractVector{U}, std::V) where {T <: Real, U <: Real, V <: Real}
    log_p = 0.0
    for (x, mu) in zip(xs, mus)
        log_p += Gen.logpdf(normal, x, mu, std)
    end
    return log_p
end


Gen.has_output_grad(::DiagonalNormal) = true
Gen.has_argument_grads(::DiagonalNormal) = (true, true)


abstract type ProductDistribution{T} <: Distribution{Vector{T}} end

function unsqueeze(a, d)
        if ndims(a) == 0
            return [a]
        end
        if d<0
            d = ndims(a) - d
        end
        return reshape(a, (size(a)[1:d-1]..., 1, size(a)[d:end]...))
end

function mycat(xs::Vector{T}; dims) where T
    d = dims
    if d<0
        d = ndims(xs[1]) + d + 1
    end
    return cat(xs...; dims=d)
end

function num_factors(args::Tuple, d::Int)
    a = args[1]
    d = d > 0 ? d : ndims(a)+d+1
    return size(a, d)
end

function slicedim(a, d::Int, i::Int)
    if ndims(a) == 1
        return a[i]
    end
    sel = selectdim(a, d > 0 ? d : ndims(a)+d+1, i)
    T = eltype(sel)
    return AbstractArray{T}(sel)
    # Why the conversion to `AbstractArray`?
    # See `slicedim_old` below.
    # TL;DR - Unfortunately some distributions expect concrete vector types
    # and can't handle something that only resolves to an abstract vector.
end

function slicedim_old(a, d::Int, i::Int)
    sel = selectdim(a, d > 0 ? d : ndims(a)+d+1, i)
    return ndims(a) == 1 ? a[i] : sel
end

struct HomogeneousProduct{T} <: ProductDistribution{T}
    dist::Distribution{T}
    slicedim::Int  # indicates the dimension specifying
                   # the arguments for each dist
                   # Default is along 1st dimension - differs
                   # from Gen's mixtures.
end
ProductDistribution(dist::Distribution{T}, s::Int) where T = HomogeneousProduct{T}(dist, s)
ProductDistribution(dist::Distribution{T})         where T = HomogeneousProduct{T}(dist, 1)


function Gen.random(Q::HomogeneousProduct, args...)
    p = Q.dist
    d = Q.slicedim
    n = num_factors(args, d)

    ys = [p((slicedim(a, d, i) for a in args)...) for i=1:n]

    if d == 1 return ys end

    # This part slows it down quite a lot
    # but technically is necessary to logpdf eval
    # actual samples
    ys = [
        unsqueeze(y, d) for y in ys
    ]
    ys = mycat(ys, dims=d)
    return ys
end

(Q::HomogeneousProduct)(args...) = Gen.random(Q, args...)

function Gen.logpdf(Q::HomogeneousProduct{T}, xs, args...) where T
    p = Q.dist
    d = Q.slicedim
    n = num_factors(args, d)
    return sum([
        Gen.logpdf(p, slicedim(xs, d, i), (slicedim(a, d, i) for a in args)...) for i=1:n
    ])
end

function Gen.logpdf_grad(Q::HomogeneousProduct{T}, xs, args...) where T

    p = Q.dist
    d = Q.slicedim
    n = num_factors(args, d)
    k = length(args) + 1
    grads = [
        Gen.logpdf_grad(p, slicedim(xs, d, i), (slicedim(a, d, i) for a in args)...) for i=1:n
    ]
    grad_slices = [
        [unsqueeze(grads[i][j], d) for i=1:n] for j=1:k
    ]
    rearranged_grads = [
        mycat(slice,dims=d) for slice in grad_slices
    ]

    return rearranged_grads
end

Gen.has_output_grad(Q::HomogeneousProduct)    = Gen.has_output_grad(Q.dist)
Gen.has_argument_grads(Q::HomogeneousProduct) = Gen.has_argument_grads(Q.dist)
