module DirichletProcessMixtures

using ConjugatePriors
using Devectorize
using ArrayViews
using Distributions
using PDMats

include("TSBPMM.jl")
include("student.jl")
include("gaussian_mixture.jl")

# Multidimensional gamma / partial gamma function
function lpgamma(p::Int, a::Float64)
    res::Float64 = p * (p - 1.0) / 4.0 * log(pi)
    for ii in 1:p
        res += Distributions.lgamma(a + (1.0 - ii) / 2.0)
    end
    return res
end
end
