module DirichletProcessMixtures

using NumericExtensions
using Devectorize
using ArrayViews
using Distributions
using ConjugatePriors
using PDMats

include("TSBPMM.jl")
include("student.jl")
include("gaussian_mixture.jl")

end
