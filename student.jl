using NumericExtensions
using Distributions

import NumericExtensions.dim

immutable MultivariateStudent
    nu::Float64
    mu::Vector{Float64}
    sigma::AbstractPDMat
end

#function MultivariateStudent(nu::Float64, mu::Vector{Float64},sigma::AbstractPDMat)
#    return new(nu, mu, sigma)
#end

function MultivariateStudent(nu::Float64, mu::Vector{Float64}, sigma::Matrix{Float64})
    return MultivariateStudent(nu, mu, PDMat(sigma))
end 

dim(t::MultivariateStudent) = dim(t.sigma)

function lognorm(t::MultivariateStudent)
    return lgamma(t.nu / 2) + (dim(t)/2) * (log(t.nu) + log(pi)) + 0.5 * logdet(t.sigma) - lgamma((t.nu + dim(t)) / 2)
end

function logpdf(t::MultivariateStudent, x::Vector{Float64})
    return - (t.nu + dim(t)) / 2 * log(1 + invquad(t.sigma, x - t.mu) / t.nu) - lognorm(t)
end

function pdf(t::MultivariateStudent, x::Vector{Float64})
    return exp(logpdf(t, x))
end

function predictive(nw::NormalWishart)
    nu = nw.nu - nw.dim + 1
    return MultivariateStudent(nu, nw.mu, inv(nw.Tchol) * (nw.kappa + 1)/(nw.kappa * nu))
end

