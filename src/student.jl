import PDMats.dim
import ConjugatePriors.NormalWishart

immutable MultivariateStudent
    nu::Float64
    mu::Vector{Float64}
    sigma::AbstractPDMat
end

function MultivariateStudent(nu::Float64, mu::Vector{Float64}, sigma::Matrix{Float64})
    return MultivariateStudent(nu, mu, PDMat(sigma))
end 

dim(t::MultivariateStudent) = dim(t.sigma)

function lognorm(t::MultivariateStudent)
    return lgamma(t.nu / 2) + (dim(t)/2) * (log(t.nu) + log(pi)) + 0.5 * logdet(t.sigma) - lgamma((t.nu + dim(t)) / 2)
end

function logpdf{Tf <: FloatingPoint}(t::MultivariateStudent, x::Vector{Tf})
    return - (t.nu + dim(t)) / 2 * log(1 + invquad(t.sigma, x - t.mu) / t.nu) - lognorm(t)
end

function pdf{Tf <: FloatingPoint}(t::MultivariateStudent, x::Vector{Tf})
    return exp(logpdf(t, x))
end

function predictive(nw::NormalWishart)
    nu = nw.nu - nw.dim + 1
    return MultivariateStudent(nu, nw.mu, inv(nw.Tchol) * (nw.kappa + 1)/(nw.kappa * nu))
end

