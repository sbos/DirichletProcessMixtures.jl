require("src/DirichletProcessMixtures.jl")

using DirichletProcessMixtures
using Distributions
using ConjugatePriors

import ConjugatePriors.NormalWishart

function ball(N::Int64, x::Float64, y::Float64)
    return randn(2, N) .+ [x, y]
end

function balley(M::Int64, R::Float64)
    return hcat(ball(M, 0., 0.),
            ball(M,  R,  R),
            ball(M,  R, -R),
            ball(M, -R,  R),
            ball(M, -R, -R))
end

B = 60
x = balley(B, 3.)
xtest = balley(B, 3.)

N = B * 5
M = N

prior = NormalWishart(zeros(2), 1e-7, eye(2) / 4, 4.0001)

T = 20
maxiter = 4000
gm, theta, predictive_likelihood = gaussian_mixture(prior, T, 1e-1, x)

lb_log = zeros(maxiter)
tl_log = zeros(maxiter)

tic()
function iter_callback(mix::TSBPMM, iter::Int64, lower_bound::Float64)
    pl = sum(predictive_likelihood(xtest)) / M
    lb_log[iter] = lower_bound
    tl_log[iter] = pl
    toc()
    println("iteration $iter test likelihood=$pl, lower_bound=$lower_bound")
    tic()
end

niter = infer(gm, maxiter, 1e-5; iter_callback=iter_callback)

using PyCall
@pyimport pylab

#convergence plot
#pylab.plot([1:niter], lb_log[1:niter]; color=[1., 0., 0.])
pylab.plot([1:niter], tl_log[1:niter]; color=[0., 0., 1.])

pylab.show()
    
z = map_assignments(gm)
for k=1:T
xk = x[:, z .== k]
    pylab.scatter(xk[1, :], xk[2, :]; color=rand(3))
end
pylab.show()
