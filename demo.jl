using DPMM
using Distributions
    
function gen_sin(N::Int64)
    x = zeros(2, N)
    step = 4*pi / N

    for i=1:N
        t = i * step - 6
        x[1, i] = t + randn() * 0.2
        x[2, i] = 3 * (sin(t) + randn() * 0.2)
    end

    x = x[:, randperm(N)]

    return x
end

x = readdlm("balley_10_3.csv", ',')'
xtest = readdlm("balley_3_3.csv", ',')'
#x = gen_sin(100)
#xtest = gen_sin(100)
prior = NormalWishart(zeros(2), 1e-7, eye(2) / 4, 4.0001)
T = 20
maxiter = 4000
M = size(xtest, 2)
gm, theta, predictive_likelihood, posterior_draw = gaussian_mixture(prior, T, 1e-1, x)

lb_log = zeros(maxiter)
tl_log = zeros(maxiter)

function iter_callback(mix::TSBPMM, iter::Int64, lower_bound::Float64)
    pl = sum(predictive_likelihood(xtest))
    lb_log[iter] = lower_bound
    tl_log[iter] = pl
    println("iteration $iter test likelihood=$pl, lower_bound=$lower_bound")
end

niter = infer(gm, maxiter, 1e-5; iter_callback=iter_callback)

using PyCall
@pyimport pylab

#convergence plot
pylab.plot([1:niter], lb_log[1:niter]; color=[1., 0., 0.])
pylab.plot([1:niter], tl_log[1:niter]; color=[0., 0., 1.])

pylab.show()

zn, xn = posterior_draw(300)
for k=1:T
xk = xn[:, zn .== k]
    pylab.scatter(xk[1, :], xk[2, :]; color=rand(3))
end
pylab.show()
    
#z = map_assignments(gm)
#for k=1:T
#xk = x[:, z .== k]
#    pylab.scatter(xk[1, :], xk[2, :]; color=rand(3))
#end
#pylab.show()
