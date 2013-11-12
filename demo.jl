using DPMM
using Distributions
    
x = readdlm("balley_10_4.csv", ',')'
prior = NormalWishart(zeros(2), 1e-7, eye(2), 3.001)
T = 20
gm = gaussian_mixture(prior, T, 0.5, x)

infer(gm, 4000, 1e-3)

using PyCall
@pyimport pylab

z = map_assignments(gm)
for k=1:T
    xk = x[:, z .== k]
    pylab.scatter(xk[1, :], xk[2, :]; color=rand(3))
end
pylab.show()
