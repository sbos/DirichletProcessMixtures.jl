[![Build Status](https://travis-ci.org/sbos/DirichletProcessMixtures.jl.png)](https://travis-ci.org/sbos/DirichletProcessMixtures.jl)
[![DirichletProcessMixtures](http://pkg.julialang.org/badges/DirichletProcessMixtures_release.svg)](http://pkg.julialang.org/?pkg=DirichletProcessMixtures&ver=release)

DirichletProcessMixtures.jl
=======

This package implements Dirichlet Process Mixture Models in Julia using variational inference for truncated stick-breaking representation of Dirichlet Process.

## (almost) infinite mixture of Gaussians

Most likely you need this package especially for this purpose, this is how to do Gaussian clustering. You may check [demo code](demo.jl) which contains almost all functionality you may need.

First off, you define your prior over parameters of mixture component (i.e. mean and precision matrix) using `NormalWishart` distribution:
```julia
using DirichletProcessMixtures
using Distributions

prior = NormalWishart(zeros(2), 1e-7, eye(2) / 4, 4.0001)
```
Then you generate your mixture
```julia
x = ... # your data, x[:, i] - is i-th data point
T = 20 # truncation level
alpha = 0.1 # Dirichlet process parameter, controls how many clusters you need a priori
gm, theta, predictive_likelihood = gaussian_mixture(prior, T, alpha, x)
```

`gm` is an internal representation of mixture model. `theta` is array of size `T` whose elements refer to parameters of posterior `NormalWishart`'s. Finally, `predictive_likelihood` is a function which takes a matrix containing test data and returns per-point test loglikelihood. Now we can perform inference in our model

```julia
function iter_callback(mix::TSBPMM, iter::Int64, lower_bound::Float64)
    pl = sum(predictive_likelihood(xtest)) / M
    println("iteration $iter test likelihood=$pl, lower_bound=$lower_bound")
end

maxiter = 200
ltol = 1e-5
niter = infer(gm, maxiter, ltol; iter_callback=iter_callback)
```
You may see that `infer` method performs not more than `maxiter` iterations until lower bound tolerance reaches `ltol` value, calling `iter_callback` at each iteration if provided.

Another useful quantities you may need from mixture model:
* `gm.z` - TxN array with expected mixture component assignments
* `gm.qv` - posterior `Beta` distributions for stick-breaking proportions

## General interface

It is also possible to implement custom mixture models with conjugate priors for mixture components, but this remains to be documented yet. For a reference implementation of custom mixture model use [mixture of Gaussians](src/gaussian_mixture.jl).
