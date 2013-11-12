using Distributions
using NumericExtensions
using Devectorize

import NumericExtensions.entropy

immutable TSBPMM
    α::Float64
    qv::Vector{Beta}
    π::Vector{Float64}
    z::Matrix{Float64}

    cluster_update::Function
    cluster_loglikelihood::Function
    object_loglikelihood::Function
    cluster_entropy::Function

    function TSBPMM(N::Int64, T::Int64, α::Float64, 
            cluster_update::Function,
            cluster_loglikelihood::Function,
            object_loglikelihood::Function,
            cluster_entropy::Function)
        return new(α, [Beta(1., α) for i=1:T-1], 
                ones(T) / T,
                ones(N, T) / T,
                cluster_update,
                cluster_loglikelihood,
                object_loglikelihood,
                cluster_entropy)
    end
end

N(mix::TSBPMM) = size(mix.z, 1)
T(mix::TSBPMM) = size(mix.z, 2)

function infer(mix::TSBPMM, niter::Int64, ltol::Float64)
    prev_lb = variational_lower_bound(mix)
    println("TSBPMM iteration 0, lbound=$prev_lb")
    for iter=1:niter
        variational_update(mix)
        
        lb = variational_lower_bound(mix)
        println("TSBPMM iteration $iter, lbound=$lb")

        @assert lb > prev_lb "Not monotone"
        if abs(lb - prev_lb) < ltol
            println("Converged")
            return lb
        end

        prev_lb = lb
    end
end

function variational_update(mix::TSBPMM)
    z = zeros(T(mix))
    for i=1:N(mix)
        for k=1:T(mix)
            z[k] = mix.π[k] + mix.object_loglikelihood(k, i)
        end

        @devec z[:] -= max(z)
        exp!(z)
        @devec z[:] ./= sum(z)

        assert(abs(sum(z) - 1.) < 1e-7)

        mix.z[i, :] = z
    end

    ts = 0.
    for k=T(mix):-1:1
        zk = unsafe_view(mix.z, :, k)
        mix.cluster_update(k, zk)
        zs = sum(zk)
        if k < T(mix)
            mix.qv[k] = Beta(1. + zs, mix.α + ts)
        end
        ts += zs
    end

    logpi!(mix.π, mix)    
end

function variational_lower_bound(mix::TSBPMM)
    return loglikelihood(mix) + entropy(mix)
end

logmean(beta::Beta) = digamma(beta.alpha) - digamma(beta.alpha + beta.beta)

loginvmean(beta::Beta) = digamma(beta.beta) - digamma(beta.alpha + beta.beta)

function logpi!(π::Vector{Float64}, mix::TSBPMM)
    r = 0.
    for k=1:T(mix)-1
        π[k] = logmean(mix.qv[k]) + r
        r += loginvmean(mix.qv[k])
    end
    π[T(mix)] = r
end

function loglikelihood(mix::TSBPMM)
    ll = 0.

    ts = 0.
    for k=T(mix):-1:1
        zk = unsafe_view(mix.z, :, k)
#        zk = mix.z[:, k]
        ll += mix.cluster_loglikelihood(k, zk)
        assert(!isnan(ll))

        zs = sum(zk)
        if k <= T(mix) - 1
            qv = mix.qv[k]        
            ll += zs * logmean(qv) + (mix.α+ts-1) * loginvmean(qv) - lbeta(1., mix.α)
            assert(!isnan(ll))
        end

        ts += zs
    end

    return ll
end

function entropy(mix::TSBPMM)
    ee = 0.
    ee += entropy(mix.z)

    for k=1:T(mix)
        if k < T(mix)
            ee += entropy(mix.qv[k])
        end
        ee += mix.cluster_entropy(k)
    end

    return ee
end

function map_assignments(mix::TSBPMM)
    z = zeros(Int64, N(mix))
    for i=1:N(mix)
        z[i] = indmax(sub(mix.z, i, :))
    end
    return z
end

export TSBPMM, infer, variational_lower_bound, map_assignments

