import Distributions.MvNormalStats, Distributions.lpgamma, Distributions.suffstats, Distributions.mean

function suffstats{Tf <: FloatingPoint}(D::Type{MvNormal}, x::DenseArray{Tf, 2}, 
        w::DenseArray{Float64}, z::DenseArray{Tf, 2})
    d = size(x, 1)
    n = size(x, 2)

    tw = sum(w)
    s = zeros(d)
    for i=1:n
        @devec s[:] += x[:, i] .* w[i]
    end
    m = s * inv(tw)
    for i in 1:n
        z_i = view(z, :, i)
        x_i = view(x, :, i)
        @devec z_i[:] = (x_i .- m) .* sqrt(w[i])
    end
    s2 = A_mul_Bt(z, z)

    MvNormalStats(s, m, s2, tw)
end

function posterior_cool(prior::NormalWishart, ss::MvNormalStats)
    if ss.tw < eps() return prior end

    mu0 = prior.mu
    kappa0 = prior.kappa
    TC0 = prior.Tchol
    nu0 = prior.nu

    kappa = kappa0 + ss.tw
    nu = nu0 + ss.tw
    mu = (kappa0.*mu0 + ss.s) ./ kappa

    Lam0 = inv(TC0)
    z = prior.zeromean ? ss.m : ss.m - mu0
    Lam = inv(Lam0 + ss.s2 + kappa0*ss.tw/kappa*(z*z'))

    return NormalWishart(mu, kappa, cholfact(Lam), nu)
end

function expected_logdet(nw::NormalWishart)
    logd = 0

    for i=1:nw.dim
        logd += digamma(0.5 * (nw.nu + 1 - i))
    end

    logd += nw.dim * log(2)
    logd += logdet(nw.Tchol)

    return logd
end

function expected_truncation_level(nw::NormalWishart)
    return nw.Tchol[:U]' * nw.Tchol[:U] * nw.nu 
end

function mean(nw::NormalWishart)
    T = expected_truncation_level(nw)
    return MvNormalCanon(T * nw.mu, T)
end

function lognorm(nw::NormalWishart)
    return (nw.dim / 2) * (log(2 * pi) - log(nw.kappa)) + (nw.nu / 2) * logdet(nw.Tchol) + (nw.dim * nw.nu / 2) * log(2.) + lpgamma(nw.dim, nw.nu / 2)
end

function entropy(nw::NormalWishart)
    en = 0.

    logd = expected_logdet(nw)

    en -= logd
    en -= nw.dim * (log(nw.kappa) - log(2*pi))
    en += nw.dim
    en /= 2
    
    en -= 0.5 * (nw.nu - nw.dim - 1) * logd
    en += nw.nu * nw.dim / 2
    en += 0.5 * nw.nu * logdet(nw.Tchol) + 0.5 * nw.nu * nw.dim * log(2.) + lpgamma(nw.dim, nw.nu/2)

    return en
end

function marginal_loglikelihood(prior::NormalWishart, posterior::NormalWishart, n::Float64)
    d = prior.dim
    return lognorm(posterior) - lognorm(prior) - (n*d/2) * log(2 * pi)
end

function gaussian_mixture{Tf <: FloatingPoint}(prior::NormalWishart, T::Int64, alpha::Float64, x::DenseArray{Tf, 2})
    dim, N = size(x)
    theta = Array(NormalWishart, T)

    buffer = similar(x)

    for k=1:T
        theta[k] = prior
    end

    function cluster_update(k::Int64, z::DenseArray{Float64})
        nw = posterior_cool(prior, suffstats(MvNormal, x, z, buffer))
        theta[k] = nw
    end

    function cluster_entropy(k::Int64)
        return entropy(theta[k])
    end

    function cluster_loglikelihood(k::Int64, z::DenseArray{Float64})
        post = posterior_cool(prior, suffstats(MvNormal, x, z, buffer))
        n = sum(z)

        ll = -lognorm(prior)

        nw = theta[k]
        ll -= nw.dim * n / 2 * log(2*pi)

        ll += (post.nu - post.dim)/2 * expected_logdet(nw)
        W = nw.Tchol[:U]' * nw.Tchol[:U]
        ll -= nw.nu * trace(inv(post.Tchol) * W) / 2

        dm = nw.Tchol[:U] * (nw.mu - post.mu)
        ll -= post.kappa * (nw.dim / nw.kappa + nw.nu * dot(dm, dm)) / 2

        return ll
    end

    function object_loglikelihood(k::Int64, i::Int64)
        nw = theta[k]
        ll = -nw.dim/2 * log(2*pi) + expected_logdet(nw)/2
        dx = nw.Tchol[:U] * (x[:, i] - nw.mu)
        ll -= (nw.dim / nw.kappa + nw.nu * dot(dx, dx)) / 2

        return ll
    end

    mm = TSBPMM(N, T, alpha, cluster_update,
            cluster_loglikelihood,
            object_loglikelihood, 
            cluster_entropy; random_init=true)

    function predictive_loglikelihood{Tf <: FloatingPoint}(xt::DenseArray{Tf, 2})
        nt = size(xt, 2)
        logp = zeros(nt)

        π = zeros(T)
        pi!(π, mm)

        pred = map(predictive, theta)

        for i=1:nt
            for k=1:T
                logp[i] += π[k] * pdf(pred[k], x[:, i])
            end
        end
        log!(logp)

        return logp
    end

    return mm, theta, predictive_loglikelihood
end

export gaussian_mixture
