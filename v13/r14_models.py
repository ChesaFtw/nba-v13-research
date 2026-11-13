"""Round 14 count models: three-point attempts x conversion, and rebound opportunity GLM."""
import math

import numpy as np


def nb_pmf_vector(mu, phi, kmax):
    """P(X = 0..kmax) for X ~ NegBin(mean mu, Var = mu + phi mu^2)."""
    mu = max(mu, 1e-6); r = 1.0 / max(phi, 1e-6)
    p = np.empty(kmax + 1)
    p[0] = (r / (r + mu)) ** r
    for k in range(kmax):
        p[k + 1] = p[k] * (k + r) / (k + 1) * mu / (r + mu)
    return p


def nb_logpmf(k, mu, phi):
    mu = max(mu, 1e-6); r = 1.0 / max(phi, 1e-6)
    k = int(round(k))
    return (math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
            + r * math.log(r / (r + mu)) + k * math.log(mu / (r + mu)))


def threes_over_prob(mu_att, phi_att, pct, line, amax=40):
    """P(3PM > line) when attempts ~ NB(mu_att, phi_att) and makes | attempts ~ Binomial(attempts, pct)."""
    pa = nb_pmf_vector(mu_att, phi_att, amax)
    k = int(math.floor(line))
    total = 0.0
    for a in range(amax + 1):
        if a <= k:
            continue
        # P(Bin(a, pct) <= k)
        cdf, term = 0.0, (1 - pct) ** a
        for j in range(k + 1):
            cdf += term
            term *= (a - j) / (j + 1) * pct / max(1e-12, 1 - pct)
        total += pa[a] * (1 - cdf)
    return min(max(total, 1e-6), 1 - 1e-6)


def threes_logpmf(m, mu_att, phi_att, pct, amax=40):
    pa = nb_pmf_vector(mu_att, phi_att, amax)
    m = int(round(m))
    s = 0.0
    for a in range(m, amax + 1):
        s += pa[a] * math.comb(a, m) * pct ** m * (1 - pct) ** (a - m)
    return math.log(max(s, 1e-300))


class PoissonGLM:
    """log E[y] = log(offset) + b0 + X b, ridge-penalized (not the intercept); IRLS."""

    def __init__(self, ridge=1.0, iters=50):
        self.ridge, self.iters = ridge, iters

    def fit(self, X, y, offset):
        X = np.asarray(X, float); y = np.asarray(y, float); lo = np.log(np.maximum(offset, 1e-6))
        self.mu_, self.sd_ = X.mean(0), X.std(0)
        self.sd_[self.sd_ < 1e-9] = 1.0
        Z = np.column_stack([np.ones(len(X)), (X - self.mu_) / self.sd_])
        b = np.zeros(Z.shape[1])
        b[0] = math.log(max(y.mean(), 1e-6) / max(np.exp(lo).mean(), 1e-6))
        pen = np.eye(Z.shape[1]) * self.ridge; pen[0, 0] = 0
        self.converged_ = False
        for _ in range(self.iters):
            eta = np.clip(lo + Z @ b, -20, 20); m = np.exp(eta)
            g = Z.T @ (y - m) - pen @ b
            H = Z.T @ (Z * m[:, None]) + pen + np.eye(Z.shape[1]) * 1e-8
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                step = np.linalg.lstsq(H, g, rcond=None)[0]
            if not np.all(np.isfinite(step)):
                break
            mx = np.max(np.abs(step))
            if mx > 2.0:               # damped Newton: never jump more than 2 log-units per iteration
                step *= 2.0 / mx
            b = b + step
            if mx < 1e-8:
                self.converged_ = True
                break
        self.beta_ = b
        H = Z.T @ (Z * np.exp(np.clip(lo + Z @ b, -20, 20))[:, None]) + pen + np.eye(Z.shape[1]) * 1e-8
        try:
            self.se_ = np.sqrt(np.clip(np.diag(np.linalg.inv(H)), 0, None))
        except np.linalg.LinAlgError:
            self.se_ = np.full(Z.shape[1], np.nan)
        return self

    def predict(self, X, offset):
        X = np.asarray(X, float)
        Z = np.column_stack([np.ones(len(X)), (X - self.mu_) / self.sd_])
        return np.exp(np.clip(np.log(np.maximum(offset, 1e-6)) + Z @ self.beta_, -20, 20))


def nb_dispersion(mu, y):
    mu, y = np.asarray(mu, float), np.asarray(y, float)
    return max(float(np.mean((y - mu) ** 2 - mu) / max(1e-9, np.mean(mu ** 2))), 1e-6)
