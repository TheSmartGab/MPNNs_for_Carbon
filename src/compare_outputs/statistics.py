"""Statistical computation functions for comparing model outputs and computing uncertainty quantification.

This module provides utilities for Bayesian posterior sampling of mean and variance parameters,
and computing statistical error metrics (MSE, RMSE, MAE) with confidence intervals using non-informative priors.

Functions:
    sample_joint_posterior_noninformative: Sample from joint posterior of mu and sigma^2 under Jeffreys prior
    compute_mse: Compute Mean Squared Error from posterior samples
    compute_mae: Compute Mean Absolute Error from posterior samples
    compute_stats_distribution: Compute full distribution of error metrics
    compute_stats_quantiles: Compute quantile values for statistical metrics
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import invgamma, norm
from scipy.special import erf


# I have checked that 10000 should be a good default value of samples. it controlls how many samples are drawn to sample the mu, sigma2 distributions.
def sample_joint_posterior_noninformative(mean, s2, N, n_samples=10000):
    """Sample from the joint posterior of mean and variance under Jeffreys prior.

    This function draws samples from p(mu, sigma^2 | data) using the non-informative
    Jeffrey's prior. It first samples sigma^2 from an inverse-gamma distribution,
    then samples mu conditional on sigma^2 from a normal distribution.

    Args:
        mean: Float or array-like sample mean (or expected value of mu).
        s2: Float or array-like sample variance estimate.
        N: Integer number of data points used to compute mean and s2.
        n_samples: Integer number of posterior samples to draw (default 10000).

    Returns:
        tuple: (mu_samples, sigma2_samples) where each is an array of size n_samples.
    """
    # Posterior parameters under Jeffreys prior
    alpha = (N - 1) / 2
    beta = (N - 1) * s2 / 2

    # Sample sigma^2
    sigma2 = invgamma.rvs(alpha, scale=beta, size=n_samples)

    # Sample mu | sigma^2
    mu = norm.rvs(loc=mean, scale=np.sqrt(sigma2 / N))

    return mu, sigma2


def compute_mse(mu, sigma2):
    """Compute Mean Squared Error from posterior samples of mean and variance.

    MSE = E[(x - x_true)^2] = (E[x - x_true])^2 + Var(x) = mu^2 + sigma2

    Args:
        mu: Array-like posterior samples of the mean/error centering.
        sigma2: Array-like posterior samples of the variance.

    Returns:
        array: MSE values computed as mu**2 + sigma2.
    """
    return mu ** 2 + sigma2


PREFACTOR = np.sqrt(2 / np.pi)


def compute_mae(mu, sigma2):
    """Compute Mean Absolute Error from posterior samples of mean and variance.

    MAE = E[|x - x_true|] is computed using the closed-form expression for the
    absolute value of a normal random variable.

    Args:
        mu: Array-like posterior samples of the mean/error centering.
        sigma2: Array-like posterior samples of the variance.

    Returns:
        array: MAE values computed using the closed-form formula involving erf.
    """
    ratio = mu / np.sqrt(2 * sigma2)
    return PREFACTOR * np.sqrt(sigma2) * np.exp(-(ratio ** 2)) + mu * erf(ratio)


def compute_stats_distribution(mus, sigmas2):
    """Compute full distribution of error metrics (MSE, RMSE, MAE, MEAN, VARIANCE).

    Args:
        mus: Array-like posterior samples of the mean.
        sigmas2: Array-like posterior samples of the variance.

    Returns:
        dict: Dictionary with keys 'MSE', 'RMSE', 'MAE', 'MEAN', 'VARIANCE' and array values.
    """
    mses = compute_mse(mus, sigmas2)
    rmses = np.sqrt(mses)
    maes = compute_mae(mus, sigmas2)

    return {"MSE": mses, "RMSE": rmses, "MAE": maes, "MEAN": mus, "VARIANCE": sigmas2}


def compute_stats_quantiles(mean, s2, N, nsamples=10000, quantiles=[0.025, 0.5, 0.975]):
    """Compute quantile values for statistical metrics from posterior samples.

    This function computes Bayesian credible intervals (e.g., 95% CI using 2.5% and 97.5% quantiles)
    for MSE, RMSE, MAE, MEAN, and VARIANCE metrics based on posterior sampling.

    Args:
        mean: Float or array-like sample mean estimate.
        s2: Float or array-like sample variance estimate.
        N: Integer number of data points used to compute mean and s2.
        nsamples: Integer number of posterior samples to draw (default 10000).
        quantiles: List of quantile probabilities to compute (default [0.025, 0.5, 0.975]).

    Returns:
        dict or None: Nested dictionary with metric names as keys and quantile values as nested dict.
                     Returns None if N < 2 (cannot compute meaningful statistics).
    """
    # cannot do much with a single data
    if N < 2:
        return None

    mus, sigmas2 = sample_joint_posterior_noninformative(mean, s2, N, nsamples)

    distributions = compute_stats_distribution(mus, sigmas2)

    # a bit pythonic but should be efficient with list and dict comprehension
    quantile_values = {
        key: {q: np.quantile(distribution, q) for q in quantiles}
        for key, distribution in distributions.items()
    }

    return quantile_values
