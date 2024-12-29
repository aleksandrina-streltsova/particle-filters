import numpy as np
import scipy.stats as stats


def sample_trajectory(n_steps, mu0, sigma0, a, m, c, h, gamma):
    """
    Samples trajectory according to the following equations:

    u_j+1 = a * u_j + xi_j
    y_j+1 = h * u_j+1 + eta_j+1

    where u_0 ~ N(mu0, sigma0), xi_j ~ N(m, c), eta_j ~ N(0, gamma),

    :param mu0:     initial condition mean
    :param sigma0:  initial condition variance
    :param a:       process linear coefficient
    :param m:       process noise mean
    :param c:       process variance
    :param h:       observation linear coefficient
    :param gamma:   observation noise variance

    :return: sampled trajectory and observations
    """
    u = np.zeros((n_steps + 1,))
    y = np.zeros((n_steps + 1,))

    xi = stats.norm.rvs(m, c ** 0.5, size=(n_steps,))  # conditional distribution
    eta = stats.norm.rvs(0, gamma ** 0.5, size=(n_steps + 1,))  # noise

    for i in range(0, n_steps + 1):
        if i == 0:
            u[0] = stats.norm.rvs(mu0, sigma0 ** 0.5)  # initial condition
        else:
            u[i] = a * u[i - 1] + xi[i - 1]
        y[i] = h * u[i] + eta[i]

    return u, y


def kalman_filter(y, mu0, sigma0, a, m, c, h, gamma):
    """
    Estimates the state of a linear dynamical system with linear observations and Gaussian noise.
    For the description of the system see :py:func:`sample_trajectory`

    :param y: observations

    :return: estimated mean and variance for the state at each step
    """
    n_steps = len(y) - 1

    mu_p = np.zeros((n_steps + 1,))
    sigma_p = np.zeros((n_steps + 1,))

    mu_a = np.zeros((n_steps + 1,))
    sigma_a = np.zeros((n_steps + 1,))
    mu_a[0] = mu0
    sigma_a[0] = sigma0

    for j in range(1, n_steps + 1):
        # prediction step
        mu_p[j] = a * mu_a[j - 1] + m
        # sigma_p[j] = sigma_a[j - 1] * a + c
        sigma_p[j] = a * sigma_a[j - 1] * a + c

        # analysis step
        d = y[j] - h * mu_p[j]
        s = h * sigma_p[j] * h + gamma
        k = sigma_p[j] * h / s

        mu_a[j] = mu_p[j] + k * d
        sigma_a[j] = (1 - k * h) * sigma_p[j] * (1 - k * h) + k * gamma * k

    return mu_a, sigma_a


def ensemble_kf(y, n_particles, mu0, sigma0, a, m, c, h, gamma):
    """
    Estimates the state of a linear dynamical system with linear observations and Gaussian noise
    by approximating the distribution with an ensemble of particles.
    For the description of the system see :py:func:`sample_trajectory`

    :param y:           observations
    :param n_particles: number of particles

    :return: estimated particle states at each step
    """
    n_steps = len(y) - 1

    u_en = np.zeros((n_steps + 1, n_particles))
    y_en = np.zeros((n_steps + 1, n_particles))
    u_en_forecast = np.zeros((n_steps + 1, n_particles))

    xi_en = stats.norm.rvs(m, c ** 0.5, size=(n_steps, n_particles))
    eta_en = stats.norm.rvs(0, gamma ** 0.5, size=(n_steps + 1, n_particles))

    S = (np.eye(n_particles) - np.ones((n_particles, n_particles)) / n_particles)
    S2 = S @ S.T

    for j in range(0, n_steps + 1):
        # prediction step
        if j == 0:
            u_en_forecast[0] = stats.norm.rvs(mu0, sigma0 ** 0.5, size=(n_particles,))
        else:
            u_en_forecast[j] = a * u_en[j - 1] + xi_en[j - 1]

        # analysis step
        y_en[j] = h * u_en_forecast[j] + eta_en[j]

        # Y = y_en[j] @ S
        # U = u_en_forecast[j] @ S
        # K * Y * Y.T = U * Y.T
        K = (u_en_forecast[j] @ S2 @ y_en[j]) / (y_en[j] @ S2 @ y_en[j])
        u_en[j] = u_en_forecast[j] + K * (y[j] - y_en[j])

    return u_en


def bootstrap_pf(y, n_particles, mu0, sigma0, a, m, c, h, gamma, no_resampling=False):
    """
    Estimates the state of a linear dynamical system with linear observations and Gaussian noise
    by approximating the distribution with an ensemble of weighted particles.
    For the description of the system see :py:func:`sample_trajectory`

    :param y:           observations
    :param n_particles: number of particles

    :return: estimated particle states with their weights and estimated sample size at each step
    """
    n_steps = len(y) - 1

    u_bs = np.zeros((n_steps + 1, n_particles))
    w_bs = np.zeros((n_steps + 1, n_particles))
    u_bs_resample = np.zeros((n_steps + 1, n_particles))
    ess = np.zeros((n_steps + 1,))

    xi_bs = stats.norm.rvs(m, c ** 0.5, size=(n_steps, n_particles))

    for i in range(1, n_steps + 1):
        # resampling step
        if i == 1:
            u_bs_resample[0] = stats.norm.rvs(mu0, sigma0 ** 0.5)
        else:
            if not no_resampling:
                u_bs_resample[i - 1] = np.random.choice(u_bs[i - 1], n_particles, p=w_bs[i - 1])
            else:
                u_bs_resample[i - 1] = u_bs[i - 1]

        # prediction step
        u_bs[i] = a * u_bs_resample[i - 1] + xi_bs[i - 1]

        # analysis step
        w_bs[i] = stats.norm(0, gamma ** 0.5).pdf(y[i] - h * u_bs[i])
        w_bs[i] = w_bs[i] / w_bs[i].sum()

        ess[i] = 1 / n_particles / np.sum(w_bs[i] ** 2)

    return u_bs, w_bs, ess
