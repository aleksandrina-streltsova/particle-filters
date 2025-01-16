import numpy as np


def sample_trajectory(n_steps, p0, p, nu):
    """
    Samples trajectory according to the provided dynamical system parameters:

    u_0   <-- p0
    u_j+1 <-- p(u_j)
    y_j+1 <-- nu(u_j+1)

    :param p0:  initial distribution
    :param p:   transition model
    :param nu:  observation model

    :return: sampled trajectory and observations
    """
    u = np.zeros((n_steps + 1,))
    y = np.zeros((n_steps + 1,))

    for i in range(0, n_steps + 1):
        if i == 0:
            u[0] = p0.rvs()
        else:
            u[i] = p(u[i - 1]).rvs()
        y[i] = nu(u[i]).rvs()

    return u, y


def kalman_filter(y, mu0, sigma0, a, m, c, h, gamma):
    """
    Estimates the state of a linear dynamical system with linear observations and Gaussian noise.
    For the description of the system see :py:mod:`linear`.

    :param y:       observations
    :param mu0:     initial condition mean
    :param sigma0:  initial condition variance
    :param a:       process linear coefficient
    :param m:       process noise mean
    :param c:       process variance
    :param h:       observation linear coefficient
    :param gamma:   observation noise variance

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
        sigma_p[j] = a * sigma_a[j - 1] * a + c

        # analysis step
        d = y[j] - h * mu_p[j]
        s = h * sigma_p[j] * h + gamma
        k = sigma_p[j] * h / s

        mu_a[j] = mu_p[j] + k * d
        sigma_a[j] = (1 - k * h) * sigma_p[j]

    return mu_a, sigma_a


def ensemble_kf(y, n_particles, p0, p, nu):
    """
    Estimates the state of a dynamical system by approximating the distribution with an ensemble of particles.
    For the description of the system see :py:func:`sample_trajectory`

    :param y:           observations
    :param n_particles: number of particles
    :param p0:          initial distribution of the dynamical system
    :param p:           transition model of the dynamical system
    :param nu:          observation model of the dynamical system

    :return: estimated particle states at each step
    """
    n_steps = len(y) - 1

    u_en = np.zeros((n_steps + 1, n_particles))
    y_en = np.zeros((n_steps + 1, n_particles))
    u_en_forecast = np.zeros((n_steps + 1, n_particles))

    for j in range(0, n_steps + 1):
        # prediction step
        if j == 0:
            u_en[0] = p0.rvs(size=(n_particles,))
            continue
        else:
            u_en_forecast[j] = p(u_en[j - 1]).rvs(size=(n_particles,))

        # analysis step
        y_en[j] = nu(u_en_forecast[j]).rvs(size=(n_particles,))

        # Y = y_en[j] @ S
        # U = u_en_forecast[j] @ S
        # K * Y * Y.T = U * Y.T
        cov = np.cov(u_en_forecast[j], y_en[j], ddof=0)
        K = cov[0, 1] / cov[1, 1]

        u_en[j] = u_en_forecast[j] + K * (y[j] - y_en[j])

    return u_en


def bootstrap_pf(y, n_particles, p0, p, nu, no_resampling=False):
    """
    Estimates the state of a dynamical system by approximating the distribution with an ensemble of  weighted particles.
    For the description of the system see :py:func:`sample_trajectory`

    :param y:           observations
    :param n_particles: number of particles
    :param p0:          initial distribution of the dynamical system
    :param p:           transition model of the dynamical system
    :param nu:          observation model of the dynamical system

    :return: estimated particle states with their weights and estimated sample size at each step
    """
    n_steps = len(y) - 1

    u_bs = np.zeros((n_steps + 1, n_particles))
    w_bs = np.zeros((n_steps + 1, n_particles))
    u_bs_resample = np.zeros((n_steps + 1, n_particles))
    ess = np.zeros((n_steps + 1,))

    for i in range(1, n_steps + 1):
        # resampling step
        if i == 1:
            u_bs_resample[0] = p0.rvs(size=(n_particles,))
        else:
            if not no_resampling:
                u_bs_resample[i - 1] = np.random.choice(u_bs[i - 1], n_particles, p=w_bs[i - 1])
            else:
                u_bs_resample[i - 1] = u_bs[i - 1]

        # prediction step
        u_bs[i] = p(u_bs_resample[i - 1]).rvs(size=(n_particles,))

        # analysis step
        w_bs[i] = nu(u_bs[i]).pdf(y[i])
        w_bs[i] = w_bs[i] / w_bs[i].sum()

        ess[i] = 1 / n_particles / np.sum(w_bs[i] ** 2)

    return u_bs, w_bs, ess


def log_likelihood(u, y, nu):
    prob_y = nu(u).pdf(y[:, np.newaxis]).mean(axis=1)
    return np.sum(np.log(prob_y))