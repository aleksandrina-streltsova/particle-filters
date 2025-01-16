import numpy as np
import scipy.stats as stats


def initial_distribution(sigma2):
    """
    u_0 ~ N(0, sigma0)

    :param sigma2: variance

    :return: initial state generator
    """
    return stats.norm(0, sigma2 ** 0.5)


def transition_model(phi, sigma2):
    """
    u_j+1 ~ N(phi * u_j, sigma^2)

    :param phi:   linear coefficient
    :param sigma2: variance

    :return: transition model, which accepts state and returns next state generator
    """
    return lambda u: stats.norm(phi * u, sigma2 ** 0.5)


def observation_model(beta2):
    """
    y_j+1 ~ N(0, beta^2 * exp(u_j+1))

    :param beta2: variance coefficient

    :return: observation model, which accepts a state and returns observation generator
    """
    return lambda u: stats.norm(0, (beta2 * np.exp(u)) ** 0.5)
