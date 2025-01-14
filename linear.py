import scipy.stats as stats


def initial_distribution(mu0, sigma0):
    """
    u_0 ~ N(mu0, sigma0)

    :param mu0:     mean
    :param sigma0:  variance

    :return: initial state generator
    """
    return stats.norm(mu0, sigma0 ** 0.5)


def transition_model(a, m, c):
    """
    u_j+1 ~ N(a * u_j + m, c)

    :param a: linear coefficient
    :param m: noise mean
    :param c: noise variance

    :return: transition model, which accepts state and returns next state generator
    """
    return lambda u: stats.norm(a * u + m, c ** 0.5)


def observation_model(h, gamma):
    """
    y_j+1 ~ N(h * u_j+1, gamma)

    :param h:     linear coefficient
    :param gamma: noise variance

    :return: observation model, which accepts a state and returns observation generator
    """
    return lambda u: stats.norm(h * u, gamma ** 0.5)
