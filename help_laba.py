import numpy as np
from scipy.optimize import curve_fit


def LSM(x, y):
    function = lambda x, a, b: a*x  + b
    popt, pcov = curve_fit(function, xdata=x, ydata=y)

    sigma_a = np.sqrt(pcov[0,0])
    sigma_b = np.sqrt(pcov[1, 1])

    return popt[0], popt[1], sigma_a, sigma_b


def chi_sq(x, y, err):
    function = lambda x, a, b: a * x + b
    popt, pcov = curve_fit(function, xdata=x, ydata=y, sigma=err)

    sigma_a = np.sqrt(pcov[0, 0])
    sigma_b = np.sqrt(pcov[1, 1])

    return popt[0], popt[1], sigma_a, sigma_b

