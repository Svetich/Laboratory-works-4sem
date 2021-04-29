import numpy as np
import matplotlib.pyplot as plt
import help_laba

from math import sqrt

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'


def rings():
    m = np.array([1, 2, 3, 4, 5])
    r = np.array([2.5, 3.5, 4.5, 5.3, 6])
    L = 78.5
    n_o = 2.29
    l = 2.6
    lambda_wave = 0.63 * 10 ** (-4)
    y_error = 2 * (np.linspace(0.1, 0.1, len(m)) / r)
    a, b, sigma_a, sigma_b = help_laba.chi_sq(m, r ** 2, y_error)

    print('a = ' + str(a) + '+-' + str(sigma_a))
    print('b = ' + str(b) + '+-' + str(sigma_b))

    plt.plot(m, r ** 2, '.', label='Измерения', color='orange', markersize=10)
    plt.plot(m, m * a + b, '-', label='Аппроксимация по $\chi^2$ \n' +
                                      'a = 7,70 $\pm$ 0,17 см$^2$ \n' +
                                      'b = -2,6 $\pm$ 0,7 см$^2$', color='blue', linewidth=0.8)
    plt.errorbar(m, r ** 2, yerr=y_error, fmt='.', color='orange', markersize=10, ecolor='orange')
    plt.xlabel('$m$', size=16)
    plt.ylabel(r'$r^2$, см$^2$', size=16)
    plt.legend()
    plt.savefig(r'r2_m.png')
    plt.show()

    n_o_n_e = lambda_wave * (n_o * L) ** 2 / (l * a)
    sigma_o_e = n_o_n_e * sqrt((2 * 0.1/L) ** 2 + (0.1/l) ** 2 + (sigma_a / a) ** 2)
    print('n_o - n_e = ' + str(n_o_n_e) + '+-' + str(sigma_o_e))


rings()