import numpy as np
import matplotlib.pyplot as plt
import help_laba

from math import sqrt

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'


def radius():
    R1 = np.array([0.31, 0.82, 1.19, 1.49, 1.69, 1.90, 2.10, 2.21, 2.42, 2.57, 2.72, 2.85, 2.91, 3.08, 3.22, 3.35, 3.45,
                   3.54, 3.65]) * 0.1
    R2 = (4 - np.array([3.58, 3.11, 2.78, 2.42, 2.27, 2.06, 1.82, 1.69, 1.52, 1.38, 1.22, 1.09,
                       0.98, 0.84, 0.78, 0.60, 0.49, 0.38, 0.29])) * 0.1  # мм
    N = np.linspace(1, 20, 19)
    R = (R1 + R2) / 2
    y_err = 2 * np.linspace(sqrt(2) * 0.1, sqrt(2) * 0.1, len(R)) / R
    a, b, sigma_a, sigma_b = help_laba.chi_sq(N, R ** 2, y_err)
    print('a = ' + str(a) + '+-' + str(sigma_a))
    print('b = ' + str(b) + '+-' + str(sigma_b))

    plt.plot(N, R ** 2, '.', label='Измерения', color='crimson', markersize=10)
    plt.plot(N, N * a + b, '-', label='Линейная аппроксимация по методу $\chi^2$ \n'
                                      'a = $(71,7 \pm 0,5) \cdot 10^{-4}$ мм $^2$ \n'
                                      'b = $(-79 \pm 7) \cdot 10^{-4}$ мм $^2$', color='crimson', linewidth=0.8)
    # plt.errorbar(N, R ** 2, yerr=y_err, fmt='.', color='crimson', markersize=10, ecolor='crimson', elinewidth=0.8)
    plt.xlabel('$m$', size=16)
    plt.ylabel(r'$r_m^2$, мм$^2$', size=16)
    plt.legend()
    plt.savefig(r'r2_m.png')
    plt.show()

    lambda_wave = 577 * 10 ** (-9)
    sigma_lambda = 10 * 10 ** (-9)

    Rl = a * 10 ** (-6) / lambda_wave
    sigma_Rl = Rl * sqrt((sigma_a / a) ** 2 + (sigma_lambda / lambda_wave) ** 2)
    print('R l = ' + str(Rl) + '+-' + str(sigma_Rl))



radius()