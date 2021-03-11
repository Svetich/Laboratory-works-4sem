import numpy as np
import matplotlib.pyplot as plt
import help_laba

from math import sqrt

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'


def Fresnel_1():
    print('А. Френель на щели')
    n = np.array([5, 4, 3, 2, 1])
    all_z = np.array([45.5, 45.6, 45.7, 46, 46.6])
    z = (all_z - np.linspace(43, 43, 5))
    width = 0.02 * np.array([6, 7, 7, 4, 2])
    lambda_wave = 546.1 * 10 ** (-7)
    ksi_n = np.sqrt(z * n * lambda_wave)
    eps_z = sqrt(2) * 1 / z
    error_ksi = ksi_n * eps_z / 2

    k, b, sigma_k, sigma_b = help_laba.LSM(n, 2 * ksi_n)

    print('k = ' + str(k) + '+-' + str(sigma_k))
    print('m = ' + str(b) + '+-' + str(sigma_b))

    plt.figure(figsize=(10, 7))
    plt.errorbar(n, 2 * ksi_n, yerr=error_ksi, fmt='.', color='blue', ecolor='blue')
    plt.plot(n, 2 * ksi_n, '.', label='Измерения', color='blue')
    plt.plot(n, k * n + b, '-',
             label='Линейная аппроксимация $y=kx+a$' + '\n' + 'k = (5,99 $\pm$ 0,29) $\cdot$ 10$^{-3}$ см,' +
                   '\n' + 'a = (23,3 $\pm$ 0,4)$ \cdot$ 10$^{-3}$ см',
             color='c', linewidth=1)
    plt.plot(n, np.linspace(0.0316, 0.0316, 5), '-', label='b = 0.0316 см', color='crimson')
    plt.xlabel('n', size=16)
    plt.ylabel(r'2 $\xi$, см', size=16)
    plt.xticks(np.arange(1, 6, step=1), size=16)
    plt.yticks(np.arange(0.02, 0.07, step=0.01), size=16)
    plt.legend(prop={'size':14})
    plt.savefig('Френель1.png')
    plt.show()
    plt.close()
    ksi_n_mean, sigma = help_laba.mean_square_error(2 * ksi_n)
    print('Среднее значени = ' + str(ksi_n_mean) + '+-' + str(sigma))
    print('Расхождение = ' + str((b - ksi_n_mean)/(ksi_n_mean) * 100) + '%')


def Fraunhofer_1():
    print('Б. Фраунгофер на щели')
    m = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
    x_m = np.array([-64, -41, -27, -8, 12, 29, 53, 71, 87, 104, 127])
    lambda_wave = 546.1 * 10 ** (-9)
    focus2 = 0.125
    error_x_m = np.linspace(5, 5, 11)

    k, b, sigma_k, sigma_b = help_laba.LSM(m, x_m)

    print('k = ' + str(k) + '+-' + str(sigma_k) + ' мкм')
    print('b = ' + str(b) + '+-' + str(sigma_b) + ' мкм')

    plt.figure(figsize=(10, 7))
    plt.errorbar(m, x_m, yerr=error_x_m, fmt='.', color='blue', ecolor='blue')
    plt.plot(m, x_m, '.', label='Измерения', color='blue')
    plt.plot(m, k * m + b, '-',
             label='Линейная аппроксимация $y=kx+a$' + '\n' +
                   'k = 18,87 $\pm$ 0,19 мкм,' + '\n'
                   + 'a = 31,2 $\pm$ 0,6 мкм', color='c', linewidth=1)
    plt.xlabel('m', size=16)
    plt.ylabel(r'x$_{m}$, мкм', size=16)
    plt.yticks(np.arange(-80, 160, step=20), size=16)
    plt.xticks(np.arange(-5, 6, step=1), size=16)
    plt.legend(prop={'size':14})
    plt.savefig('Фраунгофер1.png')
    plt.show()
    plt.close()

    b_th = lambda_wave / (k * 10 ** (-6)) * focus2  #####

    sigma_b_th = b_th * sqrt((sigma_k / k) ** 2 + (0.001 / focus2) ** 2) ######

    print('b_th = ' + str(b_th * 10 ** 6) + '+-' + str(sigma_b_th * 10 ** 6) + ' мкм')

    b_exp = 440 * 10 ** (-6)

    eps = (b_exp - b_th) / b_exp

    print('Расхождение результатов = ' + str(eps * 100) + '%')


def Fraunhofer_2():
    print('В. Фраунгофер на двух щелях')
    D = 320 * 10 ** (-6)
    sigma_D = 0.02 * 10 ** (-3)
    f2 = 10.2 * 10 ** (-2)
    f1 = 12.5 * 10 ** (-2)
    lambda_wave = 546.1 * 10 ** (-9)
    delta_x = D / 5
    sigma_delta_x = delta_x * sigma_D / D
    print('delta x = ' + str(delta_x) + '+-' + str(sigma_delta_x))
    d = f2 * lambda_wave / delta_x
    sigma_d = d * sqrt((0.01/f2) ** 2 + (sigma_delta_x / delta_x) ** 2)
    print('d формула = ' + str(d) + '+-' + str(sigma_d))
    d_exp = 0.000890
    n_th = 2 * d / (491 * 10 ** (-6))
    sigma_n_th = n_th * sqrt((sigma_d / d) ** 2 + (1 / 295) ** 2)
    print('n th = ' + str(n_th) + '+-' + str(sigma_n_th))
    b0_exp = 0.772
    b0_th = lambda_wave * f1 / d
    sigma_b0_th = b0_th * sqrt((0.01 / f1) ** 2 + (sigma_d / d) ** 2)
    print('b0 th = ' + str(b0_th) + '+-' + str(sigma_b0_th))


def resolution_capacity():
    print('Г. Разрешающая способность')
    f1 = 12.5 * 10 ** (-2)
    lambda_wave = 546.1 * 10 ** (-9)
    d = 40 * 0.02 * 10 ** (-3)
    print('d = ' + str(d) + '+-' + str(0.02 * 10 ** (-3)))
    b1 = 8 * 0.02 * 10 ** (-3)
    b2 = 12 * 0.02 * 10 ** (-3)
    print('b1 = ' + str(b1))
    print('b2 = ' + str(b2))
    b0_th = lambda_wave * f1 / d
    sigma_b0_th = b0_th * sqrt((0.01 / f1) ** 2 + (0.02 * 10 ** (-3) / d) ** 2)
    print('b0 th = ' + str(b0_th) + '+-' + str(sigma_b0_th))

Fresnel_1()
Fraunhofer_1()
Fraunhofer_2()
resolution_capacity()