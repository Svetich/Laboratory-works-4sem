import numpy as np
import matplotlib.pyplot as plt

from math import sqrt

import help_laba

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

def thin_lens():
    print('1. Проверка формулы линзы')
    L = np.array([103.5, 77.7, 110.1, 120.5, 58.7, 46.5])
    s_small = np.array([92, 65.3, 97.5, 108.5, 45.3, 30])
    s_big = np.array([11, 11, 10.6, 10.5, 17.3, 20.3])
    s = s_small

    sigma_s = 0.5
    sigma_L = 0.5
    f = 1 / (1/s + 1/(L - s))
    sigma_L_s = sqrt(2) * 0.5
    sigma_f_random = np.std(f)
    sigma_f_system = sqrt(sigma_s ** 2 + sigma_L_s ** 2)
    sigma_f = sqrt(sigma_f_system ** 2 + sigma_f_random ** 2)

    print('F = ' + str(f))
    print('F mean = ' + str(np.mean(f)) + '+-' + str(sigma_f))

    y = L * s - s ** 2
    x = L

    sigma_s2 = s ** 2 * 2 * sigma_s / s
    sigma_Ls = L * s * np.sqrt(np.power(sigma_L / L, 2) + np.power(sigma_s/s, 2))

    sigma_y = np.sqrt(np.power(sigma_s2, 2) + np.power(sigma_Ls, 2))
    print('Погрешности y = ' + str(sigma_y))

    a, b, sigma_a, sigma_b = help_laba.chi_sq(x, y, sigma_y)
    print('Chi_square ax+b, a=', str(a), '+-', str(sigma_a), ' b=', str(b), '+-', str(sigma_b))
    a1, b1, sigma_a1, sigma_b1 = help_laba.LSM(x, y)
    print('LSM ax+b, a=', str(a1), '+-', str(sigma_a1), ' b=', str(b1), '+-', str(sigma_b1))

    f_gr = a
    print('F graphic = ' + str(f_gr) + '+-' + str(sigma_a))

    # plt.plot(x, a1 * x + b1, color='red')
    plt.plot(x, y, 'o', label='Измерения', color='blue', markersize=4)
    plt.plot(x, a * x + b, '-', label='Линейная аппроксимация по методу $\chi^2$ \n' +
                                       'y = ax, где \n' +
                                       'a = 10,8 $ \pm 0,4$ см', color='darkcyan', linewidth=1)
    plt.errorbar(x, y, yerr=sigma_y, xerr=np.linspace(sigma_s, sigma_s, len(x)),
                 fmt='o', color='blue', markersize=4, ecolor='blue', elinewidth=1)
    plt.legend()
    plt.grid()
    plt.ylabel(r'$s(L - s)$, см$^2$', size=16)
    plt.xlabel(r'$L$, см', size=16)
    plt.savefig('thin_lens.png')
    plt.show()
    plt.close()


def bessel():
    print('2. Фокусное расстояние и оптический интервал по методу Бесселя')
    print('2, 3, 1')

    L_arr = np.array([68.5, 101.2, 83])
    s_big_arr = np.array([12.3, 26.7, 9.5])
    s_small_arr = np.array([56.7, 74, 74])
    l_arr = s_small_arr - s_big_arr

    f = (np.power(L_arr, 2) - np.power(l_arr, 2)) / (4 * L_arr)

    sigma_L = sqrt(2) * 0.05
    sigma_l = sqrt(2) * 0.05

    sigma_L2_arr = np.power(L_arr, 2) * 2 * sigma_L / L_arr
    sigma_l2_arr = np.power(l_arr, 2) * 2 * sigma_l / l_arr
    sigma_L2_l2 = np.sqrt(np.power(sigma_l2_arr, 2) + np.power(sigma_L2_arr, 2))
    sigma_f = f * np.sqrt(np.power(sigma_L2_l2 / (np.power(L_arr, 2) - np.power(l_arr, 2)), 2) + np.power(sigma_L / L_arr, 2))
    print('F = ' + str(f) + '+-' + str(sigma_f))


    L = np.array([103.5, 77.7, 110.1, 120.5, 58.7, 46.5])
    s_small = np.array([92, 65.3, 97.5, 108.5, 45.3, 30])
    s_big = np.array([11, 11, 10.6, 10.5, 17.3, 20.3])

    s = s_small

    sigma_s = 0.5
    sigma_L = 0.5

    l = s_small - s_big
    sigma_l = sqrt(2) * sigma_s

    y = np.power(L, 2) - np.power(l, 2)
    sigma_L2 = np.power(L, 2) * 2 * sigma_L / L
    sigma_l2 = np.power(l, 2) * 2 * sigma_l / l
    sigma_y = np.sqrt(np.power(sigma_L2, 2) + np.power(sigma_l2, 2))

    print('Погрешности y = ' + str(sigma_y))

    a, b, sigma_a, sigma_b = help_laba.chi_sq(L, y, sigma_y)
    print('Chi_square ax+b, a=', str(a), '+-', str(sigma_a), ' b=', str(b), '+-', str(sigma_b))


    plt.plot(L, y, 'o', label='Измерения', color='blue', markersize=3)
    plt.plot(L, a * L + b, '-', label='Линейная аппроксимация по методу $\chi^2$ \n' +
                                       'y = ax + b, где \n' +
                                       'a = 37,4 $\pm$ 1,9 см \n' +
                                       'b = 352 $\pm$ 127 см$^2$', color='darkorange', linewidth=1)
    plt.errorbar(L, y, yerr=sigma_y, xerr=np.linspace(sigma_L, sigma_L, len(y)),
                 fmt='o', color='blue', markersize=3, ecolor='blue', elinewidth=1)
    plt.xlabel(r'$L$, см', size=16)
    plt.ylabel(r'$L^2 - l^2$, см$^2$', size=16)
    plt.legend()
    plt.grid()
    plt.savefig('bessel.png')
    plt.show()
    plt.close()

    delta = -7.793258581044665
    delta = (a - sqrt(a**2 + 4*b)) / 2
    f = 13.251491077944321
    f = (a - 2 * delta) / 4

    sigma_a2 = a * sqrt(2) * sigma_a
    sigma_sum = sqrt(sigma_a2 ** 2 + sigma_b ** 2)
    sigma_sqrt = sqrt(a**2 + 4 * b) / sqrt(2) * sigma_sum / (a**2 + 4 * b)
    sigma_delta = sqrt(sigma_a ** 2 + sigma_sqrt ** 2)
    sigma_f = sqrt(sigma_a ** 2 + sigma_delta ** 2)
    print('delta = ' + str(delta) + '+-' + str(sigma_delta))
    print('f = ' + str(f) + '+-' + str(sigma_f))



def spyglass_scat():
    print('3. Зрительная труба + РЛ')
    f1 = 9.5
    f2 = 11.3
    f3 = 19
    f4 = 32
    f5 = 49


def Kepler():
    print('4. Труба Кеплера')
    f2 = 11.3  # окуляр
    f3 = 19  # коллиматор
    f4 = 32  # объектив
    sigma_f = sqrt(2) * 0.5
    l = 42.5  # между объективом и окуляром
    sigma_l = sqrt(2) * 0.5
    f4_f3 = f2 + f4
    sigma_f4_f3 = sqrt(2) * sigma_f

    print('Измеренная сумма фокусов = ' + str(l) + '+-' + str(sigma_l))
    print('Сумма фокусов = ' + str(f4_f3) + '+-' + str(sigma_f4_f3))

    N1 = f4 / f2
    sigma_N1 = N1 * sqrt((sigma_f / f4) ** 2 + (sigma_f/f2) ** 2)
    print('Увеличение (фокусы) N = ' + str(N1) + '+-' + str(sigma_N1))

    one_bef = 9
    one_after = 24

    N2 = one_after / one_bef
    sigma_N2 = N2 * sqrt((1/one_after) ** 2 + (1/one_bef) ** 2)
    print('Увеличение (деления) N = ' + str(N2) + '+-' + str(sigma_N2))

    D_real = 3.4
    D_im = 1.1
    sigma_D = 0.1

    N3 = D_real / D_im
    sigma_N3 = N3 * sqrt((sigma_D / D_real) ** 2 + (sigma_D / D_im) ** 2)
    print('Увеличение (диаметры) N = ' + str(N3) + '+-' + str(sigma_N3))




thin_lens()
bessel()
Kepler()