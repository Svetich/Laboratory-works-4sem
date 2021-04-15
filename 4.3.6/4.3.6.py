import numpy as np
import matplotlib.pyplot as plt
import help_laba

from math import sqrt

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'


def A_one():
    print('А. Исследование двумерных решёток')
    print('1. Определение периода решёток по их пространственному спектру')
    print('1, 2, 3, 4, 5')

    X = np.array([27.8, 27.8, 29, 13, 14])  # см
    sigma_rul = 0.1
    m = np.array([9, 13, 26, 23, 33]) - 1
    L = 4.5 + 100.5 + 23.1
    sigma_L = sqrt(3) * sigma_rul
    lambda_wave = 532 * 10 ** (-7)  # см
    x = X / m
    sigma_x = x * np.sqrt(np.power(sigma_rul / X, 2))
    d = L * lambda_wave / x * 10 ** (-2)
    sigma_d = d * np.sqrt(np.power(sigma_L / L, 2) + np.power(sigma_x/x, 2))
    print('d = ' + str(d * 10 ** 6) + '+-' + str(sigma_d * 10 ** 6) + ' мк')

    return d


def A_two():
    print('2. Определение периода решёток по изображению, увеличенному с помощью линзы')
    print('3, 4, 5')
    D = np.array([1, 3, 4])  # мм
    a = 5.5 * 10
    b = (4.5 + 100.5 + 19.9) * 10
    sigma_rul = 1
    sigma_b = sqrt(3) * sigma_rul
    d = D * a / b * 10 ** (-3)
    sigma_d = d * np.sqrt(np.power(sigma_rul/D, 2) + (sigma_rul/a) ** 2 + (sigma_b/b) ** 2)
    print('d = ' + str(d * 10 ** 6) + '+-' + str(sigma_d * 10 ** 6) + ' мк')

    return d


def A_three():
    print('3. Исследование эффекта саморепродукции с помощью сеток')
    print(3, 4, 5)
    n3 = np.array([0, 1, 2, 3, 4, 5])
    n4 = np.array([0, 1, 2, 3, 4, 5])
    n5 = np.array([0, 1, 2])

    z3 = np.array([6.745, 6.075, 5.40, 4.72, 4.04, 3.455])
    z4 = np.array([6.825, 6.60, 6.010, 3.91, 3.05, 1.31])
    z5 = np.array([6.825, 4.965, 1.91])

    a3, b3, sigma_a3, sigma_b3 = help_laba.LSM(n3, z3)
    a4, b4, sigma_a4, sigma_b4 = help_laba.LSM(n4, z4)
    a5, b5, sigma_a5, sigma_b5 = help_laba.LSM(n5, z5)

    print('Коэф. наклона a = ' + str(np.array([a3, a4, a5])) + '+-' + str(np.array([sigma_a3, sigma_a4, sigma_a5])))
    print('b = ' + str(np.array([b3, b4, b5])) + '+-' + str(np.array([sigma_b3, sigma_b4, sigma_b5])))

    lambda_wave = 532 * 10 ** (-9)
    d = np.sqrt(lambda_wave * np.array([abs(a3), abs(a4), abs(a5)]) * 10 ** (-2)/2)
    sigma_d = d * np.array([sigma_a3, sigma_a4, sigma_a5]) / (np.array([abs(a3), abs(a4), abs(a5)]) * sqrt(2))

    print('d = ' + str(d * 10 ** 6) + '+-' + str(sigma_d * 10 ** 6) + ' мк')


    plt.errorbar(n3, z3, yerr=np.linspace(0.01, 0.01, len(z3)), fmt='.', color='deepskyblue', ecolor='deepskyblue', elinewidth=0.8)
    plt.plot(n3, z3, '.', label='Измерения для сетки 3', color='deepskyblue')
    plt.plot(n3, n3 * a3 + b3, '-', label='Линейная аппроксимация по МНК \n' +
                                       'y = ax + b, где \n' +
                                       'a = -0,664 $\pm$ 0,007 мм \n' +
                                       'b = 6,732 $\pm$ 0,022 мм', color='deepskyblue', linewidth=0.5)
    plt.legend(prop={'size': 11})
    plt.xlabel('n', size=16)
    plt.ylabel(r'z, мм', size=16)
    plt.savefig('Саморепродукция 3.png')
    # plt.show()
    plt.close()

    plt.errorbar(n4, z4, yerr=np.linspace(0.01, 0.01, len(z4)), fmt='.', color='coral', ecolor='coral', elinewidth=0.8)
    plt.plot(n4, z4, '.', label='Измерения для сетки 4', color='coral')
    plt.plot(n4, n4 * a4 + b4, '-', label='Линейная аппроксимация по МНК \n' +
                                       'y = ax + b, где \n' +
                                       'a = -1,15 $\pm$ 0,14 мм \n' +
                                       'b = 7,5 $\pm 0,4$ мм', color='coral', linewidth=0.5)
    plt.legend(prop={'size': 11})
    plt.xlabel('n', size=16)
    plt.ylabel(r'z, мм', size=16)
    plt.savefig('Саморепродукция 4.png')
    # plt.show()
    plt.close()

    plt.errorbar(n5, z5, yerr=np.linspace(0.01, 0.01, len(z5)), fmt='.', color='m', ecolor='m', elinewidth=0.8)
    plt.plot(n5, z5, '.', label='Измерения для сетки 5', color='m')
    plt.plot(n5, n5 * a5 + b5, '-', label='Линейная аппроксимация по МНК \n' +
                                       'y = ax + b, где \n' +
                                       'a = -2,46 $\pm$ 0,34 мм \n' +
                                       'b = 7,0 $\pm$ 0,4 мм', color='m', linewidth=0.5)
    plt.legend(prop={'size':11})
    plt.xlabel('n', size=16)
    plt.ylabel(r'z, мм', size=16)
    plt.savefig('Саморепродукция 5.png')
    # plt.show()
    plt.close()

    return d


def world_grid(number, X, m, D, n, z):
    print('Б. Исследование решёток миры')
    print(number)
    a = 70
    b = 1208  # мм
    L = a + b
    sigma_rul = 1
    sigma_L = sqrt(2) * sigma_rul
    lambda_wave = 532 * 10 ** (-9)
    x = X / m
    sigma_x = x * np.sqrt(np.power(sigma_rul / X, 2))
    d1 = L * lambda_wave / x
    sigma_d1 = d1 * np.sqrt(np.power(sigma_L / L, 2) + np.power(sigma_x / x, 2))

    print('d1 = ' + str(d1 * 10 ** 6) + '+-' + str(sigma_d1 * 10 ** 6) + ' мк')

    d2 = D * a / b * 10 ** (-3)
    sigma_d2 = d2 * np.sqrt(np.power(sigma_rul / D, 2) + (sigma_rul / a) ** 2 + (sigma_rul / b) ** 2)
    print('d2 = ' + str(d2 * 10 ** 6) + '+-' + str(sigma_d2 * 10 ** 6) + ' мк')

    a, b, sigma_a, sigma_b = help_laba.LSM(n, z)

    print('Коэф. наклона = ' + str(a) + '+-' + str(sigma_a))
    print('a = ' + str(b) + '+-' + str(sigma_b))

    lambda_wave = 532 * 10 ** (-9)
    d3 = np.sqrt((lambda_wave * a) * 10 ** (-3)/2)
    sigma_d3 = d3 * sigma_a / a * sqrt(2)

    print('d3 = ' + str(d3 * 10 ** 6) + '+-' + str(sigma_d3 * 10 ** 6) + ' мк')

    plt.errorbar(n, z, yerr=np.linspace(0.01, 0.01, len(z)), fmt='.', color='red', ecolor='red', elinewidth=0.8)
    plt.plot(n, z, '.', label='Измерения для миры 20', color='red')
    plt.plot(n, n * a + b, '-', label='Линейная аппроксимация по МНК \n' +
                                          'y = ax + b, где \n' +
                                          'a = 5,84 $\pm$ 0,29 мм \n' +
                                          'b = 25,89 $\pm$ 0,13 мм', color='red', linewidth=0.5)
    plt.legend(prop={'size': 11})
    plt.xlabel('n', size=16)
    plt.ylabel(r'z, мм', size=16)
    # plt.savefig('Саморепродукция миры' + number +'.png')
    # plt.show()
    plt.close()




d1 = A_one()
d2 = A_two()
d3 = A_three()

world_grid('25', 165, 9, 18/17, np.array([0, 1, 2, 3, 4, 5, 6]), np.array([17, 20, 22.7, 27, 32, 39, 42.45]))
world_grid('20', 160, 10, 20/14, np.array([0, 1, 2, 3]), np.array([26, 31.6, 37.45, 43.5]))

