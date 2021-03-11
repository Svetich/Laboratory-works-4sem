import numpy as np
import matplotlib.pyplot as plt
import help_laba

from math import sqrt

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'


def diffraction_picture():
    print('Дифракционная картина')
    nu_work = 1.2 * 10 ** 6  # 7 полос
    sigma_nu = 0.01
    f = 30 * 0.01
    sigma_f = 0.01
    Lambda = 2 * 0.65 * 10 ** (-3)
    sigma_Lambda = sqrt(2) * 0.01 * 10 ** (-3)
    velocity = Lambda * nu_work
    sigma_velocity = velocity * sqrt((sigma_nu/nu_work) ** 2 + (sigma_Lambda/Lambda) ** 2)

    print('Длина УЗ волны = ' + str(Lambda) + '+-' + str(sigma_Lambda))
    print('Скорость звука (оценка на месте) = ' + str(velocity) + '+-' + str(sigma_velocity))

    nu_1 = nu_work
    m_1 = np.array([-3, -2, -1, 0, 1, 2, 3])
    xm_1 = np.array([-25, -8, -3, 18, 37, 52, 68]) * 0.01
    b1, a1, sigma_b1, sigma_a1 = help_laba.LSM(m_1, xm_1)
    print('Коэф. b 1 = ' + str(b1) + '+-' + str(sigma_b1))
    print('Коэф. a 1 = ' + str(a1) + '+-' + str(sigma_a1))

    nu_2 = 3.00 * 10 ** 6
    m_2 = np.array([-1, 0, 1])
    xm_2 = np.array([-18, 21, 63]) * 0.01
    b2, a2, sigma_b2, sigma_a2 = help_laba.LSM(m_2, xm_2)
    print('Коэф. b 2 = ' + str(b2) + '+-' + str(sigma_b2))
    print('Коэф. a 2 = ' + str(a2) + '+-' + str(sigma_a2))

    nu_3 = 2 * 10 ** 6
    m_3 = np.array([-1, 0, 1])
    xm_3 = np.array([-9, 25, 48]) * 0.01
    b3, a3, sigma_b3, sigma_a3 = help_laba.LSM(m_3, xm_3)
    print('Коэф. b 3 = ' + str(b3) + '+-' + str(sigma_b3))
    print('Коэф. a 3 = ' + str(a3) + '+-' + str(sigma_a3))

    nu_4 = 10 ** 6
    m_4 = np.array([-2, -1, 0, 1, 2])
    xm_4 = np.array([-7, 7, 21, 35, 48]) * 0.01
    b4, a4, sigma_b4, sigma_a4 = help_laba.LSM(m_4, xm_4)
    print('Коэф. b 4 = ' + str(b4) + '+-' + str(sigma_b4))
    print('Коэф. a 4 = ' + str(a4) + '+-' + str(sigma_a4))
    plt.figure(figsize=(10, 7))
    plt.plot(m_1, xm_1, 'o', label='Измерения для 1,2 МГЦ', markersize=4, color='navy')
    # plt.errorbar(m_1, xm_1, yerr=np.linspace(0.01, 0.01, len(m_1)), fmt='o', color='navy', ecolor='navy')
    plt.plot(m_1, m_1 * b1 + a1, '-', label='Линейная аппрокцимация $y=b_{1} x + a_{1}$ \n' +
            '$b_{1}=0,157 \pm 0,006$ мм \n' + '$a_{1}=0,20 \pm 0,04$ мм', color='c')
    plt.plot(m_2, xm_2, 'o', label='Измерения для 3,002 МГЦ', markersize=4, color='darkgreen')
    # plt.errorbar(m_2, xm_2, yerr=np.linspace(0.01, 0.01, len(m_2)), fmt='o', color='darkgreen', ecolor='darkgreen')
    plt.plot(m_2, m_2 * b2 + a2, '-', label='Линейная аппрокцимация $y=b_{2} x + a_{2}$ \n' +
            '$b_{2}=0,405 \pm 0,005$ мм \n' + '$a_{2}=0,220 \pm 0,012$ мм', color='lawngreen')
    plt.plot(m_3, xm_3, 'o', label='Измерения для 2 МГЦ', markersize=4, color='darkorange')
    # plt.errorbar(m_3, xm_3, yerr=np.linspace(0.01, 0.01, len(m_3)), fmt='o', color='darkorange', ecolor='darkorange')
    plt.plot(m_3, m_3 * b3 + a3, '-', label='Линейная аппрокцимация $y=b_{3} x + a_{3}$ \n' +
            '$b_{3}=0,285 \pm 0,018$ мм \n' + '$a_{3}=0,21 \pm 0,06$ мм', color='yellow')
    plt.plot(m_4, xm_4, 'o', label='Измерения для 1 МГЦ', markersize=4, color='red')
    # plt.errorbar(m_4, xm_4, yerr=np.linspace(0.01, 0.01, len(m_4)), fmt='o', color='red', ecolor='red')
    plt.plot(m_4, m_4 * b4 + a4, '-', label='Линейная аппрокцимация $y=b_{4} x + a_{4}$ \n' +
            '$b_{4}=0,1380 \pm 0,0009$ мм \n' + '$a_{4}=0,208 \pm 0,006$ мм', color='lightcoral')
    plt.xlabel('m', size=18)
    plt.ylabel(r'$x_{m}$, мм', size=18)
    plt.yticks(np.arange(-0.2, 0.9, step=0.1), size=16)
    plt.xticks(np.arange(-3, 3, step=1), size=16)
    plt.legend(prop={'size':10})
    plt.savefig(r'xm_m.png')
    # plt.show()

    lambda_red = 6400 * 10 ** (-10)
    sigma_lambda_red = 200 * 10 ** (-10)

    lambda_1 = f * lambda_red / b1 * 10 ** (3)
    sigma_l_1 = lambda_1 * sqrt((sigma_f/f) ** 2 + (sigma_lambda_red/lambda_red) ** 2 + (sigma_b1/b1) ** 2)
    lambda_2 = f * lambda_red / b2 * 10 ** 3
    sigma_l_2 = lambda_2 * sqrt((sigma_f / f) ** 2 + (sigma_lambda_red / lambda_red) ** 2 + (sigma_b2 / b2) ** 2)
    lambda_3 = f * lambda_red / b3 * 10 ** 3
    sigma_l_3 = lambda_3 * sqrt((sigma_f / f) ** 2 + (sigma_lambda_red / lambda_red) ** 2 + (sigma_b3 / b3) ** 2)
    lambda_4 = f * lambda_red / b4 * 10 ** 3
    sigma_l_4 = lambda_4 * sqrt((sigma_f / f) ** 2 + (sigma_lambda_red / lambda_red) ** 2 + (sigma_b4 / b4) ** 2)

    lambda_arr = np.array([lambda_1, lambda_2, lambda_3, lambda_4])
    sigma_lambda_arr = np.array([sigma_l_1, sigma_l_2, sigma_l_3, sigma_l_4])
    nu_arr = np.array([nu_1, nu_2, nu_3, nu_4])

    velocity_arr = lambda_arr * nu_arr
    sigma_velocity_arr = velocity_arr * np.sqrt(np.power(sigma_lambda_arr / lambda_arr, 2) + np.power(sigma_nu / nu_arr, 2))

    print('Lambda = ' + str(lambda_arr))
    print('Sigma Lambda = ' + str(sigma_lambda_arr))
    print('Velocity array = ' + str(velocity_arr))
    print('Sigma velocity = ' + str(sigma_velocity_arr))

    velocity_mean, sigma_vel_mean = help_laba.mean_square_error(velocity_arr)

    print('Mean velocity = ' + str(np.mean(velocity_arr)) + '+-' + str(sigma_vel_mean))


def dark_field_method():
    print('Тёмное поле')
    one_div = 1 / 18 * 10 ** (-3)
    nu = np.array([1.07, 1.26, 1.19, 1.02]) * 10 ** 6
    sigma_nu = 0.01
    divs = np.array([200, 120, 120, 160]) * one_div
    gaps = np.array([10, 7, 7, 8])
    Lambda = divs / gaps
    sigma_Lambda = Lambda * np.sqrt(np.power(one_div / divs, 2) + np.power(1/gaps, 2))
    velocity = Lambda * nu
    sigma_velocity = np.sqrt(np.power(sigma_Lambda / Lambda, 2) + np.power(sigma_nu / nu, 2))
    print('Lambda = ' + str(Lambda) + '+-' + str(sigma_Lambda))
    print('Скорость = ' + str(velocity))
    print('Sigma = ' + str(sigma_velocity))

    vel_mean, sigma_vel_mean = help_laba.mean_square_error(velocity)

    print('Средняя скорость = ' + str(vel_mean) + '+-' + str(sigma_vel_mean))

diffraction_picture()
dark_field_method()