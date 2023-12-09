import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import t

# functions which calculates the mean of the elements of an array
def mean(array): 
    sum = 0
    for i in range(len(array)): sum += array[i]
    return (sum / len(array))

# functions which calculates the standard deviation of the elements of an array
def dev_st(array): 
    sum = 0
    mean_a = mean(array)
    for i in range(len(array)):
        sum += (array[i] - mean_a)**2
        print((array[i] - mean_a)**2)

    return np.sqrt(sum / ((len(array) - 1)*len(array)))

# function for t-test
def t_student (x, mu, s, nu):
    t_st = abs(x - mu) / s
    return 200*t.sf(t_st, nu)

def t_student_cross (x_1, s_1, n_1, x_2, s_2, n_2):
    s = np.sqrt(((n_1 - 1) * s_1**2 + (n_2 - 1) * s_2**2) / (n_1 + n_2 - 2))
    t_st = abs(x_1 - x_2) / (s * np.sqrt((1/n_1) + (1/n_2)))
    return 200*t.sf(t_st, n_1 + n_2 - 2)

def linear_regression(x, y, y_err):
    
    nu = len(x) - 2
    
    sum_p = 0
    sum_x = 0
    sum_xx = 0
    sum_y = 0
    sum_yy = 0
    sum_xy = 0
    
    for i in range(len(x)):
        
        sum_p += 1 / (y_err[i])**2
        sum_x += x[i] / (y_err[i])**2
        sum_xx += (x[i])**2 / (y_err[i])**2
        sum_y += y[i] / (y_err[i])**2
        sum_yy += (y[i])**2 / (y_err[i])**2
        sum_xy += x[i] * y[i] / (y_err[i])**2
    
    delta_x = sum_p * sum_xx - (sum_x)**2
    delta_y = sum_p * sum_yy - (sum_y)**2
    
    intercept = (sum_xx * sum_y - sum_x * sum_xy) / delta_x
    slope = (sum_p * sum_xy - sum_x * sum_y) / delta_x
    
    sigma_intercept = np.sqrt(sum_xx / delta_x)
    sigma_slope = np.sqrt(sum_p / delta_x)
    
    r = slope * np.sqrt(delta_x / delta_y)
    
    return {'slope': slope, 'inter': intercept, 's_slp': sigma_slope, 's_int': sigma_intercept, 'nu': nu, 'r': r}

##############

e_m_th = 1.75882001076e11

def data (path_data):

    data_csv = pd.read_csv(path_data, sep=',')

    x = data_csv['x']
    y = data_csv['y']
    s_y = data_csv['s_y']

    reg = linear_regression(x, y, s_y)

    e_m = 1. / reg['slope']
    e_m_err = reg['s_slp'] / reg['slope']**2

    return {'em' : e_m, 's' : e_m_err, 'n' : len(x)}

ort = data('ortogonale.csv')
par = data('parallelo.csv')
ant = data('antiparallelo.csv')

with open("t_student.txt", 'w') as text:

    print("Risultati delle regressioni:\n", file=text)

    print("Ortogonale:", file=text)
    print("e/m:", ort['em'], file=text)
    print("err:", ort['s'], file=text)

    print("\nParallelo:", file=text)
    print("e/m:", par['em'], file=text)
    print("err:", par['s'], file=text)

    print("\nAntiparallelo:", file=text)
    print("e/m:", ant['em'], file=text)
    print("err:", ant['s'], file=text)

    print("\n", file=text)

    print("\nTest di compatibilità col valore atteso (t di Student):\n", file=text)

    print("Ortogonale:", t_student(ort['em'], e_m_th, ort['s'], ort['n'] - 1), "%", file=text)

    print("\nParallelo:", t_student(par['em'], e_m_th, par['s'], par['n'] - 1), "%", file=text)

    print("\nAntiparallelo:", t_student(ant['em'], e_m_th, ant['s'], ant['n'] - 1), "%", file=text)

    print("\n", file=text)

    print("\nTest di cross-compatibility (t di Student):\n", file=text)

    print("Ortogonale - parallelo:", t_student_cross(ort['em'], ort['s'], ort['n'] - 1, par['em'], par['s'], par['n'] - 1), "%", file=text)

    print("\nParallelo - antiparallelo:", t_student_cross(par['em'], par['s'], par['n'] - 1, ant['em'], ant['s'], ant['n'] - 1), "%", file=text)

    print("\nAntiparallelo - ortogonale:", t_student_cross(ant['em'], ant['s'], ant['n'] - 1, ort['em'], ort['s'], ort['n'] - 1), "%", file=text)
