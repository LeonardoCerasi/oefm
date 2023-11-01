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

##############

D = 13.28 # m
s_D = 0.01 # m
a = 0.474 # m
s_a = 0.001 # m
f = 0.252 # m

s_sist = (4 * D * f) * np.sqrt(((D + 2*a - 2*f) * s_D)**2 + (D * s_a)**2) / (D + a - f)**2 # m^2 

def data (path_data):

    data_csv = pd.read_csv(path_data, sep=',')

    d_omega = data_csv['deltaomega'].to_numpy() # rad/s
    d_delta = data_csv['deltadelta'].to_numpy() * 10**(-3) # m
    c = data_csv['c'].to_numpy() # m/s
    sist_err = (mean(d_omega) / mean(d_delta)) * s_sist # m/s

    scarti = data_csv['scarti'].to_numpy() # (m/s)^2
    sum = 0
    for i in range(len(scarti)): sum += scarti[i]
    stat_err = np.sqrt(sum / (len(scarti) * (len(scarti) - 1)))

    err = np.sqrt(sist_err**2 + stat_err**2)

    return {'o': d_omega, 'd': d_delta, 'c': c, 'sist': sist_err, 'stat': stat_err, 'e': err}

data_set = {"CCW_min_max" : data("../csv/CCW_min_max.csv"), "CCW_min_mid" : data("../csv/CCW_min_mid.csv"), "CW_CCW" : data("../csv/CW_CCW.csv"), "CW_min_max" : data("../csv/CW_min_max.csv"), "CW_min_mid" : data("../csv/CW_min_mid.csv")}

with open("t_student.txt", 'w') as text:

    print("Compatibility with expected value:\n", file=text)
    for key in data_set:
        print(key, ":", file=text)
        print(t_student(mean(data_set[key]['c']), 299792456.2, data_set[key]['e'], len(data_set[key]['c']) - 1), file=text)
        print('\n', file=text)

    print('\n\n\n', file=text)
    print("Cross-compatibility between datasets:\n", file=text)
    for key_1 in data_set:
        for key_2 in data_set:
            if (key_1 != key_2):
                print(key_1, "-", key_2, ":", file=text)
                print(t_student_cross(mean(data_set[key_1]['c']), data_set[key_1]['e'], len(data_set[key_1]['c']), mean(data_set[key_2]['c']), data_set[key_2]['e'], len(data_set[key_2]['c'])), file=text)
                print('\n', file=text)
    
    c = 0
    div = 0
    for key in data_set:
        if (key != "CW_min_max"):
            c += (mean(data_set[key]['c']))/(data_set[key]['e'])**2
            div += 1/(data_set[key]['e'])**2

    c /= div
    err = 1 / np.sqrt(div)
    print('\n\n\n', file=text)
    print("Final c value: ", c, file=text)
    print("Error on final c value: ", err, file=text)
    print("n_sigma: ", (299792456.2 - c) / err, file=text)
    print("n_sigma: ", (299792456.2 - 297900000) / 900000, file=text)
