import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

data_ort = pd.read_csv('lr_ortogonale.csv') # importa .csv come DataFrame
x_ort = data_ort['x'].to_numpy() # trasforma colonna di DataFrame in array di numpy
y_ort = data_ort['y'].to_numpy()
s_y_ort = data_ort['s_y'].to_numpy()

lr_ort = linear_regression(x_ort, y_ort, s_y_ort)

x1 = np.linspace(min(x_ort)-1, max(x_ort)+1, 1000)
def y1(x1):
    return lr_ort['slope'] * x1 + lr_ort['inter']

plt.errorbar(x_ort, y_ort, s_y_ort, color="#A50021", ecolor="#FF4367", fmt='.', label="data")
plt.plot(x1, y1(x1), color="#0047A6", label="y = Ax + B")
plt.grid(True)
plt.xlabel("2ΔV [V]")
plt.ylabel("r²B² [N²/A²]")
plt.title("Regressione lineare per la configurazione ortogonale")
plt.legend()
plt.savefig("graph_ort.png", dpi=240)
plt.close()

data_par = pd.read_csv('lr_parallelo.csv') # importa .csv come DataFrame
x_par = data_par['x'].to_numpy() # trasforma colonna di DataFrame in array di numpy
y_par = data_par['y'].to_numpy()
s_y_par = data_par['s_y'].to_numpy()

lr_par = linear_regression(x_par, y_par, s_y_par)

x2 = np.linspace(min(x_par)-1, max(x_par)+1, 1000)
def y2(x2):
    return lr_par['slope'] * x2 + lr_par['inter']

plt.errorbar(x_par, y_par, s_y_par, color="#A50021", ecolor="#FF4367", fmt='.', label="data")
plt.plot(x2, y2(x2), color="#0047A6", label="y = Ax + B")
plt.grid(True)
plt.xlabel("2ΔV [V]")
plt.ylabel("r²B² [N²/A²]")
plt.title("Regressione lineare per la configurazione parallela")
plt.legend()
plt.savefig("graph_par.png", dpi=240)
plt.close()


data_apar = pd.read_csv('lr_antiparallelo.csv') # importa .csv come DataFrame
x_apar = data_apar['x'].to_numpy() # trasforma colonna di DataFrame in array di numpy
y_apar = data_apar['y'].to_numpy()
s_y_apar = data_apar['s_y'].to_numpy()

lr_apar = linear_regression(x_apar, y_apar, s_y_apar)

x3 = np.linspace(min(x_apar)-1, max(x_apar)+1, 1000)
def y3(x):
    return lr_apar['slope'] * x + lr_apar['inter']

plt.errorbar(x_apar, y_apar, s_y_apar, color="#A50021", ecolor="#FF4367", fmt='.', label="data")
plt.plot(x3, y3(x3), color="#0047A6", label="y = Ax + B")
plt.grid(True)
plt.xlabel("2ΔV [V]")
plt.ylabel("r²B² [N²/A²]")
plt.title("Regressione lineare per la configurazione antiparallela")
plt.legend()
plt.savefig("graph_apar.png", dpi=240)
plt.close()