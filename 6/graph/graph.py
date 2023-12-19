import numpy as np
import matplotlib.pyplot as plt

x_r = [6.11716e+12, 6.01615e+12, 5.27089e+12, 4.14444e+12, 3.36129e+12, 3.00927e+12, 2.98902e+12]
y_r = [3.40249, 3.39589, 3.34170, 3.26662, 3.21960, 3.19958, 3.19888]
y_err_r = [0.00437, 0.00435, 0.00424, 0.00410, 0.00401, 0.00397, 0.00397]

def y(x):
    return 3.00161 + 6.514e-14 * x
x = np.linspace(2.7e12, 6.3e12, 100000)

plt.errorbar(x_r, y_r, y_err_r, label="data")
plt.plot(x, y(x), label="regression")
plt.grid(True)
plt.legend()
plt.xlabel("1/λ² [1/m^2]")
plt.ylabel("n²")
plt.savefig("graph.png", dpi=240)
