import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("charges2.csv")

charges = dataset['q'].to_numpy()
charges_err = dataset['err'].to_numpy()

def S(q):
    sum = 0
    for Q in charges:
        sum += (q - (Q / np.floor(0.5 + (Q / q))))**2
    return sum

q = np.linspace(1.4e-19, 1.8e-19, 10000)

plt.plot(q, S(q))
plt.grid(True)
plt.xlabel("q [C]")
plt.ylabel("S(q) [C²]")
plt.savefig("graph.png", dpi=240)

min = q[0]
for i in range(len(q)):
    if (S(q[i]) < S(min)):
        min = q[i]
print("min:", min)
err_stat = np.sqrt(S(min) / (len(charges) * (len(charges) - 1)))
print("err-stat:", err_stat)

err_sist = 0
for i in range(len(charges)):
    err_sist += (charges_err[i] / (len(charges) * np.floor(0.5 + (charges[i] / min))))**2
err_sist = np.sqrt(err_sist)
print("err-sist:", err_sist)

err = np.sqrt(err_stat**2 + err_sist**2)
print("err:", err)

e = 1.602176634e-19
print("sigma:", np.abs(min - e) / err)
print("perc:", 100 * np.abs(e - min) / e, "%")
