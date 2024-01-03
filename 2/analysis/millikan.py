import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("charges-all.csv")

charges = dataset['q'].to_numpy()
charges_err = dataset['err'].to_numpy()

def S(q):
    sum = 0
    for Q in charges:
        sum += (q - (Q / np.floor(0.5 + (Q / q))))**2
    return sum

dict_d = {}

for i in range(20, 101):
    q_d = np.linspace(1.4e-19, 1.8e-19, i)
    min_d = q_d[0]
    for j in range(len(q_d)):
        if (S(q_d[j]) < S(min_d)):
            min_d = q_d[j]
    e_d = 1.602176634e-19
    perc_d = np.abs(e_d - min_d) / e_d
    dict_d[i] = perc_d

val_min = dict_d[20]
index_min = 20
for i in range(20, 101):
    if dict_d[i] <= val_min:
        val_min = dict_d[i]
        index_min = i
print('index:', index_min)
print("")

q = np.linspace(1.4e-19, 1.8e-19, index_min)

plt.scatter(q, S(q), marker='.', s=10)
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
