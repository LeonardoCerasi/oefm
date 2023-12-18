import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("charges.csv")
dataset_err = pd.read_csv("charges_err.csv")

charges = dataset['q'].to_numpy()
charges_err = dataset_err['err'].to_numpy()

def S(q):
    sum = 0
    for Q in charges:
        sum += (q - (Q / np.floor(0.5 + (Q / q))))**2
    return sum

q = np.linspace(1.4e-19, 1.8e-19, 10000)

plt.plot(q, S(q))
plt.savefig("graph_py.png")

min = q[0]
for i in range(len(q)):
    if (S(q[i]) < S(min)):
        min = q[i]
print("min:", min)
print("err-stat:", np.sqrt(S(min) / (len(charges) * (len(charges) - 1))))

err = 0
for i in range(len(charges)):
    err += charges_err[i] / (len(charges) * np.floor(0.5 + (charges[i] / min)))
print("err-sist:", err)
