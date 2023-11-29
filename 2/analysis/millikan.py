import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("charges.csv")

charges = dataset['q'].to_numpy()
charges = np.sort(charges)

def S(q):
    sum = 0
    for Q in charges:
        sum += (q - (Q / np.floor(0.5 + (Q / q))))**2
    return sum

q = np.linspace(1.4e-19, 1.8e-19, 1000000)

plt.plot(q, S(q))
plt.savefig("graph_py.png")

x = np.arange(charges.shape[0])
plt.plot(x, charges)
plt.show()
