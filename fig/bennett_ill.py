#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

X = np.linspace(0,7,100)

def os_bennett(l, fo, tnd):
    return ((l*fo + (np.exp(l) - l - 1)*tnd)/(np.exp(l/2)-1))

for i in np.exp(np.linspace(-2,1,10)):
    plt.plot(X,os_bennett(X,10,i), label=f"{i.round(2)}")
plt.legend()
plt.savefig("fig/test.png")
plt.close()
