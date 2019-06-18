import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 1, 1000)
print(x)

# S_current = S_target + (S_start - S_target) * (1 - time)**k


def func(x, k=3, S_start=0, S_target=0.9):
    return S_target + (S_start - S_target) * (1 - x)**k


for i in range(1, 8):
    plt.plot(x, func(x, i), label="k = {}".format(i))
plt.xlabel("time")
plt.ylabel("sparsity")
plt.legend()
plt.show()
