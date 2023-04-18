import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

x = np.arange(-10, 10, 0.1)
y = softmax(x)

plt.plot(x, y)
plt.title('Softmax Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.show()
