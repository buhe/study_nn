import numpy as np

a = np.array([1,2,3])
b = a.reshape(1, 3)
c = np.array([1,2,3])

print(c / a)
print(c / b)
