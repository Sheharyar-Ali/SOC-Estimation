import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
x_1 = np.linspace(-1.618,0,50)
y_1 = x_1**3 + x_1**2 - x_1
integral_1 = scipy.integrate.simpson(y_1,x_1)
x_2 = np.linspace(0,0.618,50)
y_2 = x_2**3 + x_2**2 -x_2
integral_2 = scipy.integrate.simpson(y_2,x_2)

x_full = np.linspace(-1.618,0.618,50)
y_full = x_full**3 +x_full**2 -x_full
integral_full = scipy.integrate.simpson(y_full,x_full)

combined= np.zeros_like(x_full)

print(integral_full)
print(integral_1)
print(integral_2)
print(integral_1+integral_2)
plt.plot(x_full,y_full,label="full")
#plt.plot(x_2,y_2,label="y2")
plt.legend()
plt.grid()
plt.show()