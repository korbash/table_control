from scipy.integrate import quad, simps, cumtrapz
from scipy.interpolate import interp1d
from pynverse import inversefunc
import matplotlib.pyplot as plt

import numpy as np
import time


def Map(fun, x):
    return np.array(list(map(fun, x)))


def caunter(r0=62.5, Ltorch=0.6, lw=30, rw=20, dr=1):


    def integr(y, x):
        inte = np.hstack((0, cumtrapz(y, x)))
        return inte

    def Map(fun, x):
        return np.array(list(map(fun, x)))

    lw = lw - Ltorch
    radius = np.linspace(2.5, 62.5, 13)
    thetas = np.array(
        [97.9162688, 36.01154809, 22.44529847, 16.77319816, 14.08756338, 13.28444282, 14.76885344, 19.37716274,
         27.14935304,
         37.4940508, 49.76789206, 63.43683519, 78.09095269])/5
    Theta = interp1d(radius, thetas, kind='cubic')
    r = np.arange(rw, r0, dr)
    dz = Map(lambda x: 1 / float(Theta(x)), r)
    z = integr(dz, r)
    z = z[-1] - z
    inte = integr((r ** 2), z)
    # L = 1 / (r ** 2) * (rw ** 2 * (lw) + 2 * (-inte[-1] + inte))
    L = 1 / (r ** 2) * (rw ** 2 * (lw) + 2 * (-inte))
    x = 2 * z + L - L[-1]
    print(x)
    plt.plot(z, r)
    plt.title('r(z)')
    plt.show()
    plt.plot(x, L-x)
    plt.title('L-x (x)')
    plt.show()
    x = np.append(x, -0.1)
    r = np.append(r, r[-1])
    L = np.append(L, L[-1])
    R_x = interp1d(x, r, kind='cubic')
    L_x = interp1d(x, L, kind='cubic')
    xMax = x[0]
    return L_x, R_x, xMax

'''plt.plot(x, r)
    plt.title('R(x)')
    plt.show()
    plt.plot(x, L)
    plt.title('L(x)')
    plt.show()
    plt.plot(x, L - x)
    plt.title('L(x)-x')
    plt.show()
    plt.plot(z, r)
    plt.title('r(z)')
    plt.show()'''




'''r0 = 125.0 / 2 
lw = 20
rw = 20
Ltorch = 0.6
dr = 0.1
radius = np.linspace(2.5, 62.5, 13)
thetas = np.array(
    [97.9162688, 36.01154809, 22.44529847, 16.77319816, 14.08756338, 13.28444282, 14.76885344, 19.37716274, 27.14935304,
     37.4940508, 49.76789206, 63.43683519, 78.09095269])
Theta = interp1d(radius, thetas, kind='cubic')
r = np.arange(rw, r0, dr)
dz = Map(lambda x: 1 / float(Theta(x)), r)
z = integr(dz, r)
z = z[-1] - z
inte = integr((r ** 2), z)
L = 1 / (r ** 2) * (r0 ** 2 * Ltorch + 2 * (-inte[-1] + inte))
x = 2 * z + L - Ltorch

plt.plot(x, r)
plt.title('R(x)')
plt.show()
plt.plot(x, L)
plt.title('L(x)')
plt.show()
plt.plot(x, L - x)
plt.title('L(x)-x')
plt.show()
plt.plot(z, r)
plt.title('r(z)')
plt.show()'''

'''plt.plot(r, Map(Theta, r))
plt.title('tet(r)')
plt.show()'''
'''x = np.arange(10, 0, -1)
y = np.arange(0, 10, 1)
plt.plot(np.hstack((0, cumtrapz(y, x))))
plt.title('r(z)')
plt.show()'''

L_x, R_x, xMax = caunter()
print(xMax)

x = np.linspace(0, xMax, 100)
plt.plot(x, Map(L_x, x))
plt.title('L(x)')
plt.show()
plt.plot(x, Map(R_x, x))
plt.title('R(x)')
plt.show()
