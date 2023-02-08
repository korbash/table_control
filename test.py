# import pandas as pd
# from mylib.sacred_logger import save

# df = pd.DataFrame({'q': [1, 3, 6, 9], 'w': [2, 4, 7, 10]})
# save(df, 'data', 'test_pull')

from matplotlib import pyplot as plt
import numpy as np
from scipy import optimize
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d

def caunter(r0=62.5, Ltorch=0.6, lw=30, rw=15, dr=1):
    def integr(y, x):
        inte = np.hstack((0, cumtrapz(y, x)))
        return inte

    def Map(fun, x):
        return np.array(list(map(fun, x)))

    lw = lw - Ltorch
    radius = np.linspace(2.5, 62.5, 13)
    thetas = np.array([
        97.9162688, 36.01154809, 22.44529847, 16.77319816, 14.08756338,
        13.28444282, 14.76885344, 19.37716274, 27.14935304, 37.4940508,
        49.76789206, 63.43683519, 78.09095269
    ]) / 2
    Theta = interp1d(radius, thetas, kind='cubic')
    r = np.arange(rw, r0, dr)
    dz = Map(lambda x: 1 / float(Theta(x)), r)
    z = integr(dz, r)
    z = z[-1] - z
    inte = integr((r**2), z)
    # L = 1 / (r ** 2) * (rw ** 2 * (lw) + 2 * (-inte[-1] + inte))
    L = 1 / (r**2) * (rw**2 * (lw) + 2 * (-inte))
    x = 2 * z + L - L[-1]

    x = np.append(x, -0.1)
    r = np.append(r, r[-1])
    L = np.append(L, L[-1])
    R_x = interp1d(x, r, kind='cubic')
    L_x = interp1d(x, L, kind='cubic')
    xMax = x[0]
    print(z)
    print(inte)
    return L_x, R_x, xMax


L_x, R_x, xMax = caunter(r0=62.5, Ltorch=0.6, lw=30, rw=15, dr=1)

x= np.linspace(0,xMax,20)
plt.plot(L_x(x))
plt.show()