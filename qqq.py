from korbash_lib import Motor, MotorSystem, Puller, Everage, Exp_everage
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

x = pd.Series({3: 4, 6: 2, 9: 3})
y = pd.Series({3: 5, 6: 3, 9: 7})
print(x)
dat = pd.DataFrame({
    'x': x,
    'y': y
})
dat.loc[15] = {'y': 1}
dat.loc[25] = {'x': 10}
# print(dat['x'].iloc[-1], pd.isnull(dat['y'].iloc[-1]), float('nan'))
# print(dat)
# dat.plot()
# plt.show()
dat.loc[26] = {'x': 3, 'z': 5}
dat.loc[26, 'z'] = 5
dat.reset_index(drop=True, inplace=True)
print(dat)
print(len(dat))
print(dat.shape)
dat.drop(range(3,6), inplace=True)
print(dat.shape)
print(dat)
