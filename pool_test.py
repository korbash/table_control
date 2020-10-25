from korbash_lib import Puller
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

pl = Puller(simulate=True)
pl.ms.MoveToStart(zapas=0.1)
while pl.ms.IsInMotion():
    pass
t0 = time.time()
t = t0
while t - t0 < 10:
    t = time.time()
    pl.Read()
    if not pl.ms.IsInMotion():
        pl.ms.PulMove(10, 20, 0.5)
    time.sleep(0.01)

print(pl.data)
pl.data.plot(x='time', y=['motorL', 'motorR'])
plt.show(block=True)
