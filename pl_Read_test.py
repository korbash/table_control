from korbash_lib import Puller, sglad, Time, DataBase
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

Time.SetZeroTime()
Time.freez()
pl = Puller(simulate=True)
v = 3
a = 7
dv = 0.001
pl.ms.x0 = 11

pl.ms.MoveToStart(v=100, a=100, zapas=22)
pl.ms.motorM.MoveTo(12.1, v=100, a=100)
while pl.ms.IsInMotion(all=True):
    Time.sleep(0.01)

for i in range(400):
    pl.Read()
    if Time.time()>15:
        dv = pl.obrSvas(17000, dv)
        print(dv)
    if pl.ms.PulMotorsControl(v, a, dv, 0, upFl=True, stFl=False,
                              dhKof=0.5, ah=9) == -1:
        break
    Time.sleep(0.1)
pl.data.plot(x='time',y=['tensionEXPgl','tensionWgl', 'tension'])
# DataBase.data[['tension', 'tensionWgl', 'tensionEXPgl']].plot()
plt.show(block=True)
# dp = PlotDisplayer()
# dp.CreateMaket(mainParam='time', tension=pl.data[['time', 'tension', 'tensionWgl', 'tensionEXPgl']])
# show(dp.grid)
print(pl.data)
2 + 2
