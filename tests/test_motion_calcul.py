from korbash_lib import Puller, Time, self
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def chek_calcX_V_A(x0, v0, a0, n=100):
    masX = []
    masV = []
    masA = []
    tMax = pl.ms.motorR.CalculateMottonTime(x0, v0, a0) * 1.2
    masT = np.linspace(0, tMax, n)
    print('starting test')
    for t in masT:
        x, v, a = pl.ms.motorR.calcX_V_A(t, x0, v0, a0)
        masX += [x]
        masV += [v]
        masA += [a]

    plt.plot(masT, masX)
    plt.title('x')
    plt.show()

    plt.plot(masT, masV)
    plt.title('v')
    plt.show()

    plt.plot(masT, masA)
    plt.title('a')
    plt.show()


def chek_virtMoveTo(x0, v0, a0, n=100):
    masX = []
    masX2 = []
    masV = []
    masA = []
    masT = []
    xSt = pl.ms.motorR.Getposition()
    tMax = pl.ms.motorR.CalculateMottonTime(x0 - xSt, v0, a0) * 1.2
    dt = tMax / n
    print('starting test', 'tMax=', tMax)
    t0 = Time.time()
    pl.ms.motorR.MoveTo(x0, v0, a0)

    for i in range(n):
        x, v, a, isin = pl.ms.motorR.calcX_V_A_IsInMot()
        x2 = pl.ms.motorR.Getposition()
        masT += [Time.time() - t0]
        masX += [x]
        masX2 += [x2]
        masV += [v]
        masA += [a]
        Time.sleep(dt)

    masX = pd.Series(masX, name='x')
    masX2 = pd.Series(masX2, name='x2')
    masV = pd.Series(masV, name='v')
    masA = pd.Series(masA, name='a')
    masT = pd.Series(masT, name='time')
    dat = pd.DataFrame((masT, masX, masX2, masV, masA)).T
    print(dat)

    dat.plot(x='time', y=['x', 'x2'])
    plt.show(block=True)
    dat.plot(x='time', y=['v', 'a'])
    plt.show(block=True)


def chek_virtMove(x0, v0, a0, n=100):
    masX = []
    masXL = []
    masV = []
    masA = []
    masT = []
    xSt = pl.ms.motorR.Getposition()
    tMax = pl.ms.motorR.CalculateMottonTime(x0, v0, a0) * 1.2
    dt = tMax / n
    print('starting test', 'tMax=', tMax)
    t0 = Time.time()
    pl.ms.Move(x0, v0, a0, 0.0, 0)
    for i in range(n):
        x, v, a, isin = pl.ms.motorR.calcX_V_A_IsInMot()
        xL = pl.ms.motorL.Getposition()
        masT += [Time.time() - t0]
        masX += [x]
        masXL += [xL]
        masV += [v]
        masA += [a]
        Time.sleep(dt)

    masX = pd.Series(masX, name='x')
    masXL = pd.Series(masXL, name='xL')
    masV = pd.Series(masV, name='v')
    masA = pd.Series(masA, name='a')
    masT = pd.Series(masT, name='time')
    dat = pd.DataFrame((masT, masX, masXL, masV, masA)).T
    dat.plot(x='time', y=['x', 'xL'])
    plt.show(block=True)
    dat.plot(x='time', y=['v', 'a'])
    plt.show(block=True)


def chek_PulMove(v0, a0, dv=0, n=100, x0=15, steps=1):
    masX = []
    masXL = []
    masV = []
    masA = []
    masT = []
    t0 = Time.time()
    xSt = pl.ms.motorR.Getposition()
    tMax = pl.ms.motorR.CalculateMottonTime(x0, v0, a0) * 1.2
    dt = tMax / n
    print('starting test', 'tMax=', tMax, 't=', Time.time() - t0)
    for step in range(steps):
        fl = pl.ms.PulMove(v0, a0, dv,stFl=False)
        while pl.ms.IsInMotion():
            x, v, a, isin = pl.ms.motorR.calcX_V_A_IsInMot()
            xL = pl.ms.motorL.Getposition()
            masT += [Time.time() - t0]
            masX += [x]
            masXL += [xL]
            masV += [v]
            masA += [a]
            Time.sleep(dt)
        if fl == -1: break

    masX = pd.Series(masX, name='x')
    masXL = pd.Series(masXL, name='xL')
    masV = pd.Series(masV, name='v')
    masA = pd.Series(masA, name='a')
    masT = pd.Series(masT, name='time')
    dat = pd.DataFrame((masT, masX, masXL, masV, masA)).T
    dat.plot(x='time', y=['v', 'a'])
    plt.show(block=True)
    dat.plot(x='time', y=['x', 'xL'])
    plt.show(block=True)


def chek_PulMotorsControl(v0, a0, dv=0, tau=5):
    masX = []
    masXL = []
    masM = []
    masV = []
    masA = []
    masT = []
    t0 = Time.time()
    print('starting test')
    i = 0
    t = 0
    pl.v=v0
    pl.a=a0
    pl.dv=dv
    while t < tau:
        t = Time.time() - t0
        if pl.PulMotorsControl(1 - 0*t / 30,20)==-1: break
        x, v, a, isin = pl.ms.motorR.calcX_V_A_IsInMot()
        xL = pl.ms.motorL.Getposition()
        xM = pl.ms.motorM.Getposition()
        masT += [Time.time() - t0]
        masX += [x]
        masM += [xM]
        masXL += [xL]
        masV += [v]
        masA += [a]
        Time.sleep(0.01)
        i += 1

    masX = pd.Series(masX, name='x')
    masM1 = pd.Series(masM, name='xM1') * 50 - 500 + 20
    masM2 = (pd.Series(masM, name='xM2')-11)*10
    masXL = pd.Series(masXL, name='xL')
    masV = pd.Series(masV, name='v')
    masA = pd.Series(masA, name='a')
    masT = pd.Series(masT, name='time')
    dat = pd.DataFrame((masT, masX, masM2, masM1, masXL, masV, masA)).T
    dat.plot(x='time', y=['v', 'a', 'xM2'])
    plt.show(block=True)
    dat.plot(x='time', y=['x', 'xL', 'xM1'])
    plt.show(block=True)


pl = Puller(simulate=True)
Time.SetZeroTime()
Time.freez()

# chek_calcX_V_A(x0=10, v0=1, a0=0.01)
#
# chek_virtMoveTo(x0=60, v0=20, a0=30)
# chek_virtMoveTo(x0=30, v0=10, a0=20)

# pl.ms.motorR.MoveTo(84.27880674523512, 100, 100)
# pl.ms.motorL.MoveTo(84.27880674523512, 100, 100)
# while pl.ms.IsInMotion():
#     pass
# chek_virtMove(x0=-4.721193254764884, v0=10, a0=20)
# chek_virtMove(x0=8.463381224320372, v0=10, a0=20)

# pl.ms.MoveToStart(v=100, a=100, zapas=0.1)
# while pl.ms.IsInMotion():
#     pass
# print(pl.ms.motorR.Getposition(), pl.ms.motorL.Getposition())
# chek_PulMove(10, 100, dv=1, x0=10, steps=10)

pl.ms.MoveToStart(v=100, a=100, zapas=20)
pl.ms.motorM.MoveTo(12.1, v=100, a=100)
pl.ms.x0 = 10
while pl.ms.IsInMotion(all=True):
    Time.sleep(0.01)
print(pl.ms.motorR.Getposition(), pl.ms.motorL.Getposition())
chek_PulMotorsControl(3, 7, dv=0.5, tau=100)
