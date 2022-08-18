import numpy as np
import math
import pandas as pd
from scipy import optimize
from mylib import *
from .sacred_logger import ex, save_data
from pathlib import Path
# переделать стоп button
main_path = Path(__file__).parents[1]


def Vdiff(R, L):
    r0 = 125.0 / 2
    v0 = 0.01 * 0.8 * 3
    vf = v0 * 2
    rl = r0 / (np.sqrt(vf / v0) - 1)
    L0 = 15
    return v0 * (r0 + rl)**2 / (R + rl)**2 * (L / L0)


class Puller():
    kStr = 58591.17  # gram/mm * mm жёзскость пружины

    def __init__(self, simulate=False):
        if simulate:
            self.sim = simulator(104, self.kStr,
                                 -2.078990076470489)  # -0.0075585384235655265
            tg = self.sim.tg
            pm = self.sim.pm
        else:
            self.sim = None
            tg = dr.tensionGauge()
            Time.sleep(0.001)
            pm = dr.powerMeter()
            Time.sleep(0.001)
        self.tg = ReadingDevise(tg, 'tension', weightCoef=-2.078990076470489)
        Time.sleep(0.001)
        self.pm = ReadingDevise(pm, 'power', weightCoef=1000)
        self.ms = MotorSystem(simulate=simulate, simulator=self.sim)
        self.pidI = 0
        self.trueKmas = np.array([])
        self.Clear()
        self.v = 5
        self.a = 9
        self.dv = 0
        self.tact = 0
        self.times = np.array([])
        self.trueKmasGl = np.array([])

        self.VdifRec = 0

        self.sg = sglad(0.08, 0.2)
        self.MotorsControlStart()
        '''self.Slider = {}
        self.slidersBtn = {}'''

    def __del__(self):
        # del self.ms
        del self.pm
        del self.tg

    def Save(self):
        save_data(self.data, name='pull_resalts.csv')
        ex.run()
        ex.add_artifact(str(main_path / 'data' / 'pull_resalts.csv'))
        # Save(self.data, name='crude.csv', dirSubName='main_data\\DATE')
        # Save(pd.DataFrame({
        #     'time': self.times,
        #     'kof': self.trueKmas,
        #     'sglKof': self.trueKmasGl,
        # }),
        #      name='odrKof.csv',
        #      dirSubName='main_data\\DATE')

    def Clear(self):
        self.data = pd.DataFrame(columns=[
            'time', 'tension', 'power', 'motorL', 'motorR', 'motorM', 'dt',
            'x', 'vL', 'vR', 'vM', 'VdifRec', 'tensionWgl', 'tensionEXPgl', 'dv'
        ])
        self.sg = sglad()

    def Read(self, motoL=True, motoR=True, motoM=True):
        param = {}
        tSt = Time.time()
        param['time'] = tSt
        param['motorL'] = self.ms.motorL.Getposition()
        param['motorR'] = self.ms.motorR.Getposition()
        param['motorM'] = self.ms.motorM.Getposition()
        param['power'] = self.pm.ReadValue()
        param['tension'] = self.tg.ReadValue()
        tFn = Time.time()
        param['dt'] = tFn - tSt
        param['x'], param['L'] = self.ms.calcX_L()
        param['vL'], param['aL'] = self.ms.motorL.calcX_V_A()[1:3]
        param['vR'], param['aR'] = self.ms.motorR.calcX_V_A()[1:3]
        param['vM'], param['aM'] = self.ms.motorM.calcX_V_A()[1:3]
        param['pressure'] = param['tension'] * self.ms.R_x(0)**2 / self.ms.R_x(
            param['x'])**2
        param['dv'] = self.dv
        # param['VdifRec'] = Vdiff(self.ms.R_x(param['x']),
        #                          self.ms.L_x(param['x']))
        self.sg.New(param['tension'], param['vL'])
        self.data.loc[len(self.data)] = param
        if self.sg.expGl.size > 1:
            dt = (param['time'] - self.data['time'].iloc[-10]) / 10
            self.Ttrend = self.sg.trend / dt
            tNew = param['time']
            tSG = self.data.loc[self.sg.iGl, 'time']
            n = (tNew - tSG) / dt
            T1 = self.sg.expGl.iloc[-2]
            T2 = self.sg.expGl.iloc[-1]
            Tnew = T1 + (T2 - T1) * (n + 1)
            self.data.loc[self.sg.iGl, 'tensionWgl'] = self.sg.wGl.iloc[-1]
            self.data.loc[range(self.sg.iGl, len(self.data)),
                          'tensionEXPgl'] = np.linspace(
                              T2, Tnew,
                              len(self.data) - self.sg.iGl)

    def Tprog(self, tau=0):
        return self.Ttrend * tau + self.data.loc[len(self.data) - 1,
                                                 'tensionEXPgl']

    def obrSvas(self, T, Ki, Kp, Kd):
        Tnow = self.sg.level
        dT = self.sg.trend
        E = T - Tnow
        self.pidI += E * Ki * Kp
        per = self.sg.periud
        return Kp * (E + (Kd + per) * dT) + self.pidI

    def SetW(self,
             wide,
             dw=0.1,
             k=None,
             tau=1,
             quiet=True):  ## T - tension, wIdeal, w_ideal
        if k == None:
            k = self.kStr / self.ms.Distance()
        # print(k, ' ', self.ms.Distance())
        t0 = Time.time()
        i = 1
        w = -100
        while abs(wide - w) > dw:
            w = self.tg.ReadValue(tau=tau)
            dx = (wide - w) / k
            if not quiet:
                print('w= ', w, ', dw= ', w - wide, ', dx=  ', dx)
            self.ms.motorR.MoveTo(self.ms.motorR.Getposition(analitic=True) -
                                  dx,
                                  a=1)
            while self.ms.motorR.IsInMotion():
                pass
            Time.sleep(0.1)
        w = self.tg.ReadValue(tau=tau)
        print('SetW finish value=', end=' ')
        print(w)
        '''self.motorR.Getposition()
        self.motorL.Getposition()
        w_points = np.append(w_points, w)
        x_points = np.append(x_points, 200 - self.motorR.points[-1] - self.motorL.points[-1])
        k_data = pd.DataFrame({
            'w': w_points,
            'x': x_points,
        })
        k_data.plot(x='x', y='w')'''
        return w

    def SetH_avto(self,
                  n=1,
                  quiet=True,
                  T=50,
                  v1=1,
                  v2=0.05,
                  Tpr=6,
                  startPos=None):
        if startPos == None:
            startPos = self.ms.motorM.Getposition()
        self.SetW(T, dw=0.1, tau=1, quiet=quiet)
        fitData = pd.DataFrame(columns=['a', 'b', 'x0'])
        for i in range(n):
            self.ms.motorM.MoveTo(startPos)
            x0, a, b = self.SetH_podgon(Tpr=Tpr, v=v1, quiet=quiet)
            xStart = x0 + 0.9
            xFin = x0 - 1.2
            self.ms.motorM.MoveTo(xStart)
            while self.ms.motorM.IsInMotion():
                pass
            Time.sleep(1)
            x0, a, b, data = self.SetH(xStart=xStart,
                                       xFin=xFin,
                                       v=v2,
                                       quiet=quiet,
                                       x0=x0,
                                       a=a,
                                       b=b)
            fitData.loc[i] = [a, b, x0]
        Save(fitData, dirSubName='hM_T\\DATE', name='a_b_x0.csv')
        Save(data, dirSubName='hM_T\\DATE', name='fit.csv')
        x0 = fitData['x0'].mean()
        self.ms.x0 = x0
        return x0

    def SetH_podgon(self, Tpr=6, v=1, quiet=True, a=7):
        def get_f(a, b, x0):
            def f(x):
                if x < x0:
                    return a * (x - x0)**2 + b
                else:
                    return b

            return f

        def errorFun(data_loc, a, b, x0):
            f = get_f(a, b, x0)
            data_loc['error'] = (data_loc['motorM'].apply(f) -
                                 data_loc['tension'])**2
            return sum(data_loc['error'])

        # def f(x):
        #     err = errorFun(data, x[0], x[1], x[2])
        #     print(err)
        #     return err

        T0 = self.tg.ReadValue(tau=2, DataB=DataBase())
        b = T = T0
        self.ms.motorM.MoveTo(0, v, 1)
        i = 0
        DataB = DataBase()
        while T - T0 < Tpr:
            i += 1
            T = self.tg.ReadValue(lastTau=0.2, DataB=DataB)
            DataB.Wright({'motorM': self.ms.motorM.Getposition()},
                         inExsist=True)
        self.ms.Stop('M')
        # print(i)
        self.ms.motorM.FogotMotion()
        self.ms.Start('M')
        x0 = self.ms.motorM.Getposition(analitic=True) + 1
        # print(x0)
        self.ms.motorM.MoveTo(x0 + 1.5)
        data = DataB.data.copy()
        # print(data)
        # print(errorFun(data, a, b, x0))
        # DataBase.Clear(-i)
        print("SetH_podgon", np.array([a, b, x0]))
        resalt = optimize.fmin(lambda x: errorFun(data, x[0], x[1], x[2]),
                               np.array([a, b, x0]),
                               disp=False)  # errorFun(data, x[0], x[1], x[2])
        a = resalt[0]
        b = resalt[1]
        x0 = resalt[2]
        print('podgon', 'a=', a, '  b=', b, '  x0=', x0)
        # data['time'] = data['time'] - data['time'][0]
        data['fit'] = data['motorM'].apply(get_f(a, b, x0))
        if not quiet:
            dp = PlotDisplayer(mainParam='motorM',
                               pl1=data[['motorM', 'error']],
                               pl2=data[['motorM', 'tension', 'fit']])
            dp.Show()
        return x0, a, b

    def SetH(self,
             xStart=11.5,
             xFin=9.5,
             v=0.05,
             quiet=True,
             a=4.5,
             b=50,
             x0=10.7):
        def get_f(a, b, x0):
            def f(x):
                if x < x0:
                    return a * (x - x0)**2 + b
                else:
                    return b

            return f

        def errorFun(data_loc, a, b, x0):
            f = get_f(a, b, x0)
            data_loc['error'] = (data_loc['motorM'].apply(f) -
                                 data_loc['tension'])**2
            return sum(data_loc['error'])

        i = 0
        DataB = DataBase()
        self.ms.motorM.MoveTo(xStart, 1, 1)
        self.ms.motorM.MoveTo(xFin, v, 1)
        while self.ms.motorM.IsInMotion():
            i += 1
            self.tg.ReadValue(DataB=DataB)
            DataB.Wright({'motorM': self.ms.motorM.Getposition()},
                         inExsist=True)
        self.ms.motorM.MoveTo(xStart, v, 1)
        while self.ms.motorM.IsInMotion():
            i += 1
            self.tg.ReadValue(DataB=DataB)
            DataB.Wright({'motorM': self.ms.motorM.Getposition()},
                         inExsist=True)
        data = DataB.data.copy()
        # DataBase.Clear(-i)

        resalt = optimize.fmin(lambda x: errorFun(data, x[0], x[1], x[2]),
                               np.array([a, b, x0]),
                               disp=False)
        a = resalt[0]
        b = resalt[1]
        x0 = resalt[2]
        print('podgon', 'a=', a, '  b=', b, '  x0=', x0)
        # data['time'] = data['time'] - data['time'][0]
        data['fit'] = data['motorM'].apply(get_f(a, b, x0))
        if not quiet:
            dp = PlotDisplayer(mainParam='motorM',
                               pl1=data[['motorM', 'error']],
                               pl2=data[['motorM', 'tension', 'fit']])
            dp.Show()
        return x0, a, b, data

    def MotorsControlStart(self):
        self.phase = -1
        self.tStart = float('+inf')
        self.tStart1 = 0
        self.tFinish1 = 0
        self.tFinish = 0
        self.lastPhase = -1
        self.tEnd = 0
        self.tEnd2 = float('+inf')
        self.isUp = False
        self.stFl = False

    def PulMotorsControl(self,
                         NewMosH,
                         NewT,
                         upFl=True,
                         stFl=False,
                         vsFl=False,
                         dhKof=0.5,
                         ah=9,
                         Ki=0.1,
                         Kp=0.1,
                         Kd=0.1):
        self.Read()

        NewMosH += self.ms.x0
        t = Time.time()
        downPos = self.ms.motorM.position_min + 2
        tau = self.v / self.a
        dhMax = ah * tau**2 / 4
        self.lastPhase = self.phase

        if t < self.tStart:
            self.phase = -1
        elif t < self.tStart1:
            self.phase = 0
        elif t < self.tFinish1:
            self.phase = 1
        elif t < self.tFinish:
            self.phase = 2
        else:
            self.phase = 3

        if self.phase == -1:
            if self.stFl:
                while self.ms.motorM.IsInMotion():
                    Time.sleep(0.001)
                self.ms.motorM.MoveTo(downPos)
                return -1
            else:
                if not self.ms.IsInMotion():
                    self.stFl = self.ms.PulMove(self.v, self.a, 0, stFl)
                    self.sg.NewStTime()
                    self.tEnd = t + self.ms.motorR.CalculateMottonTime()
                if self.tEnd > self.tEnd2:
                    self.tStart = self.tEnd
                if not self.ms.motorM.IsInMotion():
                    if upFl:
                        if not self.isUp:
                            self.ms.motorM.MoveTo(NewMosH + dhMax * dhKof)
                            self.tEnd2 = t + self.ms.motorM.CalculateMottonTime(
                            )
                            self.isUp = True
                    else:
                        if self.isUp:
                            self.ms.motorM.MoveTo(downPos)
                            self.isUp = False

        if self.phase == 3:
            if self.phase != self.lastPhase:
                if self.ms.IsInMotion():
                    self.phase = self.lastPhase
                    return 0
                if not upFl or self.stFl:
                    self.tStart = float('+inf')
                    self.tStart1 = 0
                    self.tFinish1 = 0
                    self.tFinish = 0
                    self.tEnd = 0
                    self.tEnd2 = float('+inf')
                else:
                    if self.tact > 4:
                        if vsFl and self.tact > 6:
                            self.dv = self.obrSvas(NewT, Ki, Kp, Kd)
                            # self.tFinish - self.tStart,
                        # print(self.v, self.a, self.dv, self.t_new, self.t_last, self.T_new, self.T_last)
                    self.stFl = self.ms.PulMove(self.v, self.a, self.dv, stFl)
                    self.sg.NewStTime()
                    self.tStart = Time.time()
                    self.tStart1 = self.tStart + self.v / self.a
                    self.tFinish = self.tStart + self.ms.motorR.CalculateMottonTime(
                    )
                    self.tFinish1 = self.tFinish - self.v / self.a
                    if self.tFinish1 < self.tStart1:
                        self.tStart1 = self.tFinish1 = (self.tStart +
                                                        self.tFinish) / 2
                    dh1 = self.ms.motorM.Getposition() - NewMosH
                    p = self.ms.motorM.Getposition()
                    if abs(dh1) > dhMax:
                        dh1 = dhMax * np.sign(dh1)
                        print('you so fast, i think it is too math')
                    vh = 1 / 2 * ah * (tau - math.sqrt(
                        (ah * tau**2 - 4 * abs(dh1)) / ah))
                    while self.ms.motorM.IsInMotion():
                        Time.sleep(0.001)
                    self.ms.motorM.Move(-dh1, vh, ah)
                    self.tact += 1

        if self.phase == 2:
            if self.phase != self.lastPhase:
                while self.ms.motorM.IsInMotion():
                    Time.sleep(0.001)
                # p=self.motorM.Getposition()
                dh = dhMax * dhKof
                vh = 1 / 2 * ah * (tau - math.sqrt(
                    (ah * tau**2 - 4 * dh) / ah))
                self.ms.motorM.Move(-dh, vh, ah)
        return 0

    def Test(self):
        print('tg test:')
        self.tg.Test()
        print('\npm test:')
        self.pm.Test()
        print('\nms test:')
        self.ms.Test()