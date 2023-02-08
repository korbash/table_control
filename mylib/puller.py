import asyncio
import numpy as np
import math
import pandas as pd
from scipy import optimize
from mylib import *
from .sacred_logger import ex, save_data
from pathlib import Path
import tomli
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

    def __init__(self, simulate=False, blocking=False):
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
        self.ms = MotorSystem(simulate=simulate,
                              simulator=self.sim,
                              blocking=blocking)
        self.trueKmas = np.array([])
        self.times = np.array([])
        self.trueKmasGl = np.array([])

        self.sg = sglad()  #sglad(0.08, 0.2)
        '''self.Slider = {}
        self.slidersBtn = {}'''

    def change_params(self, from_file: bool = False, **params):
        if from_file:
            with open('params.toml') as f:
                params = tomli.load(f)
        for k, v in params.items():
            setattr(self, k, v)

    def __del__(self):
        # del self.ms
        del self.pm
        del self.tg

    async def __aenter__(self):
        self.change_params(from_file=True)
        self.Clear()
        self.MotorsControlStart()
        self.tasks = []
        self.tasks.append(asyncio.create_task(self.PulMotorsControl()))
        self.tasks.append(asyncio.create_task(self.FireMove()))
        self.tasks.append(asyncio.create_task(self.Read))

    async def __aexit__(self, *params):
        await self.task[0]
        for task in self.tasks:
            task.cancel()

    def Save(self):
        save_data(self.data, name='pull_resalts.csv')
        ex.run()
        ex.add_artifact(str(main_path / 'data' / 'pull_resalts.csv'))

    def Clear(self):
        self.data = pd.DataFrame(columns=[
            'time', 'tension', 'power', 'motorL', 'motorR', 'motorM', 'dt',
            'x', 'vL', 'vR', 'vM', 'VdifRec', 'tensionWgl', 'tensionEXPgl',
            'dv', 'tensionGoal', 'kP', 'kI', 'dL', 'Le'
        ])
        self.sg = sglad()

    async def Read(self):
        while True:
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
            param['Le'] = self.ms.funL_x(param['x'])
            param['vL'], param['aL'] = self.ms.motorL.calcX_V_A()[1:3]
            param['vR'], param['aR'] = self.ms.motorR.calcX_V_A()[1:3]
            param['vM'], param['aM'] = self.ms.motorM.calcX_V_A()[1:3]
            param['pressure'] = param['tension'] * self.ms.R_x(
                0)**2 / self.ms.R_x(param['x'])**2
            param['dv'] = self.dv
            param['hFire'] = self.ms.hFire
            param['tensionGoal'] = self.Tgoal
            self.sg.NewPoint(param['tension'], param['time'])
            l = len(self.data)
            self.data.loc[l] = param
            if self.sg.mean is None: self.wi = l
            else:
                while self.data.loc[self.wi, 'time'] <= self.sg.t:
                    self.wi += 1
                    self.data.loc[self.wi, 'tensionWgl'] = self.sg.mean
                    self.data.loc[self.wi, 'tensionEXPgl'] = self.sg.level
            asyncio.sleep(0.3)

    def Tprog(self, tau=0):
        return self.Ttrend * tau + self.data.loc[len(self.data) - 1,
                                                 'tensionEXPgl']

    def cof_forPidI(self, T0):
        T = self.sg.level
        if T > 2 * T0:
            return 0
        else:
            return 2 - T / T0

    def obrSvas(self, T, Ki, Kp, Kd):
        Tnow = self.sg.level
        dT = self.sg.trend
        E = T - Tnow
        self.pidI += max(E, 0) * Ki * Kp
        self.pidI *= self.cof_forPidI(T)
        x, L = self.ms.calcX_L()
        Imax = 0.1 + 0.7 * x / self.ms.xMax
        self.pidI = min(self.pidI, Imax)
        return Kp * (E + Kd * dT) + self.pidI

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
            asyncio.run(
                self.ms.motorR.MoveTo(
                    self.ms.motorR.Getposition(analitic=True) - dx, a=1))
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

    def MotorsControlStart(self):
        self.phase = 3
        self.isUp = False
        self.stFl = False
        self.ms.ResetBeforePull()

    async def PulMotorsControl(self):
        for i in range(5):
            self.stFl = await self.ms.PulMove(self.v, self.a, self.dv,
                                              self.stFl)
            if self.stFl:
                break

    async def FireMove(self):
        while True:
            await self.ms.motorM.MoveTo(self.hFire)

    # def PulMotorsControl(self,
    #                      NewMosH,
    #                      NewT,
    #                      upFl=True,
    #                      stFl=False,
    #                      dhKof=0.5,
    #                      Ki=0.1,
    #                      Kp=0.1,
    #                      Kd=0.1):
    #     self.NewT = NewT
    #     self.Ki = Ki
    #     self.Kp = Kp
    #     self.MoH = NewMosH
    #     self.Read()
    #     if upFl and not stFl:
    #         NewMosH += self.ms.x0
    #     else:
    #         NewMosH = self.ms.downPos
    #     t = Time.time()
    #     tasks = []
    #     if self.ms.tStart1 <= t < self.ms.tFinish1 and self.phase != 1:  # мотрчик едет с постоянной скоростью
    #         self.phase = 1
    #         # print('moving with constant speed')
    #         self.ms.PulFireMove(aEnd=20, vEnd=dhKof, vFon=self.vFon)
    #     elif self.ms.tFinish1 <= t < self.ms.tFinish and self.phase != 2:  # моторчик тормозит
    #         self.phase = 2
    #         # print('stoping')
    #         self.ms.PulFireMove(aEnd=20, vEnd=dhKof, vFon=self.vFon)
    #     elif t >= self.ms.tFinish:  # моторчик закончил движение
    #         self.phase = 3
    #         if self.sg.level is not None:
    #             self.dv = self.obrSvas(NewT, Ki, Kp, Kd)
    #         self.stFl = self.ms.PulMove(self.v, self.a, self.dv, stFl)
    #         self.sg.New_tact(self.ms.tFinish)
    #         self.vFon = self.ms.VforFireMove(NewMosH)
    #         # print(self.vFon, NewMosH, NewMosH - self.ms.motorM.Getposition())
    #         if self.stFl:
    #             self.ms.motorM.MoveTo(self.ms.downPos)
    #             return -1
    #         # print('starting')
    #         self.ms.PulFireMove(aEnd=20, vEnd=dhKof, vFon=self.vFon)
    #     return 0

    def Test(self):
        print('tg test:')
        self.tg.Test()
        print('\npm test:')
        self.pm.Test()
        print('\nms test:')
        self.ms.Test()