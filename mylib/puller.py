import asyncio
import numpy as np
import math
import pandas as pd
from scipy import optimize
from mylib import *
from .sacred_logger import ex, save_data
from pathlib import Path
import tomli
from bokeh.io import push_notebook
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
                                 2.078990076470489)  # -0.0075585384235655265
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
        self.Clear()
        self.sg = sglad()  #sglad(0.08, 0.2)
        self.dts = []
        self.dts2 = []
        self.dts3 = []
        self.dtsm = []
        '''self.Slider = {}
        self.slidersBtn = {}'''

    def change_params(self, from_file: bool = False, **params):
        if from_file:
            with open('C:/Users/ariva/projects/table_control/params.toml',
                      'rb') as f:
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
        self.pd.Show()
        self.tasks.append(asyncio.create_task(self.PulMotorsControl(), name='motorsLR'))
        # self.tasks.append(asyncio.create_task(self.FireMove(), name='motorM'))
        self.tasks.append(asyncio.create_task(self.Read()))
        self.tasks.append(asyncio.create_task(self.plotter()))

    async def __aexit__(self, *params):
        await self.tasks[0]
        for task in self.tasks:
            task.cancel()
        print(f'mean reading time = {np.mean(self.dts)}')
        print(f'max reading time = {np.max(self.dts)}')

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

    async def plotter(self):
        while True:
            # print('plotted')
            w = int(self.sl.Sl['window'])
            self.pd.Apdate(for_all=self.data.iloc[-w:])
            push_notebook()
            await asyncio.sleep(2.5)

    async def Read(self):
        print('aaaa')
        while True:
            param = {}
            tSt = Time.time()
            param['time'] = tSt
            await asyncio.sleep(0)
            param['motorL'] = self.ms.motorL.Getposition()
            await asyncio.sleep(0)
            param['motorR'] = self.ms.motorR.Getposition()
            await asyncio.sleep(0)
            param['motorM'] = self.ms.motorM.Getposition()
            await asyncio.sleep(0)
            param['power'] = self.pm.ReadValue()
            await asyncio.sleep(0)
            param['tension'] = self.tg.ReadValue()
            await asyncio.sleep(0)
            tFn = Time.time()
            param['dt'] = tFn - tSt
            await asyncio.sleep(0)
            param['x'], param['L'] = self.ms.calcX_L()
            await asyncio.sleep(0)
            param['Le'] = self.ms.funL_x(param['x'])
            await asyncio.sleep(0)
            param['vL'], param['aL'] = self.ms.motorL.calcX_V_A()[1:3]
            await asyncio.sleep(0)
            param['vR'], param['aR'] = self.ms.motorR.calcX_V_A()[1:3]
            await asyncio.sleep(0)
            param['vM'], param['aM'] = self.ms.motorM.calcX_V_A()[1:3]
            await asyncio.sleep(0)
            param['pressure'] = param['tension'] * self.ms.R_x(0)**2 / self.ms.R_x(param['x'])**2
            await asyncio.sleep(0)
            param['dv'] = self.dv
            param['hFire'] = self.ms.hFire
            await asyncio.sleep(0)
            tFn15 = Time.time()
            param['tensionGoal'] = self.NewT
            tFn17 = Time.time()
            self.sg.NewPoint(param['tension'], param['time'])
            l = len(self.data)
            self.data.loc[l] = param
            if self.sg.mean is None: 
                self.wi = l
            else:
                while self.data.loc[self.wi, 'time'] <= self.sg.t:
                    self.wi += 1
                    self.data.loc[self.wi, 'tensionWgl'] = self.sg.mean
                    self.data.loc[self.wi, 'tensionEXPgl'] = self.sg.level
            tFn2 = Time.time()
            await asyncio.sleep(.5)
            # print(f'dt={tFn2 - tSt}')
            self.dts.append(tFn2 - tSt)
            self.dts2.append(tFn2 - tFn15)
            self.dts3.append(tFn2 - tFn17)

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
        for i in range(40):
            self.dtsm.append(Time.time())
            # self.stFl = self.sl.BtnFl['end']
            self.stFl = self.sl.Sl['end']
            self.a = self.sl.Sl['a']
            self.v = self.sl.Sl['v']
            T = self.sl.Sl['T0']
            Ki = self.sl.Sl['Ki']
            Kp = self.sl.Sl['Kp']
            Kd = self.sl.Sl['Kd']
            # alf = self.sl.Sl['alf']
            x, L = self.ms.calcX_L()
            r0 = self.ms.funR_x(x)
            # r_max = self.ms.funR_x(self.ms.xMax)
            r = self.ms.funR_x(x)
            t = T * r**2 / r0**2
            self.NewT = t
            if self.sg.level is not None:
                self.dv = self.obrSvas(t, Ki, Kp, Kd)
            else:
                self.dv = 0
                self.stFl = await self.ms.PulMove(self.v, self.a, self.dv,
                                                  self.stFl, self.sg.New_tact)
            if self.stFl:
                break

    async def FireMove(self):
        while True:
            await self.ms.motorM.MoveTo(self.ms.x0 + self.sl.Sl['molibdenH'])

    def Test(self):
        print('tg test:')
        self.tg.Test()
        print('\npm test:')
        self.pm.Test()
        print('\nms test:')
        self.ms.Test()