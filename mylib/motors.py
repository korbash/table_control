from ast import Yield
from tkinter import Y
import numpy as np
import math
import asyncio
from scipy import optimize
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
import ipywidgets
# from IPython import display
from IPython.display import display
import PySimpleGUI as sg
from .utilities import *
from . import driwers as dr


def caunter(r0=62.5, Ltorch=0.6, lw=30, rw=15, dr=1, thetasdiv=1):

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
    ]) / thetasdiv
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
    R_x = interp1d(x, r, kind='cubic', fill_value='extrapolate')
    L_x = interp1d(x, L, kind='cubic', fill_value='extrapolate')
    xMax = x[0]
    return L_x, R_x, xMax


def findEndPoint(xSt, lSt, alf, L_x, xMax):
    if alf == 0:
        x = xSt
    else:
        x = optimize.root_scalar(lambda x: L_x(x) / 2 -
                                 ((x - xSt) / alf + lSt),
                                 bracket=[-20, xMax + 20],
                                 method='brentq').root
    return x, L_x(x) / 2 - lSt


class Motor():
    v_norm = 5
    a_norm = 5
    v_max = 30
    a_max = 30

    def __init__(self, mot, motor_name, p_max=99, blocking=False):

        self.mot = mot
        self.name = motor_name
        self.position_max = p_max
        self.position_min = 0
        self.blocking = blocking
        self.stopFlag = False
        self.points = np.array([])
        self.startsTime = np.array([Time.time()])
        self.finX = np.array([0, 0])

        # self.acceleration_upper_limit=30
        # self.velocity_upper_limit=30
        self.saveX = 0
        self.saveA = 0
        self.saveV = 0
        self.Set_velocity()
        self.FogotMotion(analitic=False)
        self.Start()

    def __del__(self):
        # if not self.simFlag:
        #     apt.Motor.disable()
        # del apt.Motor
        del self.mot

    def Set_velocity(self, v=v_norm, a=a_norm):
        flag = 0
        if math.isnan(v) or math.isnan(a):
            print("error in ", self.name, " -  v or a is not nuber v=", v,
                  " a=", a)
            flag = -1
        if (v > Motor.v_max or v < 0):
            print("error in ", self.name, " -  v is bad v=", v)
            flag = -1
        if (a > Motor.a_max or a <= 0):
            print("error in ", self.name, " -  a is bad a=", a)
            flag = -1
        if self.stopFlag:
            print("error in ", self.name, " -  motor was stoped ")
            flag = -1
        if (flag == 0):
            self.saveA = a
            self.saveV = v
            if v == 0:
                v = 0.1
            self.mot.set_velocity_parameters(0, a, v)
        return flag

    def Move_to_iner(self, x):
        flag = 0
        if math.isnan(x):
            print("error in ", self.name, " -  x isnt number", x, "  ",
                  type(x))
            flag = -3
        if x > self.position_max:
            print("error in ", self.name, " -  position is bad x=", x,
                  "max position=", self.position_max)
            flag = -1
            x = self.position_max
        if x < self.position_min:
            print("error in ", self.name, " -  position is bad x=", x,
                  "min position=", self.position_min)
            flag = -2
            x = self.position_min
        if self.stopFlag:
            print("error in ", self.name, " -  motor was stoped ")
            flag = -3
        if flag != -3:
            self.saveX = x
            self.startsTime = np.append(self.startsTime, Time.time())
            self.finX = np.append(self.finX, x)
            # print("Move_to_iner", self.name, "  ", self.saveX)
            # print('v and a   ', self.saveV,"  ",self.saveA)
            if x != self.Getposition():
                self.mot.move_to(x, blocking=self.blocking)
        return flag

    def Getposition(self, memory=True, analitic=True, motorNotMove=False):
        if not analitic:
            x = self.mot.position
        else:
            if motorNotMove or self.stopFlag:
                # print('not move', self.finX[-2],self.finX[-1])
                x = self.finX[-1]
            else:
                # print('in move', self.finX[-2],self.finX[-1])
                x = self.calcX_V_A()[0]
        if memory:
            self.points = np.append(self.points, x)
        # print('Getposition', x)
        return x

    def IsInMotion(self, analitic=False):
        if analitic:
            x, v, a, isin = self.calcX_V_A_IsInMot()
            return isin
        else:
            return self.mot.is_in_motion

    async def WaitWhileInMotion(self, waitForTrue=True):
        while not self.IsInMotion() and waitForTrue:
            await asyncio.sleep(0)
        while self.IsInMotion():
            await asyncio.sleep(0)

    async def Move(self, dp, v=v_norm, a=a_norm):
        await self.WaitWhileInMotion(waitForTrue=False)
        p = self.Getposition(memory=False, motorNotMove=True)
        v = self.chekV_A(v, a, dp)
        self.Set_velocity(v, a)
        flag = self.Move_to_iner(p + dp)
        await self.WaitWhileInMotion()
        return flag

    async def MoveTo(self, x, v=v_norm, a=a_norm):
        await self.WaitWhileInMotion(waitForTrue=True)
        x0 = self.Getposition(memory=False, motorNotMove=True)
        v = self.chekV_A(v, a, x - x0)
        self.Set_velocity(v, a)
        flag = self.Move_to_iner(x)
        await self.WaitWhileInMotion(waitForTrue=False)
        return flag

    def Test(self):
        x = self.Getposition()
        self.MoveTo(x + 5)
        self.MoveTo(x)

    def Clear(self, n=-1):
        if (n != -1 and len(self.points) > n):
            self.points = self.points[::-n - 1]
        else:
            self.points = np.array([])
        return 0

    def btnStopStart(self, flag=None):
        # print("btnStopStart",self.name,"  ",)
        # fl_st = self.stopFlag
        if flag != None:
            self.stopFlag = flag
        if self.stopFlag:
            # if fl_st:
            self.Start()
            return "Stop " + self.name
        else:
            self.Stop()
            return "Start " + self.name

    def Stop(self):
        # print("stop ",self.name,"  ")
        self.finX[-1] = self.Getposition(memory=False)  # важен порядок
        self.stopFlag = True
        self.mot.stop_profiled()

    def Start(self):
        self.stopFlag = False
        # print("start ",self.name,"  ", self.saveX,"  ", self.saveV,"  ", self.saveA)
        self.MoveTo(self.saveX, self.saveV, self.saveA)

    def FogotMotion(self, analitic=False, memory=False):
        # print("FogotMotion xaxa vsem meshau", self.Getposition(analitic=analitic, memory=False))
        self.finX[-1] = self.saveX = self.Getposition(analitic=analitic,
                                                      memory=False)

    def CheckPosition(self, pos):
        if (pos > self.position_max or pos < self.position_min):
            print("Invalid position")
            return -1
        else:
            return 0

    def CalculateMottonTime(self, x=None, v=None, a=None):  # исправить
        if x == None:
            x = self.finX[-1] - self.finX[-2]
        if v == None:
            v = self.saveV
        if a == None:
            a = self.saveA
        v = self.chekV_A(v, a, x)
        a = abs(a)
        x = abs(x)
        return x / v + v / a

    def CalculateMottonDist(self, t, v=None, a=None):
        if v == None:
            v = self.saveV
        if a == None:
            a = self.saveA
        v = abs(v)
        a = abs(a)
        if t < 0:
            print('CalculateMottonDist error t<0 t=', t)
            return 'error'
        if t > 2 * v / a:
            return (t - v / a) * v
        else:
            t = t / 2
            return a * t**2

    def calcX_V_A(self, t=None, x=None, v=None, a=None):
        x, v, a, isin = self.calcX_V_A_IsInMot(t=t, x=x, v=v, a=a)
        return x, v, a

    def calcX_V_A_IsInMot(self, t=None, x=None, v=None, a=None):
        if t == None:
            t = Time.time() - self.startsTime[-1]
        if x == None:
            x0 = self.finX[-2]
            x1 = self.finX[-1]
            x = x1 - x0
        else:
            x0 = 0
            x1 = x0 + x
        if x == 0:
            return x1, 0, 0, False
        if v == None:
            v = self.saveV
        if a == None:
            a = self.saveA
        v = self.chekV_A(v, a, x)
        v = abs(v) * np.sign(x)
        a = abs(a) * np.sign(x)
        # print('calcX_V_A_Is:',t=', t, 'x=', x, 'v=', v, 'a=', a)
        if t > x / v + v / a:
            # print('biiig', x, v, t)
            return x1, 0, 0, False
        elif t > x / v:
            # print('biig', x, v, t)
            tMo = self.CalculateMottonTime(x, v, a)
            dt = tMo - t
            dx = a * dt**2 / 2
            dv = a * dt
            return x1 - dx, dv, -a, True
        elif t > v / a:
            # print('big', x, v, t)
            dt = t - v / a
            dx = v**2 / (2 * a) + dt * v
            return x0 + dx, v, 0, True
        else:
            # print('small', x, v, t)
            dt = t
            dx = a * dt**2 / 2
            dv = a * dt
            return x0 + dx, dv, a, True

    def chekV_A(self, v=None, a=None, x=None):
        # print('chek_va ', x, '  ', type(self.finX[-1]), '  ', self.finX[-2])
        if v == None:
            v = self.saveV
        if a == None:
            a = self.saveA
        if x == None:
            x = self.finX[-1] - self.finX[-2]
        a = abs(a)
        x = abs(x)
        v = abs(v)
        xbl = v**2 / a
        if x < xbl:
            return math.sqrt(x / xbl) * v
        else:
            return v


class MotorSystem():
    stopFlag = False

    def __init__(self, simulate=False, simulator=None, blocking=False):
        if simulate:
            motL = simulator.motorL
            motR = simulator.motorR
            motM = simulator.motorM
        else:
            print(dr.motor.list_available_devices())
            motL = dr.motor(90113196)
            motR = dr.motor(90113197)
            motM = dr.motor(90113195)

        self.motorL = Motor(motL, "motor L", blocking=blocking)
        self.motorR = Motor(motR, "motor R", blocking=blocking)
        self.motorM = Motor(motM, "motor M",
                            blocking=blocking)  # начальная позиция 60

        self.a_norm = self.motorL.a_norm
        self.v_norm = self.motorL.v_norm

        self.l0 = 104 + 200 - self.motorL.position_max - self.motorR.position_max  # милиметров  наименишея длина струны
        self.centrStart = 0
        self.x0 = None
        self.mosH = None
        self.dhFire = 0

        self.stopButton = ipywidgets.widgets.Button(
            description="Stop All Motors")
        self.stopButton.on_click(self.btnStopStartAll)
        self.stopButtonL = ipywidgets.widgets.Button(description="Stop " +
                                                     "motor L")
        self.stopButtonL.on_click(self.btnStopStartL)
        self.stopButtonR = ipywidgets.widgets.Button(description="Stop " +
                                                     "motor R")
        self.stopButtonR.on_click(self.btnStopStartR)
        self.stopButtonM = ipywidgets.widgets.Button(description="Stop " +
                                                     "motor M")
        self.stopButtonM.on_click(self.btnStopStartM)
        self.forgotButton = ipywidgets.widgets.Button(
            description="Forgot motion")
        self.forgotButton.on_click(self.ForgotMotion)

        self.funL_x, self.funR_x, self.xMax = caunter(lw=23,
                                                      rw=10,
                                                      Ltorch=1,
                                                      thetasdiv=1)

        def f(fun, xMax, x):

            if x < 0:
                return fun(0)
            elif x > xMax:
                return fun(xMax)
            else:
                return fun(x)

        self.L_x = lambda x: f(self.funL_x, self.xMax, x)
        self.R_x = lambda x: f(self.funR_x, self.xMax, x)
        self.lStart = self.L_x(0) / 2
        self.direction = 0
        # self.MotorsControlStart()

    def ForgotMotion(self, btn=None):
        self.motorL.FogotMotion()
        self.motorR.FogotMotion()
        self.motorM.FogotMotion()

    def btnStopStartAll(self, btn=None, flag=None):
        if flag != None:
            flag = MotorSystem.stopFlag
        self.stopButtonL.description = self.motorL.btnStopStart(flag)
        self.stopButtonR.description = self.motorR.btnStopStart(flag)
        self.stopButtonM.description = self.motorM.btnStopStart(flag)
        self.apdateStopFlag()

    def btnStopStartL(self, btn=None, flag=None):
        self.stopButtonL.description = self.motorL.btnStopStart(flag)
        self.apdateStopFlag()

    def btnStopStartR(self, btn=None, flag=None):
        self.stopButtonR.description = self.motorR.btnStopStart(flag)
        self.apdateStopFlag()

    def btnStopStartM(self, btn=None, flag=None):
        self.stopButtonM.description = self.motorM.btnStopStart(flag)
        self.apdateStopFlag()

    def apdateStopFlag(self):
        MotorSystem.stopFlag = self.motorL.stopFlag and self.motorR.stopFlag and self.motorM.stopFlag
        if self.stopFlag:
            self.stopButton.description = "Start All Motors"
        else:
            self.stopButton.description = "Stop All Motors"

    def Stop(self, mot=None):
        if mot == None:
            self.btnStopStartAll(flag=False)
        elif mot == 'L':
            self.btnStopStartL(flag=False)
        elif mot == 'R':
            self.btnStopStartR(flag=False)
        elif mot == 'M':
            self.btnStopStartM(flag=False)
        else:
            print("ms stop error name shoud be L, R or M   not ", mot)
            self.btnStopStartAll(flag=False)

    def Start(self, mot=None):
        if mot == None:
            self.btnStopStartAll(flag=True)
        elif mot == 'L':
            self.btnStopStartL(flag=True)
        elif mot == 'R':
            self.btnStopStartR(flag=True)
        elif mot == 'M':
            self.btnStopStartM(flag=True)
        else:
            print("ms stop error name shoud be L, R or M   not ", mot)
            self.btnStopStartAll(flag=True)

    def StopButton(self):
        grid = ipywidgets.GridspecLayout(1, 5)
        grid[0, 0] = self.stopButton
        grid[0, 1] = self.stopButtonL
        grid[0, 2] = self.stopButtonR
        grid[0, 3] = self.stopButtonM
        grid[0, 4] = self.forgotButton
        display(grid)

    def End(self, p1=0, p2=0, p3=0):
        print('ending')
        self.Stop()
        self.ForgotMotion()
        self.Start()
        self.motorL.MoveTo(p1)
        self.motorR.MoveTo(p2)
        self.motorM.MoveTo(p3)
        self.motorL.MoveTo(p1)
        self.motorR.MoveTo(p2)
        self.motorM.MoveTo(p3)
        # f = open(
        #     r"C:\Users\Fiber\Desktop\table_control_data\sistem\normalend.txt",
        #     'w')
        # f.write('0')
        # f.close()
        print('end')

    def IsEndWasGood(self):
        f = open(
            r"C:\Users\Fiber\Desktop\table_control_data\sistem\normalend.txt",
            'r')
        flag = f.read()
        itog = "OK now it's your problem"
        if flag != '0':
            a = sg.popup_ok_cancel('end was bad \npress any kay to ignor')
            if a == None or a == 'Cancel':
                return "end was bad"
        else:
            itog = "end was good!"
        f.close()
        f = open(
            r"C:\Users\Fiber\Desktop\table_control_data\sistem\normalend.txt",
            'w')
        f.write('1')
        f.close()
        return itog

    def Distance(self, xL=None, xR=None, useL0=True):
        if xL == None:
            xL = self.motorL.Getposition(memory=False)
        if xR == None:
            xR = self.motorR.Getposition(memory=False)
        l = 0
        if useL0:
            l = self.l0
        return self.motorL.position_max + self.motorR.position_max - xL - xR + l

    async def MoveToStart(self, L=None, dL=0, v=-1, a=-1, zapas=3):
        if (v == -1):
            v = self.v_norm
        if (a == -1):
            a = self.a_norm
        if L == None:
            L = self.L_x(0) / 2 + zapas
        self.lStart = L
        t1 = asyncio.create_task(
            self.motorL.MoveTo(self.motorL.position_max - L - dL, v, a))
        t2 = asyncio.create_task(
            self.motorR.MoveTo(self.motorR.position_max - L + dL, v, a))
        await asyncio.wait([t1, t2])

    async def Move(self, L, v=-1, a=-1, vdiff=0, da=0):
        if (v == -1):
            v = self.v_norm
        if (a == -1):
            a = self.a_norm
        flag = 0
        '''t = abs(L) / v + v / (2 * a)
        dx = vdiff * t

        if (a ** 2 * t ** 2 - 2 * a * (L + dx) <= 0):
            print("Invalid Move parameters \n")
            return -1
        vL = a * t - np.sqrt(a ** 2 * t ** 2 - 2 * a * (abs(L) - math.copysign(dx, L)))
        vR = a * t - np.sqrt(a ** 2 * t ** 2 - 2 * a * (abs(L) + math.copysign(dx, L)))
        '''
        v = self.motorR.chekV_A(v, a, L)
        coffL = (v - vdiff / 2) / v
        coffB = (v + vdiff / 2) / v
        t = self.motorL.CalculateMottonTime(x=L, v=v, a=a)
        sL = self.motorL.CalculateMottonDist(t=t, v=v * coffL, a=a * coffL)
        sB = self.motorL.CalculateMottonDist(t=t, v=v * coffB, a=a * coffB)

        while self.IsInMotion():
            await asyncio.sleep(0)
        xL = self.motorL.Getposition(analitic=False)
        xR = self.motorR.Getposition(analitic=False)
        # print("Move", sL/coffL, sB/coffB, L, vdiff, xL, xR)
        if L > 0:
            flag += self.motorL.Set_velocity(v=v * coffB, a=a * coffB - da)
            flag += self.motorR.Set_velocity(v=v * coffL, a=a * coffL + da)
            flag += self.motorL.Move_to_iner(xL - sB)
            flag += self.motorR.Move_to_iner(xR + sL)
        else:
            flag += self.motorL.Set_velocity(v=v * coffL, a=a * coffL + da)
            flag += self.motorR.Set_velocity(v=v * coffB, a=a * coffB - da)
            flag += self.motorR.Move_to_iner(xR - sB)
            flag += self.motorL.Move_to_iner(xL + sL)

        if (flag < 0):
            return -1

        while self.IsInMotion():
            await asyncio.sleep(0)
        return 0

    def calcX_L(self, lStart=None, centrStart=None, posL=None, posR=None):
        if posL == None:
            posL = self.motorL.Getposition(memory=False)
        if posR == None:
            posR = self.motorR.Getposition(memory=False)
        if lStart == None:
            lStart = self.lStart
        if centrStart == None:
            centrStart = self.centrStart
        pL = self.motorL.position_max - lStart - posL - centrStart
        pR = self.motorR.position_max - lStart - posR + centrStart
        x = pL + pR  # изменить
        L = (pL - pR) / 2 - centrStart
        return x, L

    def ResetBeforePull(self):
        self.stFl = False
        self.tStart = 0
        self.tStart1 = 0
        self.tFinish1 = 0
        self.tFinish = 0
        self.tact = 0
        self.downPos = self.motorM.position_min + 10
        # self.motorM.MoveTo(self.downPos)
        self.hFire = self.motorM.Getposition()

    async def LRoscillation():
        pass

    async def PulMove(self, v, a, dv, stFl):
        dt = 0  # рудимент пока похраним
        alf = dv / v
        x, L = self.calcX_L()
        # print("PulMove", "L=", L, "x=", x, "L_x(x)=", self.L_x(x) / 2, "alf=", alf)

        if self.direction == 0:  # движжение в положительном направление
            Xnew, dLnew = findEndPoint(x, L, alf, self.L_x, self.xMax)
            # print(dLnew, "  case0")
        else:  # движжение в отрицательном направление
            Xnew, dLnew = findEndPoint(x, -L, alf, self.L_x, self.xMax)
            dLnew = -dLnew
            # print(-dLnew, v, a, alf, dt, "  case1")

        t = v / a
        print('dLnew:', dLnew)
        dLnew += v * t / 2 * np.sign(dLnew)
        if Xnew < self.xMax and not stFl:
            await self.Move(dLnew, v, a, alf * v, dt)

        else:
            await self.Move(-L, v, a, 0, dt)
            t = self.motorR.CalculateMottonTime(L, v, a)
            Xnew = x + t * alf * v
            print("xMax=", self.xMax, "L_x(xMax)=",
                  self.L_x(self.xMax) / 2, "  x=", x, "  L=", L, 'xEnd=', Xnew,
                  'lEnd=',
                  self.L_x(Xnew) / 2)
            self.stFl = True
        self.direction = not self.direction

        self.tStart = Time.time()
        self.tStart1 = self.tStart + v / a
        self.tFinish = self.tStart + self.motorR.CalculateMottonTime() * 0.95
        self.tFinish1 = self.tFinish - v / a
        if self.tFinish1 < self.tStart1:
            self.tStart1 = self.tFinish1 = (self.tStart + self.tFinish) / 2
        self.tact += 1
        return self.stFl

    async def PulFireMove(self, aEnd, vEnd, vFon):
        # t0 = Time.time()
        while self.motorM.IsInMotion():
            await asyncio.sleep(0)
        t = Time.time()
        # tp = np.array([self.tStart, self.tStart1, self.tFinish1, self.tFinish])
        # tp2 = tp[t0 - tp > 0]
        # if len(tp2) > 0:
        #     m = t0 - np.amax(tp2)
        # else:
        #     m = None
        # print(self.tact, t - t0, t0 - self.tStart, tp - self.tStart)
        if t < self.tStart1:  # фаза ускорения
            dt = self.tStart1 - t
            # print('start:', self.tStart1 - self.tStart, dt)
            self.hFire += (self.tStart1 - self.tStart) * vFon
            hG = self.hFire - self.motorM.Getposition()
        elif self.tStart1 <= t < self.tFinish1:  # фаза равномерного движения
            dt = self.tFinish1 - t
            # print('move:', self.tFinish1 - self.tStart1, dt)
            self.hFire += (self.tFinish1 - self.tStart1) * vFon
            hG = self.hFire - self.motorM.Getposition()
        elif t < self.tFinish:  # фаза торможения
            dt = self.tFinish - t
            # print('stop:', self.tFinish - self.tFinish1, dt)
            self.hFire += (self.tFinish - self.tFinish1) * vFon
            h = self.motorM.CalculateMottonDist(dt, v=vEnd, a=aEnd)
            hG = self.hFire - h - self.motorM.Getposition()
        else:
            print('слишком поздний вызов PulFireMove')
            return
        htr = self.motorM.CalculateMottonDist(dt, v=1000, a=aEnd)
        hMax = self.motorM.CalculateMottonDist(dt, v=Motor.v_max, a=aEnd)
        hMax *= 0.9
        hG *= 0.9
        if hMax < abs(hG):
            self.motorM.Move(math.copysign(hMax, hG), v=Motor.v_max, a=aEnd)
        else:
            v = self.motorM.chekV_A(v=1000, a=aEnd, x=htr)
            dh = htr - abs(hG)
            dv = math.sqrt(dh / htr) * v
            # print(dt, htr, v, dv)
            self.motorM.Move(hG, v=v - dv, a=aEnd)

    def VforFireMove(self, goal, alf=6, Vmin=0.1, Vmax=10):
        h = goal - self.hFire
        sign = math.copysign(1, h)
        h = abs(h)
        tau = self.tFinish - self.tStart
        h0 = Vmin * tau
        if h > h0:
            V = Vmin + (h - h0) / tau / alf
        else:
            V = h / tau
        return min(V, Vmax) * sign

    def IsInMotion(self, all=False):
        if (self.motorR.IsInMotion() or self.motorL.IsInMotion()
                or (all and self.motorM.IsInMotion())):
            return True
        else:
            return False

    def Test(self):
        print('motorR test')
        self.motorR.Test()
        while self.IsInMotion(all=True):
            pass
        print('motorL test')
        self.motorL.Test()
        while self.IsInMotion(all=True):
            pass
        print('motorM test')
        self.motorM.Test()
        while self.IsInMotion(all=True):
            pass

    def Clear(self, n=-1):
        self.motorR.Clear(n)
        self.motorL.Clear(n)
        self.motorM.Clear(n)
