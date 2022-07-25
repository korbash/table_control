import os
import numpy as np
import time
import math
import ipywidgets
from IPython import display
from ipywidgets import widgets
from IPython.display import display
from datetime import datetime
import PySimpleGUI as sg
import pandas as pd
from scipy import optimize
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from visualization import PlotDisplayer, Slider
import driwers as dr
import random


# переделать стоп button
def Save(data, dirSubName, name, dirName='C:\\Users\\Fiber\\Desktop\\table_control_data'):
    q = datetime.today()
    date = str(q.year) + '_' + str(q.month) + '_' + str(q.day) + '__' + str(q.hour) + '_' + str(q.minute)
    dirSubName = dirSubName.replace('DATE', date)
    name = name.replace('DATE', date)
    st = dirName + '\\' + dirSubName
    if not os.path.exists(st):
        os.mkdir(st)
    st = st + '\\' + name
    if os.path.exists(st):
        st += '(1)'
    data.to_csv(st, index=False)
    print('data saved')


def Everage(x, t, tau=1):
    dt = 0
    i = 0
    l = len(t)
    tmax = t[l - 1]
    xsr = 0
    while (dt < tau and i < l):
        i += 1
        dt = tmax - t[l - i]
        xsr += x[l - i]
    xsr = xsr / i
    return xsr


def Exp_everage(x, t, tau=1):
    dt = 0
    i = 0
    tmax = t[-1]
    xsr = 0
    csum = 0
    while (dt < tau * 10 and i < len(t)):
        i += 1
        dt = tmax - t[-i]
        c = math.exp(-dt / tau)
        xsr += x[-i] * c
        csum += c
    xsr = xsr / csum
    return xsr


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
         37.4940508, 49.76789206, 63.43683519, 78.09095269]) / 5
    Theta = interp1d(radius, thetas, kind='cubic')
    r = np.arange(rw, r0, dr)
    dz = Map(lambda x: 1 / float(Theta(x)), r)
    z = integr(dz, r)
    z = z[-1] - z
    inte = integr((r ** 2), z)
    # L = 1 / (r ** 2) * (rw ** 2 * (lw) + 2 * (-inte[-1] + inte))
    L = 1 / (r ** 2) * (rw ** 2 * (lw) + 2 * (-inte))
    x = 2 * z + L - L[-1]

    x = np.append(x, -0.1)
    r = np.append(r, r[-1])
    L = np.append(L, L[-1])
    R_x = interp1d(x, r, kind='cubic')
    L_x = interp1d(x, L, kind='cubic')
    xMax = x[0]
    return L_x, R_x, xMax


def findEndPoint(xSt, lSt, alf, L_x, xMax):
    if alf == 0:
        x = xSt
    else:
        x = optimize.root_scalar(lambda x: L_x(x) / 2 - ((x - xSt) / alf + lSt), bracket=[-20, xMax + 20],
                                 method='brentq').root
    return x, L_x(x) / 2 - lSt


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def Vdiff(R, L):
    # A = 28.0
    # V0 = 3
    r0 = 125.0 / 2
    v0 = 0.01 * 0.8 * 3
    vf = v0 * 2
    rl = r0 / (np.sqrt(vf / v0) - 1)
    ##  rl = 0
    L0 = 15

    ## return v0
    return v0 * (r0 + rl) ** 2 / (R + rl) ** 2 * (L / L0)


class ReadingDevise():
    def __init__(self, pr, name, dataB, weightCoef, zeroWeight=0):
        self.zeroWeight = zeroWeight
        self.weightCoef = weightCoef
        self.pr = pr
        self.name = name
        self.dataB = dataB

    def __del__(self):
        del self.pr

    def ReadValue(self, memory=True, tau=0, lastTau=0, inExsist=False, DataB=None, whith_std=False):
        #if DataB == None:
        #    DataB = self.dataB
        t0 = time.time()
        t = 0
        i = 0
        while t <= tau:
            x = self.pr.read()
            #print('x=', x)
            if x == 'problem':
                return 'problem'
            else:
                weight = x * self.weightCoef - self.zeroWeight
            #DataB.Wright({self.name: weight}, inExsist)
            t = time.time() - t0
            i += 1
        #j = 1
        #while DataB.l - i - j >= 0 and DataB.data.loc[DataB.l - i - j, 'time'] > t0 - lastTau:
        #  j += 1
        #j -= 1
        #eight = DataB.data.loc[range(DataB.l - i - j, DataB.l), self.name].mean()
        # if whith_std:
        #   std = DataB.data.loc[range(DataB.l - i - j, DataB.l), self.name].std()
        #    return (weight, std)
        #else:
            return weight

    def SetCoefficient(self, real_weight, tau=10):  # не проверена
        self.zeroCoef = real_weight / self.ReadValue(memory=False, tau=tau, DataB=DataBase())
        return self.zeroCoef

    def SetZeroWeight(self, tau=10, T=0):
        data = DataBase()
        self.zeroWeight += self.ReadValue(tau=tau, DataB=data) - T
        self.pogr = data.data[self.name].std()
        return self.zeroWeight

    def Test(self, n=5):
        w = np.array([])
        t1 = np.array([])
        t2 = np.array([])
        for i in range(n):
            t1 = np.append(t1, Time.time())
            w = np.append(w, self.ReadValue(DataB=DataBase()))
            t2 = np.append(t2, Time.time())
        print('value=', w)
        print('start time=', t1)
        print('fin time=', t2)


class Motor():
    v_norm = 5
    a_norm = 5
    v_max = 1000
    a_max = 3000

    def __init__(self, mot, motor_name, p_max=99):

        self.mot = mot
        self.name = motor_name
        self.position_max = p_max
        self.position_min = 0
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
            print("error in ", self.name, " -  v or a is not nuber v=", v, " a=", a)
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
            if v == 0: v = 0.1
            self.mot.set_velocity_parameters(0, a, v)
        return flag

    def Move_to_iner(self, x):
        flag = 0
        if math.isnan(x):
            print("error in ", self.name, " -  x isnt number", x, "  ", type(x))
            flag = -3
        if x > self.position_max:
            print("error in ", self.name, " -  position is bad x=", x, "max position=", self.position_max)
            flag = -1
            x = self.position_max
        if x < self.position_min:
            print("error in ", self.name, " -  position is bad x=", x, "min position=", self.position_min)
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
            self.mot.move_to(x)
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

    def Move(self, dp, v=v_norm, a=a_norm):
        while (self.IsInMotion()):
            pass
        p = self.Getposition(memory=False, motorNotMove=True)
        self.chekV_A(v, a, dp)
        self.Set_velocity(v, a)
        flag = self.Move_to_iner(p + dp)
        return flag

    def MoveTo(self, x, v=v_norm, a=a_norm):
        while (self.IsInMotion()):
            pass
        x0 = self.Getposition(memory=False, motorNotMove=True)
        v = self.chekV_A(v, a, x - x0)
        self.Set_velocity(v, a)
        flag = self.Move_to_iner(x)
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
        self.finX[-1] = self.saveX = self.Getposition(analitic=analitic, memory=False)

    def CheckPosition(self, pos):
        if (pos > self.position_max or pos < self.position_min):
            print("Invalid position")
            return -1
        else:
            return 0

    def CalculateMottonTime(self, x=None, v=None, a=None):  # исправить
        if x == None: x = self.finX[-1] - self.finX[-2]
        if v == None: v = self.saveV
        if a == None: a = self.saveA
        v = self.chekV_A(v, a, x)
        a = abs(a)
        x = abs(x)
        return x / v + v / a

    def CalculateMottonDist(self, t, v=None, a=None):
        if v == None: v = self.saveV
        if a == None: a = self.saveA
        v = abs(v)
        a = abs(a)
        if t < 0:
            print('CalculateMottonDist error t<0 t=', t)
            return 'error'
        if t > 2 * v / a:
            return (t - v / a) * v
        else:
            t = t / 2
            return a * t ** 2

    def calcX_V_A(self, t=None, x=None, v=None, a=None):
        x, v, a, isin = self.calcX_V_A_IsInMot(t=t, x=x, v=v, a=a)
        return x, v, a

    def calcX_V_A_IsInMot(self, t=None, x=None, v=None, a=None):
        if t == None: t = Time.time() - self.startsTime[-1]
        if x == None:
            x0 = self.finX[-2]
            x1 = self.finX[-1]
            x = x1 - x0
        else:
            x0 = 0
            x1 = x0 + x
        if x == 0: return x1, 0, 0, False
        if v == None: v = self.saveV
        if a == None: a = self.saveA
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
            dx = a * dt ** 2 / 2
            dv = a * dt
            return x1 - dx, dv, -a, True
        elif t > v / a:
            # print('big', x, v, t)
            dt = t - v / a
            dx = v ** 2 / (2 * a) + dt * v
            return x0 + dx, v, 0, True
        else:
            # print('small', x, v, t)
            dt = t
            dx = a * dt ** 2 / 2
            dv = a * dt
            return x0 + dx, dv, a, True

    def chekV_A(self, v=None, a=None, x=None):
        # print('chek_va ', x, '  ', type(self.finX[-1]), '  ', self.finX[-2])
        if v == None: v = self.saveV
        if a == None: a = self.saveA
        if x == None: x = self.finX[-1] - self.finX[-2]
        a = abs(a)
        x = abs(x)
        v = abs(v)
        xbl = v ** 2 / a
        if x < xbl:
            return math.sqrt(x / xbl) * v
        else:
            return v


class MotorSystem():
    stopFlag = False

    def __init__(self, simulate=False, simulator=None):
        if simulate:
            motL = simulator.motorL
            motR = simulator.motorR
            motM = simulator.motorM
        else:
            print(dr.motor.list_available_devices())
            motL = dr.motor(90113196)
            motR = dr.motor(90113197)
            motM = dr.motor(90113195)

        self.motorL = Motor(motL, "motor L")
        self.motorR = Motor(motR, "motor R")
        self.motorM = Motor(motM, "motor M", 39)  # начальная позиция 60

        self.a_norm = self.motorL.a_norm
        self.v_norm = self.motorL.v_norm

        self.l0 = 104 + 200 - self.motorL.position_max - self.motorR.position_max  # милиметров  наименишея длина струны
        self.centrStart = 0
        self.x0 = None
        self.mosH = None

        self.stopButton = widgets.Button(description="Stop All Motors")
        self.stopButton.on_click(self.btnStopStartAll)
        self.stopButtonL = widgets.Button(description="Stop " + "motor L")
        self.stopButtonL.on_click(self.btnStopStartL)
        self.stopButtonR = widgets.Button(description="Stop " + "motor R")
        self.stopButtonR.on_click(self.btnStopStartR)
        self.stopButtonM = widgets.Button(description="Stop " + "motor M")
        self.stopButtonM.on_click(self.btnStopStartM)
        self.forgotButton = widgets.Button(description="Forgot motion")
        self.forgotButton.on_click(self.ForgotMotion)

        self.funL_x, self.funR_x, self.xMax = caunter()

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
        f = open(r"C:\Users\Fiber\Desktop\table_control_data\sistem\normalend.txt", 'w')
        f.write('0')
        f.close()
        print('end')

    def IsEndWasGood(self):
        f = open(r"C:\Users\Fiber\Desktop\table_control_data\sistem\normalend.txt", 'r')
        flag = f.read()
        itog = "OK now it's your problem"
        if flag != '0':
            a = sg.popup_ok_cancel('end was bad \npress any kay to ignor')
            if a == None or a == 'Cancel':
                return "end was bad"
        else:
            itog = "end was good!"
        f.close()
        f = open(r"C:\Users\Fiber\Desktop\table_control_data\sistem\normalend.txt", 'w')
        f.write('1')
        f.close()
        return itog

    def Distance(self, xL=None, xR=None, useL0=True):
        if xL == None: xL = self.motorL.Getposition(memory=False)
        if xR == None: xR = self.motorR.Getposition(memory=False)
        l = 0
        if useL0: l = self.l0
        return self.motorL.position_max + self.motorR.position_max - xL - xR + l

    def MoveToStart(self, L=None, dL=0, v=-1, a=-1, zapas=3):
        if (v == -1):
            v = self.v_norm
        if (a == -1):
            a = self.a_norm
        if L == None: L = self.L_x(0) / 2 + zapas
        self.lStart = L
        self.motorL.MoveTo(self.motorL.position_max - L - dL, v, a)
        self.motorR.MoveTo(self.motorR.position_max - L + dL, v, a)

    def Move(self, L, v=-1, a=-1, vdiff=0, da=0):
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
            pass
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
        return 0

    def calcX_L(self, lStart=None, centrStart=None, posL=None, posR=None):
        if posL == None: posL = self.motorL.Getposition(memory=False)
        if posR == None: posR = self.motorR.Getposition(memory=False)
        if lStart == None: lStart = self.lStart
        if centrStart == None: centrStart = self.centrStart
        pL = self.motorL.position_max - lStart - posL - centrStart
        pR = self.motorR.position_max - lStart - posR + centrStart
        x = pL + pR  # изменить
        L = (pL - pR) / 2 - centrStart
        return x, L

    # def MotorsControlStart(self):
    #     self.phase = -1
    #     self.tStart = float('+inf')
    #     self.tStart1 = 0
    #     self.tFinish1 = 0
    #     self.tFinish = 0
    #     self.lastPhase = -1
    #     self.tEnd = 0
    #     self.tEnd2 = float('+inf')
    #     self.isUp = False
    #     self.stFl = False

    # def PulMotorsControl(self, v, a, dv, NewMosH, upFl=True, stFl=False, dhKof=0.5, ah=9):
    #     NewMosH += self.x0
    #     t = Time.time()
    #     downPos = self.motorM.position_max - 2
    #     tau = v / a
    #     dhMax = ah * tau ** 2 / 4
    #     self.lastPhase = self.phase
    #     if t < self.tStart:
    #         self.phase = -1
    #     elif t < self.tStart1:
    #         self.phase = 0
    #     elif t < self.tFinish1:
    #         self.phase = 1
    #     elif t < self.tFinish:
    #         self.phase = 2
    #     else:
    #         self.phase = 3
    #
    #     if self.phase == -1:
    #         if self.stFl:
    #             while self.motorM.IsInMotion():
    #                 Time.sleep(0.001)
    #             self.motorM.MoveTo(downPos)
    #             return -1
    #         else:
    #             if not self.IsInMotion():
    #                 self.stFl = self.PulMove(v, a, 0, stFl)
    #                 self.tEnd = t + self.motorR.CalculateMottonTime()
    #             if self.tEnd > self.tEnd2:
    #                 self.tStart = self.tEnd
    #             if not self.motorM.IsInMotion():
    #                 if upFl:
    #                     if not self.isUp:
    #                         self.motorM.MoveTo(NewMosH + dhMax * dhKof)
    #                         self.tEnd2 = t + self.motorM.CalculateMottonTime()
    #                         self.isUp = True
    #                 else:
    #                     if self.isUp:
    #                         self.motorM.MoveTo(downPos)
    #                         self.isUp = False
    #
    #     if self.phase == 3:
    #         if self.phase != self.lastPhase:
    #             if self.IsInMotion():
    #                 self.phase = self.lastPhase
    #                 return 0
    #             if not upFl or self.stFl:
    #                 self.tStart = float('+inf')
    #                 self.tStart1 = 0
    #                 self.tFinish1 = 0
    #                 self.tFinish = 0
    #                 self.tEnd = 0
    #                 self.tEnd2 = float('+inf')
    #             else:
    #                 self.stFl = self.PulMove(v, a, dv, stFl)
    #                 self.tStart = Time.time()
    #                 self.tStart1 = self.tStart + v / a
    #                 self.tFinish = self.tStart + self.motorR.CalculateMottonTime()
    #                 self.tFinish1 = self.tFinish - v / a
    #                 if self.tFinish1 < self.tStart1:
    #                     self.tStart1 = self.tFinish1 = (self.tStart + self.tFinish) / 2
    #                 dh1 = self.motorM.Getposition() - NewMosH
    #                 p = self.motorM.Getposition()
    #                 if abs(dh1) > dhMax:
    #                     dh1 = dhMax * np.sign(dh1)
    #                     print('you so fast, i think it is too math')
    #                 vh = 1 / 2 * ah * (tau - math.sqrt((ah * tau ** 2 - 4 * abs(dh1)) / ah))
    #                 while self.motorM.IsInMotion():
    #                     Time.sleep(0.001)
    #                 self.motorM.Move(-dh1, vh, ah)
    #
    #     if self.phase == 2:
    #         if self.phase != self.lastPhase:
    #             while self.motorM.IsInMotion():
    #                 Time.sleep(0.001)
    #             # p=self.motorM.Getposition()
    #             dh = dhMax * dhKof
    #             vh = 1 / 2 * ah * (tau - math.sqrt((ah * tau ** 2 - 4 * dh) / ah))
    #             self.motorM.Move(dh, vh, ah)
    #     return 0

    def PulMove(self, v, a, dv, stFl):
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
        dLnew += v * t / 2 * np.sign(dLnew)
        if Xnew < self.xMax and not stFl:
            self.Move(dLnew, v, a, alf * v, dt)

        else:
            self.Move(-L, v, a, alf * v, dt)
            t = self.motorR.CalculateMottonTime(L, v, a)
            Xnew = x + t * alf * v
            print("xMax=", self.xMax, "L_x(xMax)=", self.L_x(self.xMax) / 2, "  x=", x, "  L=", L, 'xEnd=', Xnew,
                  'lEnd=', self.L_x(Xnew) / 2)
            return True
        self.direction = not self.direction
        return False

    def IsInMotion(self, all=False):
        if (self.motorR.IsInMotion() or self.motorL.IsInMotion() or (all and self.motorM.IsInMotion())):
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


class Puller():
    kStr = 58591.17  # gram/mm * mm жёзскость пружины

    def __init__(self, simulate=False):
        if simulate:
            self.sim = simulator(104, self.kStr, -0.0075585384235655265)
            tg = self.sim.tg
            pm = self.sim.pm
        else:
            self.sim = None
            tg = dr.tensionGauge()
            time.sleep(0.001)
            pm = dr.powerMeter()
            time.sleep(0.001)
        df = pd.DataFrame()
        self.tg = ReadingDevise(tg, 'tension', df, weightCoef=-0.0075585384235655265)
        time.sleep(0.001)
        self.pm = ReadingDevise(pm, 'power', df, weightCoef=1000)
        self.ms = MotorSystem(simulate=simulate, simulator=self.sim)
        self.Ttrend = 0
        self.t_new = 0
        self.T_new = 0
        self.dv_new = 0
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
        Save(self.data, name='crude.csv', dirSubName='main_data\\DATE')
        Save(pd.DataFrame({
            'time': self.times,
            'kof': self.trueKmas,
            'sglKof': self.trueKmasGl,
        }), name='odrKof.csv', dirSubName='main_data\\DATE')

    def Clear(self):
        self.data = pd.DataFrame(
            columns=['time', 'tension', 'power', 'motorL', 'motorR', 'motorM', 'dt', 'x', 'vL', 'vR', 'vM', 'VdifRec',
                     'tensionWgl', 'tensionEXPgl'])
        self.sg = sglad()

    def Read(self, motoL=True, motoR=True, motoM=True):
        param = {}
        tSt = Time.time()
        param['motorL'] = self.ms.motorL.Getposition()
        param['motorR'] = self.ms.motorR.Getposition()
        param['motorM'] = self.ms.motorM.Getposition()
        param['power'] = self.pm.ReadValue()
        param['tension'] = self.tg.ReadValue()
        tFn = Time.time()
        param['dt'] = tFn - tSt
        param['x'], param['L'] = self.ms.calcX_L()
        param['vL'], param['aL'] = self.ms.motorL.calcX_V_A()[1: 3]
        param['vR'], param['aR'] = self.ms.motorR.calcX_V_A()[1: 3]
        param['vM'], param['aM'] = self.ms.motorM.calcX_V_A()[1: 3]
        param['pressure'] = param['tension'] * self.ms.R_x(0) ** 2 / self.ms.R_x(param['x']) ** 2
        param['VdifRec'] = Vdiff(self.ms.R_x(param['x']), self.ms.L_x(param['x']))
        param['time'] = Time.time()
        self.sg.New(param['tension'], param['vL'])
        # DataBase.Wright(param, inExsist=True)
        # DataBase.Apdete()
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
            # self.data.loc[self.sg.iGl, 'tensionEXPgl'] = T2
            self.data.loc[self.sg.iGl, 'tensionWgl'] = self.sg.wGl.iloc[-1]
            self.data.loc[range(self.sg.iGl, len(self.data)), 'tensionEXPgl'] = np.linspace(T2, Tnew, len(
                self.data) - self.sg.iGl)
            # self.data.loc[len(self.data) - 2, 'tensionEXPgl'] = np.nan
            # self.data.loc[len(self.data) - 1, 'tensionEXPgl'] = Tnew

    def Tprog(self, tau=0):
        return self.Ttrend * tau + self.data.loc[len(self.data) - 1, 'tensionEXPgl']

    def obrSvas(self, T, tau):
        dt = self.t_new - self.t_last
        dT = self.T_new - self.T_last
        progres = dT / dt * np.sign(T - self.T_last)
        dT2 = T - self.Tprog(tau / 2)
        trueK0 = max(self.dv_last / (progres * tau), 0.0001)
        self.trueKmas = np.append(self.trueKmas, trueK0)
        self.times = np.append(self.times, self.t_new)
        trueK = Exp_everage(self.trueKmas, self.times, tau=40)
        self.trueKmasGl = np.append(self.trueKmasGl, trueK)
        print(trueK0, trueK)
        self.dv = self.dv_last + trueK * dT2 / 4
        if self.dv < 0:
            self.dv = 0
        return self.dv

    def obrSvas2(self, T, tau, kof):
        dT = T - self.Tprog(tau * 3 / 2)
        self.dv = self.dv_last + kof * dT
        if self.dv < 0:
            self.dv = 0
        return self.dv

    def meser_param(self, dv):
        self.t_last = self.t_new
        self.t_new = Time.time()
        self.T_last = self.T_new
        self.T_new = self.Tprog(0)
        self.dv_last = self.dv_new
        self.dv_new = dv

    def SetW(self, wide, dw=0.1, k=None, tau=1, quiet=True):  ## T - tension, wIdeal, w_ideal
        if k == None: k = self.kStr / self.ms.Distance()
        # print(k, ' ', self.ms.Distance())
        t0 = Time.time()
        i = 1
        w = -100
        while abs(wide - w) > dw:
            w = self.tg.ReadValue(tau=tau)
            dx = (wide - w) / k
            if not quiet: print(w, '  ', w - wide, '  ', dx)
            self.ms.motorR.MoveTo(self.ms.motorR.Getposition(analitic=True) - dx, a=1)
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

    def SetH_avto(self, n=1, quiet=True, T=50, v1=1, v2=0.05, Tpr=6, startPos=None):
        if startPos == None: startPos = self.ms.motorM.Getposition()
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
            x0, a, b, data = self.SetH(xStart=xStart, xFin=xFin, v=v2, quiet=quiet, x0=x0, a=a, b=b)
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
                    return a * (x - x0) ** 2 + b
                else:
                    return b

            return f

        def errorFun(data_loc, a, b, x0):
            f = get_f(a, b, x0)
            data_loc['error'] = (data_loc['motorM'].apply(f) - data_loc['tension']) ** 2
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
            DataB.Wright({'motorM': self.ms.motorM.Getposition()}, inExsist=True)
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
        resalt = optimize.fmin(lambda x: errorFun(data, x[0], x[1], x[2]), np.array([a, b, x0]),
                               disp=False)  # errorFun(data, x[0], x[1], x[2])
        a = resalt[0]
        b = resalt[1]
        x0 = resalt[2]
        print('podgon', 'a=', a, '  b=', b, '  x0=', x0)
        # data['time'] = data['time'] - data['time'][0]
        data['fit'] = data['motorM'].apply(get_f(a, b, x0))
        if not quiet:
            dp = PlotDisplayer(mainParam='motorM', pl1=data[['motorM', 'error']],
                               pl2=data[['motorM', 'tension', 'fit']])
            dp.Show()
        return x0, a, b

    def SetH(self, xStart=11.5, xFin=9.5, v=0.05, quiet=True, a=4.5, b=50, x0=10.7):
        def get_f(a, b, x0):
            def f(x):
                if x < x0:
                    return a * (x - x0) ** 2 + b
                else:
                    return b

            return f

        def errorFun(data_loc, a, b, x0):
            f = get_f(a, b, x0)
            data_loc['error'] = (data_loc['motorM'].apply(f) - data_loc['tension']) ** 2
            return sum(data_loc['error'])

        i = 0
        DataB = DataBase()
        self.ms.motorM.MoveTo(xStart, 1, 1)
        self.ms.motorM.MoveTo(xFin, v, 1)
        while self.ms.motorM.IsInMotion():
            i += 1
            self.tg.ReadValue(DataB=DataB)
            DataB.Wright({'motorM': self.ms.motorM.Getposition()}, inExsist=True)
        self.ms.motorM.MoveTo(xStart, v, 1)
        while self.ms.motorM.IsInMotion():
            i += 1
            self.tg.ReadValue(DataB=DataB)
            DataB.Wright({'motorM': self.ms.motorM.Getposition()}, inExsist=True)
        data = DataB.data.copy()
        # DataBase.Clear(-i)

        resalt = optimize.fmin(lambda x: errorFun(data, x[0], x[1], x[2]), np.array([a, b, x0]), disp=False)
        a = resalt[0]
        b = resalt[1]
        x0 = resalt[2]
        print('podgon', 'a=', a, '  b=', b, '  x0=', x0)
        # data['time'] = data['time'] - data['time'][0]
        data['fit'] = data['motorM'].apply(get_f(a, b, x0))
        if not quiet:
            dp = PlotDisplayer(mainParam='motorM',
                               pl1=data[['motorM', 'error']], pl2=data[['motorM', 'tension', 'fit']])
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

    def PulMotorsControl(self, NewMosH, NewT, upFl=True, stFl=False, vsFl=False, dhKof=0.5, ah=9, Ki=0.1,Kp=0.1,Kd=0.1):
        self.Read()

        NewMosH += self.ms.x0
        t = Time.time()
        downPos = self.ms.motorM.position_max - 2
        tau = self.v / self.a
        dhMax = ah * tau ** 2 / 4
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
                            self.tEnd2 = t + self.ms.motorM.CalculateMottonTime()
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
                        self.meser_param(self.dv)
                        if vsFl and self.tact > 6:
                            self.dv = self.obrSvas2(NewT, self.tFinish - self.tStart, (Ki + Kp + Kd)/3 )
                        # print(self.v, self.a, self.dv, self.t_new, self.t_last, self.T_new, self.T_last)
                    self.stFl = self.ms.PulMove(self.v, self.a, self.dv, stFl)
                    self.sg.NewStTime()
                    self.tStart = Time.time()
                    self.tStart1 = self.tStart + self.v / self.a
                    self.tFinish = self.tStart + self.ms.motorR.CalculateMottonTime()
                    self.tFinish1 = self.tFinish - self.v / self.a
                    if self.tFinish1 < self.tStart1:
                        self.tStart1 = self.tFinish1 = (self.tStart + self.tFinish) / 2
                    dh1 = self.ms.motorM.Getposition() - NewMosH
                    p = self.ms.motorM.Getposition()
                    if abs(dh1) > dhMax:
                        dh1 = dhMax * np.sign(dh1)
                        print('you so fast, i think it is too math')
                    vh = 1 / 2 * ah * (tau - math.sqrt((ah * tau ** 2 - 4 * abs(dh1)) / ah))
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
                vh = 1 / 2 * ah * (tau - math.sqrt((ah * tau ** 2 - 4 * dh) / ah))
                self.ms.motorM.Move(dh, vh, ah)
        return 0

    def Test(self):
        print('tg test:')
        self.tg.Test()
        print('\npm test:')
        self.pm.Test()
        print('\nms test:')
        self.ms.Test()


class simulator():
    dtS = 0.01

    def __init__(self, L0, k0, weightCoff, l0=10):
        self.motorL = self.motSim()
        self.motorR = self.motSim()
        self.motorM = self.motSim()
        self.tg = self.devise(self.GetTention)
        self.pm = self.devise(self.GetPower)
        self.tg_tic = self.devise(self.GetTention2)
        self.mX = self.tic('x')
        self.mY = self.tic('y')
        self.mZ = self.tic('z')
        self.k0 = k0
        self.l0 = l0
        self.L0 = L0
        self.weightCoff = weightCoff
        self.tStart = Time.time()
        random.seed()

    def dist(self):
        return 200 - self.motorL.position - self.motorR.position

    def GetTention(self):
        if self.dist() < self.l0:
            ten = 0
        else:
            k = self.k0 / (self.dist() + self.L0)
            ten = (self.dist() - self.l0) * k
            # print(k,ten)
        Time.sleep(simulator.dtS)
        return ten / self.weightCoff

    def GetTention2(self):
        if self.mZ.getCoord()[0] > -10000:
            ten = 0
        else:
            ten = (-10000 - self.mZ.getCoord()[0]) * 0.001
        Time.sleep(simulator.dtS)
        ten += random.uniform(-0.005, 0.005)
        return ten / self.weightCoff

    def GetPower(self):
        Time.sleep(simulator.dtS)
        return 3.8 * 10 ** -5

    class tic():

        def __init__(self, name):
            self.name = name
            self.coord_last = 0
            self.coord = 0
            self.t_st = Time.time()
            self.v = 100

        def getCoord(self):
            dx = self.coord - self.coord_last
            dt = Time.time() - self.t_st
            if self.v * dt > abs(dx):
                return self.coord, False
            else:
                return self.coord_last + dx / self.v, True

        def move(self, value):
            self.coord_last = self.coord
            self.coord = self.coord_last + int(value)
            self.t_st=Time.time()

        def move_to(self, value):
            self.move(value - self.coord)


        def IsInMotion(self):
            return self.getCoord()[1]

    class motSim():
        def __init__(self):
            self.xStart = 0
            self.xFin = 0
            self.tStart = Time.time()
            self.a = 0
            self.v = 0

        def move_to(self, x):
            while self.is_in_motion:
                pass
            self.a = self.aNew
            self.v = self.vNew
            self.tStart = Time.time()
            self.xStart = self.xFin
            self.xFin = x
            Time.sleep(simulator.dtS)

        def stop_profiled(self):
            self.xFin = self.position
            self.tStart -= 10000

        def set_velocity_parameters(self, needless, a, v):
            self.vNew = v
            self.aNew = a

        @property
        def position(self):
            x, v, a, isMove = self.calcX_V_A()
            Time.sleep(simulator.dtS)
            return x

        @property
        def is_in_motion(self):
            x, v, a, isMove = self.calcX_V_A()
            Time.sleep(simulator.dtS)
            return isMove

        def calcX_V_A(self):
            t = Time.time() - self.tStart
            x0 = self.xStart
            x1 = self.xFin
            x = x1 - x0
            if x == 0: return x1, 0, 0, False
            v = self.v
            a = self.a
            v = self.chekV_A(v, a, x)
            v = abs(v) * np.sign(x)
            a = abs(a) * np.sign(x)
            if t > x / v + v / a:
                # print('biiig', x, v, t)
                return x1, 0, 0, False
            elif t > x / v:
                # print('biig', x, v, t)
                tMo = self.CalculateMottonTime(x, v, a)
                dt = tMo - t
                dx = a * dt ** 2 / 2
                dv = a * dt
                return x1 - dx, dv, -a, True
            elif t > v / a:
                # print('big', x, v, t)
                dt = t - v / a
                dx = v ** 2 / (2 * a) + dt * v
                return x0 + dx, v, 0, True
            else:
                # print('small', x, v, t)
                dt = t
                dx = a * dt ** 2 / 2
                dv = a * dt
                return x0 + dx, dv, a, True

        def chekV_A(self, v=None, a=None, x=None):
            # print('chek_va ', x, '  ', type(self.finX[-1]), '  ', self.finX[-2])
            if v == None: v = self.v
            if a == None: a = self.a
            if x == None: x = self.xFin - self.xStart
            a = abs(a)
            x = abs(x)
            v = abs(v)
            xbl = v ** 2 / a
            if x < xbl:
                return math.sqrt(x / xbl) * v
            else:
                return v

        def CalculateMottonTime(self, x=None, v=None, a=None):  # исправить
            if x == None: x = self.xFin - self.xStart
            if v == None: v = self.v
            if a == None: a = self.a
            v = self.chekV_A(v, a, x)
            a = abs(a)
            x = abs(x)
            return x / v + v / a

    class devise():
        def __init__(self, fun):
            self.read = fun


class sglad():
    def __init__(self, alpha=0.08, beta=0.2):
        self.dat = pd.Series()
        self.vDat = pd.Series()
        self.stP = np.array([])
        self.wGl = pd.Series()
        self.expGl = pd.Series()
        self.i = 0
        self.iGl = 0
        self.a = alpha
        self.b = beta
        self.stWindow = None
        self.endWindow = None
        self.readFl = True
        # self.levels = pd.Series()
        # self.trends = pd.Series()
        # self.iGLmas = np.array([0])

    # def otcat(self, i):
    #     self.i = i
    #     self.iGLmas = self.iGLmas[:i + 1]
    #     self.iGl = self.iGLmas[-1]
    #     self.level = self.levels[self.iGl]
    #     self.trend = self.trends[self.iGl]
    #     self.stP = self.stP[self.stP <= i]

    def New(self, x, v, progn=0, jastPoint=True):
        self.dat.loc[self.i] = x
        self.vDat.loc[self.i] = v
        # self.findPoint()
        resalt = (None, None)
        if len(self.stP) >= 2:
            self.periud = self.stP[-1] - self.stP[-2]
            if self.iGl + self.periud / 2 < self.i:
                wGLpoint = self.wGl.loc[self.iGl] = self.windowMean2(self.dat, self.iGl, self.periud)
                expGLpoint = self.double_exponential_smoothing(progn, jastPoint)
                self.iGl += 1
                resalt = (wGLpoint, expGLpoint)
        else:
            self.iGl = int((self.i + 1) / 2)
        self.i += 1
        # self.iGLmas = np.append(self.iGLmas, self.iGl)
        return resalt[0], resalt[1]

    def findPoint(self):
        if self.i == 0:
            self.flagSt = 0
        else:
            q = self.vDat.loc[self.i]
            flag = np.sign(self.vDat.loc[self.i] - self.vDat.loc[self.i - 1])
            if flag == 0 and self.flagSt < 0:
                self.stP = np.append(self.stP, self.i)
                print(self.i)
            self.flagSt = flag

    def NewStTime(self):
        if self.readFl:
            self.stP = np.append(self.stP, self.i)
        self.readFl = not self.readFl

    def windowMean(self, mas, i, l0):
        l = math.floor(l0 / 2)
        if mas.loc[i + l + 1:i + l + 1].sum() == 0:
            print('сам дурак')
        if self.stWindow == None:
            self.stWindow = i - l
            self.endWindow = i + l
            x = mas.loc[i - l:i + l].sum()
            self.xLast = x
            return x / (2 * l + 1)
        else:
            if i + l > self.endWindow:
                sumPl = mas.loc[self.endWindow + 1:i + l].sum()
            elif i + l == self.endWindow:
                sumPl = 0
            else:
                sumPl = -mas.loc[i + l + 1: self.endWindow].sum()
            if i - l > self.stWindow:
                sumMn = mas.loc[self.stWindow + 1:i - l].sum()
            elif i - l == self.stWindow:
                sumMn = 0
            else:
                sumMn = -mas.loc[i - l + 1: self.stWindow].sum()
            self.xLast += sumPl - sumMn
            self.stWindow = i - l
            self.endWindow = i + l
            mean = self.xLast / (2 * l + 1)
            return mean

    def windowMean2(self, mas, i, l0):
        l = math.floor(l0 / 2)
        if mas.loc[i + l + 1:i + l + 1].sum() == 0:
            print('сам дурак')
        x = mas.loc[i - l:i + l].sum()
        return x / (2 * l + 1)

    def double_exponential_smoothing(self, progn, jastPoint):
        if len(self.wGl) == 1:
            self.expGl.loc[self.iGl] = self.wGl[self.iGl]
            return
        elif len(self.wGl) == 2:
            self.level, self.trend = self.wGl[self.iGl - 1], self.wGl[self.iGl] - self.wGl[self.iGl - 1]
        value = self.wGl[self.iGl]
        last_level, self.level = self.level, self.a * value + (1 - self.a) * (self.level + self.trend)
        self.trend = self.b * (self.level - last_level) + (1 - self.b) * self.trend
        resalt = self.expGl.loc[self.iGl] = self.level + self.trend
        # self.levels[self.iGl], self.trends[self.iGl] = self.level, self.trend
        # print(self.iGl,self.level, self.trend)
        if jastPoint:
            self.expGl.loc[self.iGl + progn] = self.level + self.trend * progn
            if progn > 1 and self.iGl + progn - 1 in self.expGl.keys():
                self.expGl.drop(self.iGl + progn - 1, inplace=True)
        else:
            for i in range(progn):
                self.expGl.loc[self.iGl + i] = self.level + self.trend * i
        return resalt


class Time():
    t0 = 0
    t = 0
    isfreez = False

    def time():
        if Time.isfreez:
            return Time.t
        else:
            return time.time() - Time.t0

    def SetZeroTime(t0=0):
        Time.t0 = time.time() - t0
        Time.t = t0

    def freez():
        t = Time.time()
        Time.isfreez = True

    def unfreez():
        Time.t0 = time.time() - Time.t
        Time.isfreez = False

    def sleep(dt):
        if Time.isfreez:
            Time.t += dt
        else:
            time.sleep(dt)


class DataBase():
    def __init__(self):
        self.data = pd.DataFrame()
        self.d_data = pd.DataFrame()
        self.sg_data = pd.DataFrame()
        self.val = {}
        self.val_last = {}
        self.l = 0
        self.d_l = 0
        self.dt = 0.1
        self.t0 = 0
        self.smTen = sglad()

    def start(self):
        self.data = pd.DataFrame()
        self.d_data = pd.DataFrame()
        self.val = {}
        self.val_last = {}
        self.l = 0
        self.d_l = 0
        self.t0 = Time.time()

    def Apdete(self):
        t = self.t0 + (self.d_l) * self.dt
        endFl = False
        while t < self.data.loc[self.l - 1, 'time']:
            new = {}
            for name in self.d_data.keys():
                x1 = self.data.loc[self.val_last[name], name]
                x2 = self.data.loc[self.val[name], name]
                t1 = self.data.loc[self.val_last[name], 'time']
                t2 = self.data.loc[self.val[name], 'time']
                if t > t2:
                    endFl = True
                    break
                if t2 - t1 == 0:
                    new[name] = x2
                else:
                    new[name] = (x1 * (t2 - t) + x2 * (t - t1)) / (t2 - t1)
            if endFl:
                break
            new['time'] = t
            self.d_data.loc[self.d_l] = new
            if 'tension' in new and 'vR' in new:
                self.smTen.New(new['tension'], new['vR'])
                self.sg_data['tensionWgl'] = self.smTen.wGl
                self.sg_data['tensionEXPgl'] = self.smTen.expGl
            t += self.dt
            self.d_l += 1

    def Wright(self, elem, inExsist=False):
        t = Time.time()
        if inExsist:
            self.l -= 1
        elif not 'time' in elem.keys():
            elem['time'] = t

        for elName, elData in elem.items():
            if not elName in self.d_data.keys():
                self.d_data[elName] = elData
                self.val[elName] = self.l
            if (not inExsist) or (not elName in self.data.keys()) or pd.isna(self.data.loc[self.l, elName]):
                self.val_last[elName] = self.val[elName]
                self.val[elName] = self.l
            self.data.loc[self.l, elName] = elData
        self.l += 1

    # def Clear(n=None, mas=None):
    #     if mas == None:
    #         if n == None:
    #             mas = range(0, DataBase.l)
    #         else:
    #             if abs(n) > DataBase.l:
    #                 n = DataBase.l * int(np.sign(n))
    #             if n < 0:
    #                 mas = range(DataBase.l - n, DataBase.l)
    #             if n > 0:
    #                 mas = range(0, n)
    #     DataBase.data.drop(mas, inplace=True)
    #     DataBase.data.reset_index(drop=True, inplace=True)
    #     DataBase.l = DataBase.data.shape[0]
    #
    #     if DataBase.l == 0:
    #         DataBase.start()
    #         DataBase.smTen = sglad()
    #     else:
    #         l1 = int((DataBase.data['time'].iloc[-1] - DataBase.t0) // DataBase.dt)
    #         DataBase.d_data.drop(range(l1, DataBase.d_l))
    #         # DataBase.smTen.otcat(l1)

    def Save(self):
        Save(self.data, name='crude.csv', dirSubName='main_data\\DATE')
        Save(self.d_data, name='time_smooth.csv', dirSubName='main_data\\DATE')


class Tikalka():
    def __init__(self, simulate=False):
        if simulate:
            self.sim = simulator(104, 1, -0.0075585384235655265)
            tg = self.sim.tg_tic
            pm = self.sim.pm
            self.motX = self.sim.mX
            self.motY = self.sim.mY
            self.motZ = self.sim.mZ
        else:
            self.sim = None
            tg = dr.tensionGauge()
            pm = dr.powerMeter()
            self.motX = dr.tikalka_base('x')
            self.motY = dr.tikalka_base('y')
            self.motZ = dr.tikalka_base('z')
        self.tg = ReadingDevise(tg, 'tension', 'data_not_select', weightCoef=-0.0075585384235655265)
        self.pm = ReadingDevise(pm, 'power', 'data_not_select', weightCoef=1000)
        self.tg.SetZeroWeight(5)

    def FindZero(self, zapas=60, tochn=20, step=200, tau1=0.5):
        DataB = DataBase()
        Tpr = 5 * self.tg.pogr
        T = 0
        while (T - 0 < Tpr):
            self.motZ.move(-step)
            t = Time.time()
            while self.motZ.IsInMotion():
                self.tg.ReadValue(DataB=DataB)
            T = self.tg.ReadValue(lastTau=Time.time() - t, DataB=DataB)
            # print(self.motZ.getCoord()[0], T)
        while step > tochn:
            self.motZ.move(step)
            DataB = DataBase()
            while self.motZ.IsInMotion():
                self.tg.ReadValue(DataB=DataB)
            T = self.tg.ReadValue(tau=tau1, DataB=DataB)
            dT = DataB.data['tension'].std()
            # print(self.motZ.getCoord()[0], T)
            if not T - dT > 0:
                self.motZ.move(-step)
                step = step / 2
        self.motZ.move(zapas + 2 * step)

    def SetT(self, T, dT):
        dt_lim = 5
        DataB = DataBase()
        step = 50
        tau1 = 0.3
        Ttec = self.tg.ReadValue(DataB=DataB)
        sign = np.sign(T - Ttec)
        while (T - Ttec) * sign > 0:
            self.motZ.move(-sign * step)
            t = Time.time()
            while self.motZ.IsInMotion():
                self.tg.ReadValue(DataB=DataB)
            Ttec = self.tg.ReadValue(lastTau=Time.time() - t, DataB=DataB)
            # print(self.motZ.getCoord()[0], Ttec)
        while step >= 2:
            DataB = DataBase()
            t0 = Time.time()
            i = 0
            while i < 3 or (dTtec >= dT and abs(Ttec - T) < dTtec and Time.time() - t0 < dt_lim):
                self.tg.ReadValue(DataB=DataB)
                Ttec = DataB.data['tension'].mean()
                dTtec = DataB.data['tension'].std()
                i += 1
            print(self.motZ.getCoord()[0], Ttec, dTtec, Time.time(), step)
            if abs(Ttec - T) < dT:
                break
            if Time.time() - t0 > dt_lim:
                print('time limit problem')
                break
            sign = np.sign(T - Ttec)
            step = step // 2 + step % 2
            self.motZ.move(-step * sign)
        return dTtec

    def meserFixT(self, T, dT, tau):
        self.SetT(T, dT)
        #Time.sleep(1)
        P, dP = self.pm.ReadValue(tau=tau, whith_std=True)
        self.FindZero(tau1=0.3)
        return P, dP

    def Golden_Section_Method(self, y, eps, T, dT, tau0=1):
        tau1 = 10
        i = 0
        coef = (math.sqrt(5) - 1) / 2
        coef2 = (math.sqrt(5) + 1) / 2
        coef3 = 1 - coef2
        y0=self.motY.coord

        sc, poc = sa, pa = self.meserFixT(T, dT, tau0)
        c=a=y0
        self.motY.move(y)
        d=b=a+y
        sd, pod = sb, pb = self.meserFixT(T, dT, tau0)
        if sb <= sa:
            while sb <= sa:
                y *= coef2
                a, sa, pa = c, sc, poc
                c, sc, poc = b, sb, pb
                b+=y
                self.motY.move_to(b)
                sb, pb = self.meserFixT(T, dT, tau0)
                i+=1
            d=a+y
            self.motY.move_to(d)
            sd, pd = self.meserFixT(T, dT, tau0)
        else:
            self.motY.move(-y)
            while sb > sa:
                y *= coef2
                sb, pb = sd, pod
                sd, pod = sa, pa
                a -= y
                sa, pa = self.meserFixT(T, dT, tau0)
                i-=1
            c = b - y
            self.motY.move_to(c)
            sc, pc = self.meserFixT(T, dT, tau0)

        pogr = max(poc, pod)
        while y > eps:
            if sd < sc:
                b = d
                d = c
                c = b - (b - a) * coef
                sd = sc
                self.motY.move_to(c)
                sc, poc = self.meserFixT(T, dT, tau0)
            else:
                a = c
                c = d
                d = a + (b - a) * coef
                sc = sd
                self.motY.move_to(d)
                sd, pod = self.meserFixT(T, dT, tau0)
            # print(("     {0:.0f}    || {1:.4f} || {2:.4f}   || {3:.4f} || {4:.4f}").format(iteration - 1, x_min,
            #                                                                                function_f(x_min), x_max,
            #                                                                                function_f(x_max)))

    def Shup(self, Tmax):
        data = DataBase()
        step = 100
        i = 0
        T = self.tg.ReadValue(DataB=data)
        self.pm.ReadValue(DataB=data, inExsist=True)
        while T < Tmax:
            self.motZ.move(-step)
            while self.motZ.IsInMotion():
                T = self.tg.ReadValue(DataB=data)
                self.pm.ReadValue(DataB=data, inExsist=True)
            i += 1
        self.tg.ReadValue(DataB=data)
        self.pm.ReadValue(DataB=data, inExsist=True)
        for j in range(i):
            self.motZ.move(step)
            while self.motZ.IsInMotion():
                self.tg.ReadValue(DataB=data)
                self.pm.ReadValue(DataB=data, inExsist=True)
        return data

    # def MeserProf(self):
