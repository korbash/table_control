import random
import math
import numpy as np
from .utilities import Time


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
        return 3.8 * 10**-5

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
            self.t_st = Time.time()

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
            if v == None: v = self.v
            if a == None: a = self.a
            if x == None: x = self.xFin - self.xStart
            a = abs(a)
            x = abs(x)
            v = abs(v)
            xbl = v**2 / a
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