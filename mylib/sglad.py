from collections import deque
import numpy as np
import pandas as pd
import math


class sglad():
    ''' оконное и экспанинциальное сглаживание функции
    размер окна равен периуду движения моторчиков, тоесть 2 тактам
    и соответственно может меняться
    '''
    def __init__(self, tau_l=10, tau_t=10):
        # характерное время усреднения (для эксп сглаживания)
        self.a = tau_l  # значения
        self.b = tau_t  # производной
        self.Reset()

    def Reset(self):
        self.tacts = []
        self.data = deque()
        self.t1 = None
        self.t2 = None
        self.t = None
        self.t_last = None
        self.sum = 0
        self.n = 0
        self.mean = None
        self.level = 0
        self.trend = 0

    def NewPoint(self, x, t2):
        self.t2 = t2
        if len(self.tacts) > 0:
            self._add_data_to_que({'x': x, 't': t2})
        if len(self.tacts) == 4:
            p1 = self.tacts[1] - self.tacts[0]
            p2 = self.tacts[2] - self.tacts[1]
            p3 = self.tacts[3] - self.tacts[2]
            d = (self.t2 - self.tacts[1]) / p2
            periud = p2 + p1 + (p3 - p1) * d
            self.t1 = self.t2 - periud
            self.t = (self.t1 + self.t2) / 2

            self._remove_data_from_que()

            self.mean = self.sum / self.n
            self._doubleExpSmooth(self.mean, self.t)

    def _add_data_to_que(self, p):
        self.data.append(p)
        self.sum += p['x']
        self.n += 1

    def _remove_data_from_que(self):
        while True:
            p = self.data.popleft()
            if p['t'] >= self.t1: break
            self.sum -= p['x']
            self.n -= 1
        self.data.appendleft(p)

    def New_tact(self, t):  # добавляет время когда моторчик остановится
        if len(self.tacts) > 4:
            self.tacts.pop(0)

    def _doubleExpSmooth(self, x, t):
        if self.t_last is None:
            self.level = x
            self.trend = 0
            self.levelS = self.level
            self.levelN = 1
            self.trendS = self.trend
            self.trendN = 1
        else:
            dt = t - self.t_last
            ka = math.exp(-dt / self.a)
            kb = math.exp(-dt / self.b)
            self.levelS = ka * self.levelS + (x + dt * self.trend)
            self.levelN = ka * self.levelN + 1
            newlevel = self.levelS / self.levelN
            self.trendS = kb * self.trendS + (newlevel - self.level) / dt
            self.trendN = kb * self.trendN + 1
            self.trend = self.trendS / self.trendN
            self.level = newlevel
            print(self.level, self.levelS, self.levelN, dt, ka, self.a)
        self.t_last = self.t
