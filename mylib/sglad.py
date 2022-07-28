import numpy as np
import pandas as pd
import math


class sglad():
    def __init__(self, alpha=0.08, beta=0.2):
        self.dat = pd.Series(dtype='float64')
        self.vDat = pd.Series(dtype='float64')
        self.stP = np.array([])
        self.wGl = pd.Series(dtype='float64')
        self.expGl = pd.Series(dtype='float64')
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
        wGLpoint = None
        expGLpoint = None
        if len(self.stP) >= 2:
            self.periud = self.stP[-1] - self.stP[-2]
            if self.iGl + self.periud / 2 < self.i:
                wGLpoint = self.wGl.loc[self.iGl] = self.windowMean2(
                    self.dat, self.iGl, self.periud)
                expGLpoint = self.double_exponential_smoothing(
                    progn, jastPoint)
                self.iGl += 1
        else:
            self.iGl = int((self.i + 1) / 2)
        self.i += 1
        # self.iGLmas = np.append(self.iGLmas, self.iGl)
        return wGLpoint, expGLpoint

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
                sumPl = -mas.loc[i + l + 1:self.endWindow].sum()
            if i - l > self.stWindow:
                sumMn = mas.loc[self.stWindow + 1:i - l].sum()
            elif i - l == self.stWindow:
                sumMn = 0
            else:
                sumMn = -mas.loc[i - l + 1:self.stWindow].sum()
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
            self.level, self.trend = self.wGl[
                self.iGl - 1], self.wGl[self.iGl] - self.wGl[self.iGl - 1]
        value = self.wGl[self.iGl]
        last_level, self.level = self.level, self.a * value + (1 - self.a) * (
            self.level + self.trend)
        self.trend = self.b * (self.level - last_level) + (1 -
                                                           self.b) * self.trend
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