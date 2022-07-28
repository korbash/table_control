import time
import math
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d
from .sglad import sglad


def caunter(r0=62.5, Ltorch=0.6, lw=30, rw=20, dr=1):
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
    ]) / 5
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
    R_x = interp1d(x, r, kind='cubic')
    L_x = interp1d(x, L, kind='cubic')
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


def Save(data,
         dirSubName,
         name,
         dirName='C:\\Users\\Fiber\\Desktop\\table_control_data'):
    q = datetime.today()
    date = str(q.year) + '_' + str(q.month) + '_' + str(q.day) + '__' + str(
        q.hour) + '_' + str(q.minute)
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
    return v0 * (r0 + rl)**2 / (R + rl)**2 * (L / L0)


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
            if (not inExsist) or (not elName in self.data.keys()) or pd.isna(
                    self.data.loc[self.l, elName]):
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


class ReadingDevise():
    def __init__(self, pr, name, dataB, weightCoef, zeroWeight=0):
        self.zeroWeight = zeroWeight
        self.weightCoef = weightCoef
        self.pr = pr
        self.name = name
        self.dataB = dataB
        self.read = self.pr.read

    def __del__(self):
        del self.pr

    def ReadValue(self,
                  memory=True,
                  tau=0,
                  lastTau=0,
                  inExsist=False,
                  DataB=None,
                  whith_std=False):
        if DataB == None:
            DataB = self.dataB
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
            DataB.Wright({self.name: weight}, inExsist)
            t = time.time() - t0
            i += 1
        j = 1
        while DataB.l - i - j >= 0 and DataB.data.loc[DataB.l - i - j,
                                                      'time'] > t0 - lastTau:
            j += 1
        j -= 1
        eight = DataB.data.loc[range(DataB.l - i - j, DataB.l),
                               self.name].mean()
        if whith_std:
            std = DataB.data.loc[range(DataB.l - i - j, DataB.l),
                                 self.name].std()
            return (weight, std)
        else:
            return weight

    def SetCoefficient(self, real_weight, tau=10):  # не проверена
        self.zeroCoef = real_weight / self.ReadValue(memory=False, tau=tau)
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
