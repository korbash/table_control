import time
import math
import numpy as np
import asyncio


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


def Exp_average(x, t, tau=1):
    dt = 0
    i = 0
    tmax = t[-1]
    xsr = 0
    csum = 0
    dt = tmax - t[::-1]
    dt = dt[dt < tau * 100]
    c = np.exp(-dt / tau)
    xsr = np.sum(x[::-1] * c) / np.sum(c)


    # while (dt < tau * 10 and i < len(t)):
    #     i += 1
    #     dt = tmax - t[-i]
    #     c = np.exp(-dt / tau)
    #     xsr += x[-i] * c
    #     csum += c
    # xsr = xsr / csum
    return xsr

class Time():
    t0 = 0
    t = 0
    isfreeze = False

    def time():
        if Time.isfreeze:
            return Time.t
        else:
            return time.time() - Time.t0

    def SetZeroTime(t0=0):
        Time.t0 = time.time() - t0
        Time.t = t0

    def freeze():
        t = Time.time()
        Time.isfreeze = True

    def unfreeze():
        Time.t0 = time.time() - Time.t
        Time.isfreeze = False

    def sleep(dt):
        if Time.isfreeze:
            Time.t += dt
        else:
            time.sleep(dt)

class ReadingDevice():
    def __init__(self, pr, name, weightCoef, zeroWeight=0):
        self.zeroWeight = zeroWeight
        self.weightCoef = weightCoef
        self.pr = pr
        self.name = name
        # self.dataB = dataB
        self.read = self.pr.read
        self.pogr = 0

    def __del__(self):
        del self.pr

    async def ReadValue(self, tau=0, type='ever'):
        if tau==0:
            await asyncio.sleep(0)
            x = self.read()
            await asyncio.sleep(0)
            return self.weightCoef * x - self.zeroWeight
        x = []
        t = []
        await asyncio.sleep(0)
        t0 = Time.time()
        await asyncio.sleep(0)
        t1 = t0
        while t1 - t0 <= tau:
            await asyncio.sleep(0)
            x += [self.read()]
            t1 = Time.time()
            t += [t1]
        x = np.array(x)
        t = np.array(t)
        await asyncio.sleep(0)
        x = x * self.weightCoef - self.zeroWeight
        await asyncio.sleep(0)
        if type == 'ever':
            await asyncio.sleep(0)
            return np.mean(x)
        elif type == 'ever_std':
            await asyncio.sleep(0)
            return np.mean(x), np.std(x)
        elif type == 'exp':
            await asyncio.sleep(0)
            return Exp_average(x, t, tau)
        else:
            print('reading devise error unpossible type')

    def SetCoefficient(self, real_weight, tau=10):  # не проверена
        self.zeroCoef = real_weight / self.ReadValue(tau=tau)
        return self.zeroCoef

    async def SetZeroWeight(self, tau=10, T=0):
        Tnew, dT = await self.ReadValue(tau=tau, type='ever_std')
        self.zeroWeight += Tnew - T
        self.pogr = dT
        return Tnew - T

    def Test(self, n=5):
        t0 = Time.time()
        w = np.array([])
        t1 = np.array([])
        t2 = np.array([])
        for i in range(n):
            t1 = np.append(t1, Time.time())
            w = np.append(w, self.ReadValue())
            t2 = np.append(t2, Time.time())
        print(self.name, ' testing')
        print('value=', w, '  start time=', t1 - t0, '  fin time=', t2 - t0)
