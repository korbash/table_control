from mylib import *

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
        self.tg = ReadingDevise(tg,
                                'tension',
                                'data_not_select',
                                weightCoef=-0.0075585384235655265)
        self.pm = ReadingDevise(pm,
                                'power',
                                'data_not_select',
                                weightCoef=1000)
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
            while i < 3 or (dTtec >= dT and abs(Ttec - T) < dTtec
                            and Time.time() - t0 < dt_lim):
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
        y0 = self.motY.coord

        sc, poc = sa, pa = self.meserFixT(T, dT, tau0)
        c = a = y0
        self.motY.move(y)
        d = b = a + y
        sd, pod = sb, pb = self.meserFixT(T, dT, tau0)
        if sb <= sa:
            while sb <= sa:
                y *= coef2
                a, sa, pa = c, sc, poc
                c, sc, poc = b, sb, pb
                b += y
                self.motY.move_to(b)
                sb, pb = self.meserFixT(T, dT, tau0)
                i += 1
            d = a + y
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
                i -= 1
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