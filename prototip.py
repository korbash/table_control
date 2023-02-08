from mylib import PlotDisplayer, Slider, Time
import pandas
import asyncio
import math
import random
import time
from bokeh.io import push_notebook


class director:

    def __init__(self, pd: PlotDisplayer, sl: Slider) -> None:
        self.pd = pd
        self.sl = sl
        self.buff = 2
        self.tasks = []

    async def __aenter__(self, *args):
        print('args:', args)
        self.t0 = time.time()
        self.data = pandas.DataFrame([{'time': 0.0, 'power': 0.0}])
        self.pd.Show()
        self.tasks.append(asyncio.create_task(self.read_control()))
        self.tasks.append(asyncio.create_task(self.plotter()))
        self.tasks.append(asyncio.create_task(self.reader()))
        return self

    async def __aexit__(self, *args):
        for t in self.tasks:
            t.cancel()

    async def plotter(self):
        while True:
            print('plotted')
            self.pd.Apdate(for_all=self.data)
            push_notebook()
            await asyncio.sleep(0.3)

    async def read_control(self):
        while True:
            await asyncio.sleep(0.1)

    async def reader(self):
        while True:
            print('readed')
            new = pandas.DataFrame([{
                'time': time.time() - self.t0,
                'power': random.random()
            }])
            self.data = pandas.concat([self.data, new], ignore_index=True)
            # print(self.data)
            await asyncio.sleep(0.1)

    def add_new_val(self, v):
        self.buff = v


async def sensor(n=100, p=2):

    for i in range(n):
        x = (i / n) * 2 * math.pi
        await asyncio.sleep(1)
        yield (2 + math.sin(x)) * 2


async def main():
    d = director()
    async for res in sensor():
        d.add_new_val(res)


if __name__ == '__main__':
    asyncio.run(main())