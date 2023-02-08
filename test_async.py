import asyncio
import math


class director:

    def __init__(self) -> None:
        self.q = asyncio.Queue()
        self.q.put_nowait(0)
        self.buff = 2
        self.tasks = []
        self.tasks.append(asyncio.create_task(self.reader()))
        self.tasks.append(asyncio.create_task(self.controller()))

    async def controller(self):
        while True:
            await self.move(self.buff)

    async def reader(self):
        while True:
            await asyncio.sleep(0.4)
            print(self.buff)

    async def move(self, t):
        print('start_move')
        await asyncio.sleep(t)
        print('stoped')

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
