from random import random
import asyncio


async def reader(buf: list, cd: asyncio.Condition):
    while True:
        print(buf)
        asyncio.sleep(2)

async def writer(buf: list, cd: asyncio.Condition):
    for i in range(5):
        buf.append(i)
        asyncio.sleep(1.5)
    cd.notify_all()

async def main():
    condition = asyncio.Condition()
    tasks = []
    buf = []
    tasks.append(reader(buf, condition))
    tasks.append(writer(buf, condition))
    await asyncio.wait(tasks)


asyncio.run(main())