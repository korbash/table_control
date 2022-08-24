from collections import deque

d = deque(range(10))
# d2 = d.reverse()
for i, p in enumerate(reversed(d)):
    print(i, p)

# it = iter(d)
# # try: d.popleft()
# # except IndexError: pass

# print(next(it))
# d.popleft()
# print(next(it))
