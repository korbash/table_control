import numpy as np

def www(a):
    print(1)
    yield a
    print(2)
    yield
    print(3)


for i in www(3):
    pass

# for i in range(5):
#     # print(i)
#     print(www())

tp = np.array([2,5,3,1])
tpg = tp[2.5 - tp < 0]
m = np.amax(tpg)
print(tpg,m)