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

test = float('inf')
test2 = float('+inf')
test2 = float('-inf')
print(test > 1000)
print(test < -1000)