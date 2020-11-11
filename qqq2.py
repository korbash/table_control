from korbash_lib import Tikalka, ReadingDevise, Time
import driwers as dr

# pm = dr.powerMeter()
tg = dr.tensionGauge()
# pm = ReadingDevise(pm, 'power', weightCoef=1000)
tg = ReadingDevise(tg, 'tension', weightCoef=-0.0075585384235655265)
# print(pm.ReadValue())
tk = Tikalka()
print('yra')
tg.SetZeroWeight(5)
while (1):
    Time.sleep(0.1)
    print(tg.ReadValue())
    step = input()
    if step=='stop':
        break
    elif step==' ':
        step=100
    else:
        step=int(step)
    tk.move_relative(1, step)
