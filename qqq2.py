from korbash_lib import Puller

print('start testing')
pl = Puller()
pl.ms.motorM.MoveTo(15)
pl.SetH_avto(quiet=False)
del pl
