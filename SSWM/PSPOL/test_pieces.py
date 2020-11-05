import pkg_resources
from SSWM.PSPOL.pspol import pspolfil_memsafe as ps_m
from SSWM.PSPOL.pspol import pspolfil as ps_c
import numpy as np
import time
import pstats, cProfile

DATA_PATH = pkg_resources.resource_filename('SSWM', 'data/')
RS2 = pkg_resources.resource_filename('SSWM', 'PSPOL/data/RS2.npy')
PROF_FILE = pkg_resources.resource_filename('SSWM', 'PSPOL/data/Profile.prof')
print('loading')
sar = np.load(RS2)
approximate_calibration = 4.370677e+03
power = (np.absolute(sar) / approximate_calibration)**2
ptot = np.sum(power, axis=0) 

print("Running Filter on 500 x 500 array with window size 7")

cProfile.runctx("ps_m(img=np.absolute(sar), P=ptot, numlook=1, winsize=5, pieces=5)", globals(), locals(), PROF_FILE)

#sar = np.absolute(sar)
#ps_c(img=sar, P=ptot, numlook=1, winsize=5)

#part = sar.copy()[:,:,range(400,450)]
#ptotpart = ptot.copy()[:,range(400,450)]
#print(part.shape)
#print(ptotpart.shape)
#ps_c(img=part, P=ptotpart, numlook=1, winsize=5)

#s = pstats.Stats(PROF_FILE)
#s.strip_dirs().sort_stats("time").print_stats()
