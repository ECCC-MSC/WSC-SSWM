import pkg_resources

from PSPOL.pspol import pspolfil as ps_c
pth = 'PSPOL'
pre = ''

    
import numpy as np
import time
import pstats, cProfile

RS2 = pkg_resources.resource_filename(pth, pre+'data/RS2.npy')
PROF_FILE = pkg_resources.resource_filename(pth, pre+'data/Profile.prof')
RESULT = pkg_resources.resource_filename(pth, pre+'data/RS2_filtered.npy')

sar = np.load(RS2)
approximate_calibration = 4.370677e+03
power = (np.absolute(sar) / approximate_calibration)**2
ptot = np.sum(power, axis=0) 

phase = np.atleast_3d(sar[0,:,:] * np.conj(sar[1,:,:]))
phase = np.moveaxis(phase, 2,0)
sar =   np.concatenate( [np.absolute(sar), phase], axis=0 )


print("Running Filter on 500 x 500 array with window size 5")

cProfile.runctx("filt = ps_c(img=sar, P=ptot, numlook=1, winsize=5)", globals(), locals(), PROF_FILE)

s = pstats.Stats(PROF_FILE)
s.strip_dirs().sort_stats("time").print_stats()


np.save(RESULT, filt)

