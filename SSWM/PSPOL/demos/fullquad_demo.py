import pkg_resources
import matplotlib.pyplot as plt

try:
    from PSPOL.pspol import pspolfil as ps_c
    pth = 'SSWM'
    pre = ''
except:
    from SSWM.PSPOL.pspol import pspolfil as ps_c
    pth= "SSWM"
    pre = 'PSPOL/'
    
import numpy as np
RS2FQ = pkg_resources.resource_filename(pth, pre+'data/RS2_FQ.npy')

sar = np.load(RS2FQ)
approximate_calibration = 4.370677e+03
power = (np.absolute(sar) / approximate_calibration)**2
ptot = np.sum(power, axis=0, dtype='float64') 

phase = np.atleast_3d(sar[0,:,:] * np.conj(sar[2,:,:]))
phase = np.moveaxis(phase, 2,0)
sar =   np.array(np.concatenate( [np.absolute(sar), phase], axis=0), dtype='complex128')

filt = ps_c(img=sar, P=ptot, numlook=1, winsize=5)



def rescale(img):
    res = np.array(img, dtype=np.float32)
    p = np.percentile(img, (2,98))
    res = img / (p[1] / 255)
    res[np.greater(res, 255)] = 255
    return(np.array(res, dtype='int16'))
    

fig, ((a,b),(c,d)) = plt.subplots(2, 2, figsize=(15, 15))
a.imshow(rescale(np.absolute(np.moveaxis(sar[0:3,:,:], 0,2)))) # hsv is cyclic, like angles
a.set_title('HH-HV-VV')
a.set_axis_off()
b.imshow(rescale(np.absolute(np.moveaxis(filt[0:3,:,:], 0,2)))) # hsv is cyclic, like angles
b.set_title('HH-HV-VV')
b.set_axis_off()

c.imshow(rescale(np.angle(sar[4,:,:])), cmap='hsv') # hsv is cyclic, like angles
c.set_title('HH-VV Phase difference')
c.set_axis_off()
d.imshow(rescale(np.angle(filt[4,:,:])), cmap='hsv') # hsv is cyclic, like angles
d.set_title('HH-VV Phase difference')
d.set_axis_off()
fig.show()
   