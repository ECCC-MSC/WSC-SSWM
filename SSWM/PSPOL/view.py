import pkg_resources
import numpy as np
import matplotlib.pyplot as plt

RS2 = pkg_resources.resource_filename('PSPOL', 'data/RS2.npy')
RS2F = pkg_resources.resource_filename('PSPOL', 'data/RS2_filtered.npy')

Rin = np.load(RS2)
Rout = np.load(RS2F)

# create covariance matrix to look at phase differences
phasein = np.atleast_3d(Rin[0,:,:] * np.conj(Rin[1,:,:]))
phasein = np.moveaxis(phasein, 2,0)




def rescale(img):
    res = np.array(img, dtype=np.float32)
    p = np.percentile(img, (2,98))
    res = img / (p[1] / 255)
    res[np.greater(res, 255)] = 255
    return(np.array(res, dtype='int16'))
    
    

fig, (hv, hvf) = plt.subplots(1, 2, figsize=(15, 15))
hv.imshow(rescale(np.angle(phasein[0,:,:])), cmap='hsv') # hsv is cyclic, like angles
hv.set_title('HH-HV Phase difference')
hv.set_axis_off()
hvf.imshow(rescale(np.angle(Rout[2,:,:])),  cmap='hsv') # hsv is cyclic, like angles
hvf.set_title('Filtered HH-HV Phase difference')
hvf.set_axis_off()
fig.show()
   
    
fig, (hh, hhf) = plt.subplots(1, 2, figsize=(15, 15))
hh.imshow(rescale(np.absolute(Rin[0,:,:])), cmap='gray') # hsv is cyclic, like angles
hh.set_title('HH')
hh.set_axis_off()
hhf.imshow(rescale(np.absolute(Rout[0,:,:])),  cmap='gray') # hsv is cyclic, like angles
hhf.set_title('Filtered HH')
hhf.set_axis_off()
fig.show()


fig, (hv, hvf) = plt.subplots(1, 2, figsize=(15, 15))
hv.imshow(rescale(np.absolute(Rin[1,:,:])), cmap='gray') # hsv is cyclic, like angles
hv.set_title('HV')
hv.set_axis_off()
hvf.imshow(rescale(np.absolute(Rout[1,:,:])),  cmap='gray') # hsv is cyclic, like angles
hvf.set_title('Filtered HV')
hvf.set_axis_off()
fig.show()

