#
# Filters

#NOTE: Nothing in this file is used anymore, we use PSPOL. Not sure what is better
#
from osgeo import gdal
import numpy as np

from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance, label
from SSWM.preprocess.preutils import cloneRaster


# https://stackoverflow.com/questions/4959171/improving-memory-usage-in-an-array-wide-filter-to-avoid-block-processing

def lee_filter2(img, window = (3,3)):
    """ Apply a Lee filter to a numpy array. Does not modify original.
    
    Code is based on:
    https://stackoverflow.com/questions/39785970/speckle-lee-filter-in-python
    
    PCI implementation is found at
    http://www.pcigeomatics.com/geomatica-help/references/pciFunction_r/python/P_fle.html
    
    
    *Parameters*
    
    img : numpy array  
        Array to which filter is applied
    window : int
        Size of filter
    
    *Returns*
    
    array 
        filtered array
    """
    
    img_mean = uniform_filter(img, window)
    img_sqr = np.square(img)
    
    img_sqr_mean = uniform_filter(img_sqr, window)
    img_variance = img_sqr_mean - img_mean**2

    overall_variance = variance(img)

    img_weights = img_variance / (img_variance + overall_variance)
    img_output = img_mean + img_weights * (img - img_mean)
    
    return img_output
    
def lee_filter(img, window = (5,5)):
    """ Apply a Lee filter to a numpy array. Modifies original array
     
    *Parameters*
    
    img : numpy array  
        Array to which filter is applied
    window : int
        Size of filter
    
    """
    d = np.array(img, dtype='float32')
    img_variance, img_mean = moving_window_sd(d, window, return_mean=True, return_variance=True)
    
    lbl, nlbl = label(d)
    overall_variance = variance(d, lbl)
    
    # set overall = overall_var + img_var
    overall_variance = np.add(overall_variance, img_variance) 
    
    # set d = img - img_mean
    np.subtract(d, img_mean, out=d)
    
    # set  img_variance = img_variance / overall_var + img_var = weights
    np.divide(img_variance, overall_variance, out=img_variance)
    del overall_variance
    
    # set d = weights * img - img_mean
    np.multiply(img_variance, d, out=d)
    
    np.add(img_mean, d, out=d) # d = img_mean + weights * img - img_mean

    return d
    
def enhanced_lee_filter(d, rg_win, az_win, nlooks, damp=1):
    """PCI-style Enhanced Lee filter after Lopes, 1990."""

    # Scalar parameters
    window = [az_win, rg_win]
    cu = np.sqrt(1 / nlooks)
    cmax = np.sqrt(1 + 2 / nlooks)

    # For each polarisation
    for i in range(len(d.shape) -1):

        # Compute the spatial average, standard deviation and coefficient of variation
        ci, im = moving_window_sd2(d[i], window, return_mean=True) # here, ci = sd   ##NO IDEA WHERE THIS IS
        np.divide(ci, im, out=ci)
        # ci now equivalent to ci = sd / im
        
        # Original formulation is w = (-damp * (ci - cu)) / (cmax - ci)
        # Here, by clipping, we will obtain a value of 0 where ci <= cu,
        # and a value of -inf where ci >= cmax. After the exp, these two
        # values will be 1 and 0 respectively. A weight of 1 will make it so that
        # filtered pixel is coming entirely from the image mean, and a weight of zero
        # will mean we keep the original pixel value. Intermediate weights will result
        # in a weighted contribution from each.
        np.clip(ci, cu, cmax, out=ci)
        with np.errstate(divide='ignore'):
            w = (cu - ci) / (cmax - ci)
        if damp != 1:
            np.multiply(damp, w, out=w)
        np.exp(w, out=w)

        # Weight controls how much weight we give to the spatially averaged version,
        # relative to the original signal.
        # This part equivalent to d[i] = (im * w) + (d[i] * (1.0 - w))
        np.multiply(im, w, out=im)
        np.subtract(1.0, w, out=w)
        np.multiply(d[i], w, out=d[i])
        np.add(im, d[i], out=d[i])
        
    return d
    
def enhanced_lee(img , looks, window=7, df=1):
    """ Apply enhanced lee filter to image.  Does not modify original.
    
    Enhanced lee filter following Lopes et al. (1990) (PCI implementation)

    
    *Parameters*
    
    img : numpy array  
        Array to which filter is applied
    window : int
        Size of filter
    looks : int
        Number of looks in input image
    df : int
        Number of degrees of freedom

    *Returns*
    
    array
        
    
    R = Im for Ci <= Cu
        R = Im * W + Ic * (1-W) for Cu < Ci < Cmax
        R = Ic for Ci >= Cmax
        where:
        W = exp (-Damping Factor (Ci-Cu)/(Cmax - Ci))
        Cu = SQRT(1/Number of Looks)
        Ci = S / Im
        Cmax = SQRT(1+2/Number of Looks)
        Ic = center pixel in the kernel
        Im = mean value of intensity within the kernel
        S = standard deviation of intensity within the kernel 
    
    """
    
    img = np.array(img, dtype=np.float32) # convert from int to avoid numeric overflow 
    Cu = np.sqrt(1 / looks)
    Cmax = np.sqrt(1 + 2 / looks)

    Ic = np.copy(img) # copy so we don't modify the original
    Im = uniform_filter(img, window)
    S = window_stdev(img, window, Im)
    Ci = S / Im
    
    # Do R2 first so that Im and Ic can then be modified in-place
    # Also mask any W calculations that aren't in (Cu < Ci < Cmax) case because 
    # for (Cmax - Ci) << 1, np.exp will return overflow errors.
    M2 = np.less(Cu, Ci) * np.less(Ci, Cmax)
    W = np.exp((-df * (Ci - Cu)/(Cmax - Ci)) * M2)
    R2 = (Im * W + Ic * (1-W)) * M2
    
    M1 = np.less_equal(Ci, Cu)
    Im *= M1
    
    M3 = np.greater_equal(Ci, Cu)
    Ic *= M3
    
    R = Im + R2 + Ic
    
    return(R)

def moving_window_sd(data, window, return_mean=False, return_variance=False):
    """
    This is Ben's implementation
    Calculate a moving window standard deviation (and mean)
    """
    
    t1 = data.copy()
    t2 = np.square(t1)
    
    uniform_filter(t1, window, output=t1)
    uniform_filter(t2, window, output=t2)
    
    if return_mean:
        m = t1.copy()
        
    np.square(t1, out=t1)
    np.subtract(t2, t1, out=t2)
    t2[t2 < 0] = 0
    del t1
    
    if not return_variance:
        np.sqrt(t2, out=t2)

    if return_mean:
        return t2, m
    else:
        return t2


def window_stdev(img, window, img_mean=None, img_sqr_mean=None):
    """ Calculate standard deviation filter for an image
    
    *Parameters*
    
    img : numpy array  
        Array to which filter is applied
    window : int
        Size of filter
    img_mean : array, optional
        Mean of image calculated using an equally sized window. 
        If not provided, it is computed.
    img_sqr_mean : array, optional
        Mean of square of image calculated using an equally sized window. 
        If not provided, it is computed.
        
    The function is based on code from: 
    http://nickc1.github.io/python,/matlab/2016/05/17/Standard-Deviation-(Filters)-in-Matlab-and-Python.html
    """
    
    if img_mean is None:
        img_mean = uniform_filter(img, window)
    if img_sqr_mean is None:
        img_sqr_mean = uniform_filter(img**2, window)
    std = np.sqrt(img_sqr_mean - img_mean**2)
    
    return(std)

def energy(img, window):
    """ Apply energy texture filter to an image. Does not modify original.
    
    *Parameters*
   
    img : numpy array  
        Array to which filter is applied
    window : int
        Size of filter
        
    *Returns*
    
    array   
        Filtered array with float32 datatype ( done internally to avoid overflow)
    """
    
    img = np.array(img, dtype='float32')
    np.square(img, out=img)
    uniform_filter(img, window, output=img)
    
    return(img)
    
def filter_image(file, output=None, filter='lee', **kwargs):
    '''
    *Parameters*
    
    file : str : 
        File to filter (may have multiple bands)
    filter : str 
        Name of filter to use on each band
    output : str 
        Path to output file. If none, overwrites input file
    
    '''
    # select filter
    filter = {'lee': lee_filter,
              'elee':enhanced_lee_filter
              }[filter.lower()]
    
    # open dataset for writing or read-only
    if output is None:
        access = gdal.GF_Write
    else:
        access = gdal.GF_Read
    
    img = gdal.Open(file, access)
    
    # read data, ensuring 3-dimensional array shape
    arr = img.ReadAsArray()
    if len(arr.shape) == 2:
        arr = arr[np.newaxis, :, :]
    
    # Create output file if specified
    if output is None:
        out = img
    else:
        out = cloneRaster(img, output)
        
    # filter
    for band_i in range(0, img.RasterCount):
        print("band: {}".format(band_i + 1))
        # create filtered data and write to file
        filtered = filter(arr[band_i, :, :], **kwargs)
        out.GetRasterBand(band_i + 1).WriteArray(filtered)
        
    # close dataset(s)
    out.FlushCache() # TODO: should this be called for each band?
    del out, img
    
