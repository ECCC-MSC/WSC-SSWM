import numpy as np
import zipfile

from osgeo import gdal
import os
import re
import xml.etree.ElementTree as ET

from collections import deque

printt = print
def print(*args,**kwargs):
    printt(*args,flush=True,**kwargs)

class bandnames:
    """ Base class for classification tasks to standardize band names.

    This is used to set the names of the bands that are used in the
    classifications.  It can also be used to turn various data bands on or off
    in the classification. Bands will be written in the order they are specified
    in the DATA_BANDS attribute. This also controls the order of terms in the vector
    that is passed to the classifier for each pixel.

    Attributes
    ----------
    MASK_LABEL : list of str
        The name of the band containing water information

    DETECTED_BANDS : list of str
        Data from the satellite. These bands will be filtered
        and in some cases, derived bands will be calculated from them (for instance
        energy texture bands will be calculated)

    DERIVED_BANDS : list of str
        These have been derived from the sensed data and should not have texture
        bands created from them.

    DATA_BANDS : list of str
        The names of the bands that will be used to train and run the random forest model

    VALID_PIX_BAND : list of str
        Identifies the name of the band that identifies valid (1) and invalid (0) pixels

    MIN_F1 : float
        Minimum F1 score threshold below which image classification should be skipped.
    """

    MASK_LABEL = ['water_mask']

    DETECTED_BANDS = ['HH', 'HV', 'VV', 'VH', 'RH', 'RV', 'SE_I', 'SE_P']

    DERIVED_BANDS = ['energy_HH', 'energy_HV',
                     'energy_VH', 'energy_VV',
                     'energy_RH', 'energy_RV',
                     'energy_SE_I', 'energy_SE_P']

    DATA_BANDS = DETECTED_BANDS + DERIVED_BANDS  # ORDER MATTERS HERE

    VALID_PIX_BAND = ['Valid Data Pixels']

    MIN_F1 = 0.5

class ModelModes:
    # Just so we can change it from here and not everywhere...
    TRAIN,EVAL,PREDICT=('TRAIN','EVAL','PREDICT')

        
def rebin(a, new_shape):
    """
    Resizes a 2d array by averaging or repeating elements, 
    new dimensions must be integral factors of original dimensions
    *Parameters*
   
    a : array_like
        Input array.
    new_shape : tuple of int
        Shape of the output array
    
    *Returns*
    
    rebinned_array : ndarray
        If the new shape is smaller of the input array, the data are averaged, 
        if the new shape is bigger array elements are repeated
    
    *See Also*
    
    resize : Return a new array with the specified shape.
    
    *Examples*
    
    >>> a = np.array([[0, 1], [2, 3]])
    >>> b = rebin(a, (4, 6)) #upsize
    >>> b
    array([[0, 0, 0, 1, 1, 1],
           [0, 0, 0, 1, 1, 1],
           [2, 2, 2, 3, 3, 3],
           [2, 2, 2, 3, 3, 3]])
    >>> c = rebin(b, (2, 3)) #downsize
    >>> c
    array([[ 0. ,  0.5,  1. ],
           [ 2. ,  2.5,  3. ]])
    """
    M, N = a.shape
    m, n = new_shape
    if m<M:
        return a.reshape((m,M/m,n,N/n)).mean(3).mean(1)
    else:
        return np.repeat(np.repeat(a, m/M, axis=0), n/N, axis=1)

def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.
    Keeps the dtype of the input array
    Number of output dimensions must match number of input dimensions.
    
    *Example*
    
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)
    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]
    """
    
    if not operation.lower() in ['sum', 'mean', 'average', 'avg']:
        raise ValueError("Operation {} not supported.".format(operation))
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d, c in zip(new_shape,
                                                   ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    print(f'Flattened=>{flattened}')
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        if operation.lower() == "sum":
            ndarray = ndarray.sum(-1*(i+1))
        elif operation.lower() in ["mean", "average", "avg"]:
            ndarray = ndarray.mean(-1*(i+1))
    return ndarray


def copy_metadata(src, dst):
    """ Copy metadata from one osgeo.gdal.Dataset to another

    **Parameters**

    src : osgeo.gdal.Dataset
        An open gdal raster object
    dst : osgeo.gdal.Dataset
        A gdal raster object that is open for writing
    """
    for domain in src.GetMetadataDomainList() or ():
        dst.SetMetadata(src.GetMetadata(domain), domain)


def copy_georeferencing(src, dst):
    """ Copy geotransform and/or GCPs from one osgeo.gdal.Dataset to another

    **Parameters**

    src : osgeo.gdal.Dataset
        An open gdal raster object
    dst : osgeo.gdal.Dataset
        A gdal raster object that is open for writing
    """
    dst.SetGeoTransform(src.GetGeoTransform())
    if src.GetGCPCount():
        dst.SetGCPs(src.GetGCPs(), src.GetGCPProjection())
    else:
        dst.SetProjection(src.GetProjection())


def copy_band_metadata(src, dst, bands):
    """ Copy band metadata from one osgeo.gdal.Dataset to another

    **Parameters**
    -
    src : osgeo.gdal.Dataset
        An open gdal raster object
    dst : osgeo.gdal.Dataset
        A gdal raster object that is open for writing
    bands : int
        How many bands are in the image
    """
    for i in range(bands):
        j = i + 1
        bnd = dst.GetRasterBand(j)
        in_bnd = src.GetRasterBand(j)

        for domain in in_bnd.GetMetadataDomainList() or ():
            bnd.SetMetadata(in_bnd.GetMetadata(domain), domain)

        bnd.SetDescription(in_bnd.GetDescription())
        if in_bnd.GetNoDataValue() is not None:
            bnd.SetNoDataValue(in_bnd.GetNoDataValue())

        bnd.FlushCache()
        del bnd, in_bnd


def write_array_like(img, newRasterfn, array, dtype=6, ret=True, driver='GTiff', copy_metadata=False):
    ''' write numpy array to gdal-compatible raster.

    **Parameters**

    img : osgeo.gdal.Dataset or str
        An open gdal raster object or path to file
    newRasterfn : str
        Filename of raster to create
    array : array
        array  to be written with shape (nrow[y], ncol[x], band)
    dtype : int
        What kind of data should raster contain?
    ret : logical
        Whether to return a file handle. If false, closes file

    **Returns**

    osgeo.gdal.Dataset
        a handle for the new raster file
    '''
    if not isinstance(img, gdal.Dataset):
        img = gdal.Open(img)

    # get image dimensions
    cols = img.RasterXSize
    rows = img.RasterYSize
    bands = np.atleast_3d(array).shape[2]

    # create file
    driver = gdal.GetDriverByName(driver)
    dtype = dtype if dtype else img.GetRasterBand(1).DataType
    outRaster = driver.Create(newRasterfn, cols, rows, bands, dtype)

    # copy raster projection
    copy_georeferencing(img, outRaster)

    if copy_metadata:
        copy_metadata(img, outRaster)
        copy_band_metadata(img, outRaster, bands=bands)

    # copy band data
    for i in range(bands):
        bnd = outRaster.GetRasterBand(i + 1)
        if bands == 1:
            bnd.WriteArray(array)
        else:
            bnd.WriteArray(array[:, :, i])
        bnd.FlushCache()
        bnd = None

    # write data
    outRaster.FlushCache()

    if ret:
        return (outRaster)

    del outRaster


def cloneRaster(img, newRasterfn, ret=True, all_bands=True, coerce_dtype=None, copy_data=False):
    """ make empty raster container from gdal raster object. Does not copy data

    **Parameters**

    img : osgeo.gdal.Dataset
        An open gdal raster object
    newRasterfn str
        Filename of raster to create
    ret : boolean
        Whether to return a file handle. If False, closes file
    all_bands : boolean
        Whether or not all bands should be copied or just the first one

    **Returns**


        a handle for the new raster file (if ret is True)

    """
    close = False
    if not isinstance(img, gdal.Dataset):
        close = True
        img = gdal.Open(img)

    # get image dimensions
    cols = img.RasterXSize
    rows = img.RasterYSize
    bands = img.RasterCount if all_bands else 1

    # create image
    driver = gdal.GetDriverByName('GTiff')
    dtype = coerce_dtype if coerce_dtype else img.GetRasterBand(1).DataType
    outRaster = driver.Create(newRasterfn, cols, rows, bands, dtype)
    outRaster.FlushCache()

    print(newRasterfn)
    # copy metadata
    copy_metadata(img, outRaster)
    copy_georeferencing(img, outRaster)
    copy_band_metadata(img, outRaster, bands=bands)

    if copy_data:
        array = img.ReadAsArray()
        for i in range(bands):
            bnd = outRaster.GetRasterBand(i + 1)
            if bands == 1:
                bnd.WriteArray(array)
            else:
                bnd.WriteArray(array[i, :, :])
            bnd.FlushCache()
            bnd = None

    # write data
    outRaster.FlushCache()

    if close:
        del img
    if ret:
        return (outRaster)
    outRaster = None

