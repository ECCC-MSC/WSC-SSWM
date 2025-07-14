from osgeo import gdal
import numpy as np
import os
import re
import xml.etree.ElementTree as ET

def cloneRaster(img, newRasterfn, ret=True, all_bands=True, coerce_dtype=None, copy_data=False):
    """ make empty raster container from gdal raster object. Does not copy data
    
    *Parameters*
    
    img : osgeo.gdal.Dataset
        An open gdal raster object
    newRasterfn str
        Filename of raster to create
    ret : boolean
        Whether to return a file handle. If False, closes file
    all_bands : boolean
        Whether or not all bands should be copied or just the first one 
    
    *Returns*   
    
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
    outRaster = driver.Create(newRasterfn, cols, rows, bands, dtype)#, options=get_blocksize_options(img))
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
                bnd.WriteArray(array[i,:,:])
            bnd.FlushCache()
            bnd = None
        
    # write data
    outRaster.FlushCache()

    if close:
        del img
    if ret:
        return(outRaster)
    outRaster = None

def get_blocksize_options(img):
    """ Get raster blocksize information as a string that can be passed to gdal """
    blockx, blocky = img.GetRasterBand(1).GetBlockSize()
    opt_str = f"TILED=YES,BLOCKXSIZE={blockx},BLOCKYSIZE={blocky}".split(",")
    return(opt_str)
    
def copy_metadata(src, dst):
    """ Copy metadata from one osgeo.gdal.Dataset to another 
    
    *Parameters*
    
    src : osgeo.gdal.Dataset
        An open gdal raster object
    dst : osgeo.gdal.Dataset
        A gdal raster object that is open for writing
    """
    for domain in src.GetMetadataDomainList() or ():
        dst.SetMetadata(src.GetMetadata(domain), domain)

def copy_georeferencing(src, dst):
    """ Copy geotransform and/or GCPs from one osgeo.gdal.Dataset to another 
    
    *Parameters*
    
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
   
    *Parameters*
    
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
    
    *Parameters*
    
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
        
    *Returns*
    
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
    outRaster = driver.Create(newRasterfn, cols, rows, bands, dtype)#, options=get_blocksize_options(img))
    
    # copy raster projection
    copy_georeferencing(img, outRaster)

    if copy_metadata:
        copy_metadata(img, outRaster)
        copy_band_metadata(img, outRaster, bands=bands)
        
    # copy band data
    for i in range(bands):
        bnd = outRaster.GetRasterBand(i+1)
        if bands == 1:
            bnd.WriteArray(array)
        else:
            bnd.WriteArray(array[:,:,i])
        bnd.FlushCache()
        bnd = None
        
    # write data
    outRaster.FlushCache()
    
    if ret:
        return(outRaster)
        
    del outRaster
 

def createvalidpixrast(img, dst, band):
    """ Create valid pixel raster (0 or 1) for a gdal raster band """
    
    # read in data
    print(band)
    bnd = img.GetRasterBand(band)
    rast = bnd.ReadAsArray()
    nrow, ncol = rast.shape
    
    # create new data, coercing to byte 
    grid = np.array(rast, copy=True)
    grid[grid > 0] = 255 # convert to 0 or 1
    
    # create container
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(dst, ncol, nrow, 1, gdal.GDT_Byte)
    outRaster.SetGeoTransform(img.GetGeoTransform())
    outRaster.SetProjection(img.GetProjection())
    outRaster.GetRasterBand(1).SetDescription("Valid Pixels")
    
    # put data in container and close
    outRaster.GetRasterBand(1).WriteArray(grid)
    outRaster.FlushCache()
    outRaster = None
    
    return(True)
    
def alignraster2target(src, tgt, dst):
    tgtimg = gdal.Open(tgt, gdal.GA_ReadOnly)
    
    wo = gdal.WarpOptions()
    gdal.Warp(dst, src, options=wo)
    return(True)
     
def reproject_image_to_master(master, src, dst):
    """This function reprojects an image (``src``) to
    match the extent, resolution and projection of another
    (``master``) using GDAL. The newly reprojected image
    is a GDAL VRT file for efficiency. A different spatial
    resolution can be chosen by specifyign the optional
    ``res`` parameter. The function returns the new file's
    name.
    
    *Parameters*
    
    master: str 
        A filename (with full path if required) with the 
        master image (that that will be taken as a reference)
    src: str 
        A filename (with path if needed) with the image
        that will be reprojected
    res: float, optional
        The desired output spatial resolution, if different 
        to the one in ``master``.
    
    *Returns*
    
    The reprojected filename
    
    code credit: https://github.com/jgomezdans/eoldas_ng_observations
    """
    src_ds = gdal.Open(src)
    if src_ds is None:
        raise IOError("GDAL could not open src file {}".format(src))
    src_proj = src_ds.GetProjection()
    src_geotrans = src_ds.GetGeoTransform()
    data_type = src_ds.GetRasterBand(1).DataType
    n_bands = src_ds.RasterCount

    master_ds = gdal.Open(master)
    if master_ds is None:
        raise IOError("GDAL could not open master file {}".format(master))
    
    master_proj = master_ds.GetProjection()
    master_geotrans = master_ds.GetGeoTransform()
    w = master_ds.RasterXSize
    h = master_ds.RasterYSize

    dst_filename = dst
    dst_ds = gdal.GetDriverByName('GTiff').Create(dst_filename,
                                                w, h, n_bands, data_type)
    dst_ds.SetGeoTransform(master_geotrans)
    dst_ds.SetProjection(master_proj)

    gdal.ReprojectImage(src_ds, dst_ds, src_proj,
                         master_proj, gdal.GRA_NearestNeighbour)
    dst_ds = None  # Flush to disk
    return dst_filename
    
    
def SLC2IMG(image_file, output):
    """ Convert SLC (Re, Im) raster to amplitude. 
    
    The code is based on the SLC2IMG algorithm from PCI such that 
    DN = int(sqrt(I*I + Q*Q) + 0.5)
    
    *Parameters*
    
    image_file : str 
        Path to imagery file, usually a tiff
    output : str
        Path to output file
    """
    
    img = gdal.Open(image_file)
    
    # sanity check
    if img.RasterCount != 2:
        raise Exception("Input raster does not have 2 bands (Real, Imaginary)")
        
    # copy raster skeleton
    newR = cloneRaster(img, output, ret=True, all_bands=False, coerce_dtype=gdal.GDT_Float32)
    
    # perform conversion 
    re = np.array(img.GetRasterBand(1).ReadAsArray(), 'float32')
    im = np.array(img.GetRasterBand(2).ReadAsArray(), 'float32')
    ReIm2Amp(re, im)
    del im
    band = newR.GetRasterBand(1)
    band.WriteArray(re)
    band.FlushCache
    band = None
    
    # close 
    img.FlushCache
    newR.FlushCache
    img, newR = None, None
    
    print("{} converted from complex values to amplitude".format(os.path.basename(image_file)))
    

def ReIm2Amp(re, im, inplace=True):
    """ Convert complex components to their modulus 
    
    Addes small value to the result to ensure it is positive because the code is
    based on the SLC2IMG algorithm from PCI such that DN = int(sqrt(I*I + Q*Q) + 0.5)
    
    *Parameters*
    
    re : numpy array
        Numpy array of shape (m,n) corresponding to the real component of a 
        complex number. 
    im : 
        Numpy array of shape (m,n) corresponding to the imaginary component of a 
        complex number. 
    inplace :  boolean
        Whether the inputs should be modified in-place. If true, the final result
        is stored in the re array
    
    *Returns*
    
    array
        Modulus of real and imaginary arrays (with shape [m,n])
    """
    if inplace:
        np.square(re, out=re)
        np.square(im, out=im)
        np.add(re, im, out=re)
        del im
        np.sqrt(re, out=re)
        np.add(re, 0.5, out=re)
         
    else:
        raise(NotImplementedError)
        # # first convert to float32 - this slows things down, but if we don't do it, then 
        # # it is possible to produce silent overflow errors for large values.
        # arr = np.array(arr, dtype=np.float32)
        # arr = np.square(arr)
        # arr = np.sum(arr, axis=0)
        # arr = np.sqrt(arr) + 0.5 # add 0.5 because that's what PCI does
    
        return(arr)
    
    
def ProcessSLC(product_xml):
    """ Convert SLC values to raw DN values
    
    Checks whether a RS-2 product.xml file is associated with SLC data and if so,
    converts the two-channel (i,q) *.tif images into single-channel (amplitude) 
    images. Also updates the product.xml file data type attribute from 'Complex'
    to 'Magnitude Detected'
    
    *Parameters*
    
    product_xml : str
        file path pointing to product.xml file 
        
    *Returns*
    
    boolean
        True if completed successfully
        
    """

    schm = '{http://www.rsi.ca/rs2/prod/xml/schemas}'
    imgattr = schm + 'imageAttributes'
    rastattr  = schm + 'rasterAttributes'
    dattyp  = schm + 'dataType'

    # Check the xml to see if datatype is listed as complex
    ET.register_namespace('', 'http://www.rsi.ca/rs2/prod/xml/schemas')
    px = ET.parse(product_xml)
    root = px.getroot()
    dType = root.find(imgattr).find(rastattr).find(dattyp)
    
    if dType.text != 'Complex':
        print("product.xml datatype is not 'Complex'")
        return False
    
    print("Detected complex data, attempting to convert to amplitude \n")    
    img_files = RS2.product_xml_imagery_files(product_xml)
    if not all([Radar.TIF_channels(x) == 2 for x in img_files]):
        print("imagery does not contain two channels each")
        return True
        
    for file in img_files:
        backup = re.sub("\\.tiff?$", "_complex.tif", file)
        os.rename(file, backup)
        SLC2IMG(backup, file)
    
    # Update the xml file so that gdal doesn't interpret the imagery as complex
    # (it doesn't seem to handle the results properly!)
    dType.text = 'Magnitude Detected'
    px.write(product_xml, encoding='UTF-8', method='xml')
    
    print("product.xml updated")
    return True

    
def incidence_angle_from_xml(beta_xml, sigma_xml, nrow, complex=False):
    """ Calculate incidence angle from lutBeta and lutSigma xml files 
    
    *Parameters*
    
    beta_xml : str
        path to lutBeta.xml file
    sigma_xml : str
        path to lutSigma.xml file
    nrow : int
        number of rows in output array (number of lines in original image)
    complex : boolean
        whether or not the xml files represent complex data (in which case 
        beta and sigma values are squared before dividing)
        
    *Returns*
    
    an array of dimension (M,N) with M = nrow and N = the number of values represented
    by each the xml files.
    
    """
    gain_b, offst_b, step_b = read_calibration_gains(beta_xml)
    gain_s, offst_s, step_s = read_calibration_gains(sigma_xml)
    
    if step_b != 1:
        gain_b = interpolate_steps(gain_b, step_b)
    if step_s != 1:
        gain_s = interpolate_steps(gain_s, step_s)
    
     
    theta = incidence_angle_from_gains(gain_b, gain_s, complex=True)
    theta =  theta[np.newaxis, :] 
    theta = theta.repeat(nrow, axis=0)
    
    return(theta)

def read_calibration_gains(xml):
    """ Read calibration info from RCM or RS2 lut*.xml
    
    *Parameters*
    
    xml : str
        Path to look-up table (e.g. sigma.xml)
        
    *Returns*
    
    tuple
        tuple consisting of:
        (1) gains (array) 
        (2) offset (int)
        (3) stepsize (int)
    """ 
    cal = ET.parse(xml)
    root = cal.getroot()
    # XML tree adds {namespace} information onto all labels for RCM, so we have them
    mch = re.search('{.*}', root.tag)
    nmsp = '' if mch is None else mch.group(0)
    
    # Get info
    gains = root.find(nmsp + 'gains')
    gains = np.array(gains.text.split(' '), dtype='float32')
    
    offset = int(float(root.find(nmsp + 'offset').text)) # float first in case format '0.00e+00"

    stepsize = root.find(nmsp + 'stepSize')
    stepsize = 1 if stepsize is None else int(float(stepsize.text))
        
    return gains, offset, stepsize

def read_lut_array(xml, nrow):
    """ """
    gain, offst, step = read_calibration_gains(xml)
    gain = interpolate_steps(gain, step)
    gain = gain[np.newaxis, :] 
    gain = gain.repeat(nrow, axis=0)
    return(gain)
    
def incidence_angle_from_gains(beta_gains, sigma_gains, complex=False):
    """ calculate incidence angle array"""
    if complex:
        beta_gains *= beta_gains
        sigma_gains *= sigma_gains

    theta = np.degrees(np.arcsin(beta_gains / sigma_gains))
    
    return(theta)
    
def interpolate_steps(array, step):
    """ interpolate array with desired step size"""
    if step != 1:
        ar = np.repeat(array, step)[:-2] # last value is not stepped
        for i in np.arange(1, step): # add empty gaps to be imputed
            ar[i::step] = np.nan       
        array = interpolator(ar)
    return(array)

def _find_nan(y):
    """ find nan indices in array """
    return np.isnan(y), lambda z: z.nonzero()[0]

def interpolator(y):
    """ Interpolate missing (nan) values in an array
    
    *Parameters*
    
    y : array
        Array which may contain nan values

    *Returns* 
    
    array
        equal-length array with nan values replaced with imputed data
    
    *Example*
    
    import numpy as np
    a = np.array([1, np.nan, np.nan, 4, np.nan, 6, 7], dtype='float32')
    interpolator(a)
    """
    nans, x = _find_nan(y)
    y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    return(y)

def calibrate(array, lut, complex=False, scale=20000):
    """ apply LUT calibration to a radar array. Modifies array in-place
    
    *Parameters*
    
    array : array-like
        m x n array of raw DN values or modulus for SLC (DNi**2 + DNq**2)**0.5 
    lut : str
        path to xml for LUT
    complex : bool
        does the array represent the modulus of complex (SLC) data?
    scale : int
        scaling factor used to store results as int16 (default is 20000, same
        as CIS for visual interpretation)
    """
    sig  = read_lut_array(lut, array.shape[0])
    # np.square(array, out=array)
    # if complex:
    #     np.square(sig, out=sig)
    # np.divide(array, sig, out=array)
    # if scale != 1:
    #     np.multiply(array, scale, out=array)]
    #----> if not complex, array will be uint16, and will overflow. Rather than
    # converting to float, use sig array (saves on memory). a^2 / b = (a/sqrt(b))^2
    if not complex:
        np.sqrt(sig, out=sig)
    np.divide(array, sig, out=sig)
    np.square(sig, out=sig)
    if scale != 1:
        np.multiply(sig, scale, out=sig)
    array[:] = sig


def calibrate_in_place(file, lut, complex, scale, band=[1]):
    """ Apply LUT calibration to file and change in-place"""
    img = gdal.Open(file, gdal.GA_Update)
    for b in band:
        bnd = img.GetRasterBand(b)
        arr = bnd.ReadAsArray()
        calibrate(arr, lut, complex, scale)
        bnd.WriteArray(arr)
        bnd.FlushCache
        del bnd, arr
    del img
        
class Radar(object):
    """ Generic class for RS2 and RCM folder structures """
    def __init__(self):
        pass
        
    @classmethod
    def TIF_channels(cls, tif):
        """ Get count how many channels are in an image """
        img = gdal.Open(tif)
        n_bands = img.RasterCount
        img = None
        return(n_bands)   


    
class RS2(Radar):
    """ Class for accessing information about an RS2 dataset """
    
    GDAL_CALIB = {'Sigma':'RADARSAT_2_CALIB:SIGMA0:',
                  'Beta' : 'RADARSAT_2_CALIB:BETA0:',
                  'Gamma': 'RADARSAT_2_CALIB:GAMMA:',
                  'Uncalibrated': 'RADARSAT_2_CALIB:UNCALIB:'}
                  
    def __init__(self):
        pass
    
    @classmethod
    def product_xml_imagery_files(cls, xml):
        """ Return a list of which imagery files are associated with a RS-2 product.xml file"""
        imagery_files = [f.strip() for f in re.findall(".*tif", gdal.Info(xml))]
        return(imagery_files)
    
    @classmethod
    def product_xml_pol_modes(cls, xml):
        """ Return a list of polarization modes associated with an RS-2 product.file """
        files = product_xml_imagery_files
        modes = re.findall("_([HV]*)\\.tif", ','.join(imagery_files))
        return(modes)
    
    @classmethod
    def path_to_xml(cls, folder):
        """ given a standard folder with RS-2 data, find the product.xml file"""
        xml = os.path.join(folder, "product.xml")
        if not os.path.isfile(xml):
            xml = os.path.join(folder, os.path.basename(folder), "product.xml")
        return(xml)
        
    @classmethod
    def lut(cls, product_xml, norm='Sigma'):
        """ given product_xml path, find calibration LUTs 
        norm : str
            one of 'Beta', 'Gamma', 'Sigma' (default)
        """
        xml = re.sub("product\\.xml", "lut{}.xml".format(norm.capitalize()), product_xml)
        return(xml)
        
    @classmethod
    def img_dimensions(cls, xml):
        img = gdal.Open(xml)
        size = (img.RasterYSize, img.RasterXSize)
        img = None
        return(size)
    
    @classmethod
    def pol_from_name(cls, name):
        match = re.search('_([HV_]*)_',name).group(1)
        return(match)
        
    @classmethod
    def bm_from_name(cls, name):
        match = re.search('_([^_]*)_\\d{8}_\\d{6}',name).group(1)
        return(match)
    
    @classmethod
    def find_matching_files(cls, dir, bm=None, pol=None, ext=None):
        files = np.array(os.listdir(dir))
        if bm:
            files = files[[True if re.search(bm, x) else False for x in files]]
        if pol:
            files = files[[True if re.search(pol, x) else False for x in files]]
        if ext:
            files = files[[True if os.path.splitext(x[1])[1][1:] else False for x in files]]
        
        return list(files)
        
class RCM(Radar):
    """ Class for accessing information about an RCM dataset  """
    
    def __init__(self):
        pass
        
    @classmethod
    def path_to_xml(cls, folder):
        """ given a standard folder with RCM data, find the product.xml file"""
        xml = os.path.join(folder,"metadata",  "product.xml")
        return(xml)


class S1(Radar):
    """ Class for accessing information about an RCM dataset  """

    def __init__(self):
        pass