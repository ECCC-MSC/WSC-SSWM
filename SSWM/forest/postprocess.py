"""
Postprocessing for probability images generated using random forest classification
"""

import gdal
import geopandas as gp
import logging
import numpy as np
import os
import pandas as pd
import subprocess
import sys 

from osgeo import ogr
from rasterstats import zonal_stats 
from scipy.ndimage import maximum_filter
from skimage.filters.rank import modal

import SSWM.preprocess.preutils as du

logging = logging.getLogger(__name__)

def postprocess(classified_img, output_poly, pythonexe, gdalpolypath, extrasTXT, window=7):
    """ Postprocess a classified probability image to remove false positives

    using a techinque inspired by Bolanos et al. (2013)
    
    *Parameters*
    
    classified_img : str
        path to classified probability image
    output_poly : str
        path to output GPKG file
    pythonexe : str
        path to python executable
    gdalpolypath : str
        path to gdal_polygonize.py file
    extrasTXT : str
        path containing RF model quality metrics
    window : int
        window size to use for filtering (Default 7)
    """
    
    binary_cls = os.path.splitext(classified_img)[0] + "_classified_filt.tif"
    tmp_polygons = os.path.splitext(classified_img)[0] + "_tmppoly.gpkg"
 
    modefilter(classified_img, output=binary_cls, window=window)
    #max_filter_inplace(binary_cls, band=1, size=3) # testing

    set_nodata(binary_cls, nodata=0)
    polygonize(tmp_polygons, binary_cls, pythonexe=pythonexe, gdalpolypath = gdalpolypath) 
    print("Calculating zonal statistics")
    #stats = pd.DataFrame(zonal_stats(tmp_polygons, classified_img, stats="mean max") )

    stats = pd.DataFrame(gp.read_file(tmp_polygons))

    # Read performace metrics from txt
    openExtras = open(extrasTXT, 'r')
    name = os.path.splitext(os.path.basename(binary_cls))[0]
    components = name.split('_')
    datetimestr = str(components[5] + components[6])

    logging.info(name)
    logging.info(datetimestr)

    names = pd.Series([str(name)] * len(stats), dtype='object')
    stats['SceneID'] = names
    dates = pd.Series([datetimestr] * len(stats), dtype='object')
    stats['Date'] = dates

    count = 0
    extras = []
    for line in openExtras:
        if count > 7:
            break

        val = line.split('=')[1]
        val = val.strip()
        extras.append(val)

    openExtras = None

    TN = np.array([extras[0]] * len(stats), dtype='int64')
    stats['TrueNegative'] = TN
    FN = np.array([extras[1]] * len(stats), dtype='int64')
    stats['FalseNegative'] = FN
    FP = np.array([extras[2]] * len(stats), dtype='int64')
    stats['FalsePositive'] = FP
    TP = np.array([extras[3]] * len(stats), dtype='int64')
    stats['TruePositive'] = TP
    PREC = np.array([extras[5]] * len(stats), dtype='float64')
    stats['Precision'] = PREC
    REC = np.array([extras[7]] * len(stats), dtype='float64')
    stats['Recal'] = REC
    F1 = np.array([extras[6]] * len(stats), dtype='float64')
    stats['F1'] = F1

    #  Add attributes and filter
    polys = gp.read_file(tmp_polygons)
    for col in stats:
        polys[col] = stats[col]
    
    if len(polys):
        polys.to_file(output_poly, driver='GPKG')
    
    os.remove(tmp_polygons)
    
def threshold(input, val=50):
    """ Threshold a raster image and return the new array """
    img  = gdal.Open(input)
    
    arr = img.GetRasterBand(1).ReadAsArray()
    p50 = np.greater_equal(arr, val).astype('uint8')
    arr = None
    
    return p50

def modefilter(input, output, window=7):
    """ Threshold water classification at 50% water likelihood"""
    print(f"Running mode filter (window = {window})")   
    # split into 50% and 100% water
    p50 = threshold(input, val=50)

    # Write 
    out50 = np.empty_like(p50)
    modal(p50,  selem=np.ones((window,window)), out=out50)
    p50 = None
    du.write_array_like(input, output, out50, dtype=2)

def grow_regions(input, output, window=3, val=50):
    """ Threshold water classification and grow lakes by 1 pixel"""
    print(f"Growing regions (window = {window})")
    p50 = threshold(input, val=val)    
    
    # Write 
    out50 = np.empty_like(p50)
    modal(p50,  selem=np.ones((window,window)), out=out50)
    p50 = None
    maximum_filter(out50,  size=window, output=out50)
    du.write_array_like(input, output, out50, dtype=2)
    
    
def set_nodata(file, nodata=0):
    """ Set nodata value for raster file 
    
    *Parameters*
    
    file : str
        path to EXISTING raster file 
    nodata : numeric 
        value to set as nodata for input raster  
    """

    img = gdal.Open(file, gdal.GA_Update)
    band = img.GetRasterBand(1)
    band.SetNoDataValue(nodata)
    band.FlushCache()
    band = None
    img = None

def rasterize_inplace(rast, inshape, prefill=0):
    """  Overwrites a raster with the output of a polygon rasterization
    
    *Parameters*
    
    rast : str
        path to EXISTING raster file that will store values from rasterized 
    inshape : str
        path to vector dataset that will be rasterized
    prefill : int
        Value to write to raster before writing polygonization result
    """
        
    img = gdal.Open(rast, gdal.GA_Update)
    arr = img.ReadAsArray()
    arr[:] = prefill
    img.GetRasterBand(1).WriteArray(arr)
    img.FlushCache()
    
    del img
    
    rst = gdal.Open(rast, gdal.GA_Update)
    gdal.Rasterize(rst, inshape, burnValues=[1])
    
    del rst 
  
def polygonize(output, rast,  pythonexe="python",
                            gdalpolypath = "/usr/bin/gdal", fmt="GPKG", shell=False):
    """ Convert raster to polygons
    
    The input raster should be equal to 1 wherever a polygon is desired and 
    zero elsewhere. The raster nodata value should also be set to zero for
    maximum performance 
    
    *Parameters*
    
    output : str
        path to output polygon file with file extension
    rast : str
        path to raster file that will be polygonized
    pythonexe : str
        path to python executable
    gdalpolypath : str
        path to gdal_polygonize.py file 
    fmt : str
        GDAL-compatible format for output polygons
    shell : boolean
        Passed to subprocess. Experimental.
    
    *Returns*
    
    int
        return code for the subprocess.call function
    """
    
    print("Converting raster to polygons")
    
    if os.path.isfile(output):
        os.remove(output)
    command = "{} {} {} {} -f {}".format(pythonexe, gdalpolypath, rast, output, fmt)
    result = os.system(command)
    if result != 0:
        raise RuntimeError("Error during subprocess call to GDAL polygonize")
    
    return(result)                        
         

def raststats(inshape, raster):
    """ calculate mean and max value of a raster in each polygon """
    print("Calculating zonal statistics")
    stats = zonal_stats(inshape, raster, stats="mean max")         
    return(stats)
         
def max_filter_inplace(img_path, band=1, size=3):
    """ Run a maximum filter on a raster file and changes the values in-place """
    img = gdal.Open(img_path, gdal.GA_Update)
    
    arr = img.GetRasterBand(band).ReadAsArray()
    maximum_filter(arr, size=size, output=arr)
    img.GetRasterBand(band).WriteArray(arr)
    
    img.FlushCache()
    del img, arr

    
if __name__ == "__main__":   
    import argparse 
    
    parser = argparse.ArgumentParser(description = "Filter false positives from random forest probability image.")
    parser.add_argument('img', metavar='IMG', type=str, help="Path of image to process")
    parser.add_argument('python', metavar='PYTH', type=str, help="Path to python executable")
    parser.add_argument('gdal', metavar='GDAL', type=str, help="Path to gdal_polygonize")
    parser.add_argument('-w', '--window', metavar='W', type=int, default=7, help="Window size for majority filter")
    parser.add_argument('-i', '--include_high_estimate', action='store_true', dest='high_estimate', help="Includes high-water estimate")
    
    args = parser.parse_args()
    img = args.img
    python = args.python
    gdalpoly = args.gdal
    window = args.window
    
    outpoly = os.path.splitext(img)[0] + "_classified_filt.gpkg"
    
    postprocess(img, outpoly, python, gdalpoly, window=window)
    if args.high_estimate:
        highpoly = os.path.splitext(img)[0] + "_classified.gpkg"
        postprocess_highestimate(img, highpoly, python, gdalpoly)















