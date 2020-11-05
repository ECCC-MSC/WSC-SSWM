# preprocessing for radar images
# note that one cannot create VRTs with heterogenous data types (e.g. byte + UInt16.)
# It is necessary to go into the XML and edit it manually (or use a script)
# this complicates gathering the output significantly.
# Once the mixed-type VRT exists, one cannot use gdal_translate to change it into
# e.g. a PIX file because the intermediate step requires gdalbuildVRT.
#


import gdal
import numpy as np
import os
import re
import shutil
import sys
import tarfile
import xml.etree.ElementTree as ET
import zipfile

from os import path

from SSWM.preprocess.orthorectify import orthorectify_dem_rpc
from SSWM.preprocess.preutils        import reproject_image_to_master, createvalidpixrast, RS2, ProcessSLC, incidence_angle_from_xml, cloneRaster, RS2, calibrate_in_place
from SSWM.preprocess.filters      import filter_image, energy
from SSWM.preprocess import DEM as de
from SSWM.utils import bandnames
from SSWM.PSPOL.pspol import pspolfil as psp
from SSWM.PSPOL.pspol import pspolfil_memsafe as psp_mem

def preproRS2(product_xml, DEM_dir, cleanup=True, product="CDED"):
    """ Preprocess Radarsat-2 file in preparation for classification 
    
    *Parameters*
    
    product_xml : str
        Path to product.xml file for Radarsat-2 image
    DEM_dir : str
        Path to directory containing DEM files in appropriate folder hierarchy
    cleanup : boolean
        Whether intermediate files should be deleted
    product : str
        Which DEM product to use. 
        
    *Returns*

    str
        Path to zipped output files
    """
    
    wd = path.dirname(product_xml)

    OUT_ENERGY  = path.join(wd, "OUT_ENERGY.tif")
    TMP_DEM     = path.join(wd, "TMP_DEM.tif")
    OUT_ORTHO   = path.join(wd, "OUT_ORTHO.tif")
    OUT_VALID   = path.join(wd, "OUT_VALID.tif")

    OUT_FINAL = path.join(wd, path.basename(wd) +".vrt")
    
    merge_files = []
    imagery_files = RS2.product_xml_imagery_files(product_xml)
    
    ## CHECK FOR COMPLEX VALUES
    #=================
    print("{:#^84}".format('  Check for any complex values (SLC) and convert  '))
    complex = ProcessSLC(product_xml)
    print("{:#^84}".format('  Done!  '))
    
    ## CALIBRATE & FILTER IMAGERY
    #=================
    print("{:#^84}".format('  Begin image filtering and calibration '))
    pol = re.findall("_([HV]*)\\.tif", ','.join(imagery_files))
    for file in imagery_files:
        print("{:-^40}".format('  Applying radiometric calibration to:  {}  '.format(path.basename(file))))
        sigmalut = RS2.lut(product_xml, "Sigma")
        calibrate_in_place(file, sigmalut, complex, 2e4, [1])
        
        print("{:-^40}".format('  Filtering:  {}  '.format(path.basename(file))))

    #== begin PSPOLFIL 
    img = gdal.Open(product_xml)
    # filterbands = [i for i in range(img.RasterCount) if img.GetRasterBand(i+1).GetDescription() in bandnames.DETECTED_BANDS]
    if img.RasterCount == 1:
        amp2e4 = np.moveaxis(np.atleast_3d(img.ReadAsArray()), 2, 0)[:,:,:] 
    else:
        amp2e4 = img.ReadAsArray()[:,:,:]
    

    pow = np.square(amp2e4 / 2e4, dtype='float64')
    totpow = np.sum(pow, axis=0)
    
    filtered = psp_mem(img=pow, P=totpow, numlook=1, winsize=5, pieces=5)
    amp2e4[:] = np.sqrt(filtered) * 2e4
    
    for i, f in enumerate(imagery_files):
        imf = gdal.Open(f, gdal.GA_Update)
        print("opened",f)
        imf.GetRasterBand(1).WriteArray(amp2e4[i,:,:])
        del imf
        
    del img
    #== end PSPOLFIL
    
    print("{:#^84}".format('  Calibration and Filtering Complete  '))
    
    ## Get DEM and orthorectify RS2
    #===============================
    print("{:#^84}".format('  Begin Orthorectification  '))
            
    # load image
    img = gdal.Open(product_xml, gdal.GA_ReadOnly)
    
    # get image extent (xmin, xmax, ymin, ymax) check for existence of kml
    if os.path.isfile(re.sub("xml","kml",product_xml)):
        print("TODO:: Try to get extent from KML instead!")
        extent = de.get_spatial_extent(product_xml)
    else:
        extent = de.get_spatial_extent(product_xml)
    
    # build dem
    de.create_DEM_mosaic_from_extent(extent, dstfile=TMP_DEM, 
                                       DEM_dir = DEM_dir, product=product)

    tmpReproj = path.join(wd, "TMP_REPROJDEM.tif")
    gdal.Warp(tmpReproj, TMP_DEM, dstSRS='EPSG:4326')
    os.remove(TMP_DEM)
    os.rename(tmpReproj, TMP_DEM)
                                        
    orthorectify_dem_rpc(img, OUT_ORTHO, DEM = TMP_DEM)
    merge_files.append(OUT_ORTHO)
    del(img)
    print("{:#^84}".format('  Orthorectification Complete '))
    
    
    ## CREATE VALID BANDS GRID
    #============================ 
    print("{:#^84}".format('  Begin Valid pixel band creation  '))
    img = gdal.Open(OUT_ORTHO, gdal.GA_ReadOnly)
    createvalidpixrast(img, OUT_VALID, 1)
    merge_files.append(OUT_VALID)
    del(img)
    print("{:#^84}".format('  Valid pixel band creation complete  '))
    
    ## CREATE ENERGY TEXTURE GRID
    #============================ 
    print("{:#^84}".format('  Begin texture band creation  '))
    orig = gdal.Open(OUT_ORTHO)
    textr = cloneRaster(orig, OUT_ENERGY, ret=True, all_bands = True, coerce_dtype=gdal.GDT_Float32)
    for band in range(1, orig.RasterCount + 1):
        if orig.GetRasterBand(band).GetDescription() in bandnames.DETECTED_BANDS:
            e = energy(orig.GetRasterBand(band).ReadAsArray(), 5)
            tb = textr.GetRasterBand(band)
            tb.WriteArray(e)
            tb.FlushCache()
            del tb
    del textr, orig
    print("{:#^84}".format('  texture band creation complete  '))

    ## CREATE MERGED VRT
    #============================
    # This requires some XML mumbo-jumobo because we can't easily make
    # VRTs with mixed data types.  One option would be to rescale the float32's
    # (slope and TPI) but then we would have to keep track of the fact that the
    # downstream processes are not training on 'slope' but on '(slope * 1000)'
    # or (TPI + 100) * 1000.
    
    # The general trick is to make VRTs for each raster, then brute-force
    # paste them together and edit metadata afterwards
    
    print("{:#^84}".format('  Creating VRT  '))
    
    # build vrt for ortho
    ortho_vrt = re.sub("tiff?$", "vrt", OUT_ORTHO)
    gdal.BuildVRT(ortho_vrt, OUT_ORTHO)
    orth = ET.parse(ortho_vrt)
    j = len(orth.findall("VRTRasterBand")) + 2
    root = orth.getroot()

        
    ## build vrt for valid
    valid_vrt = re.sub("tiff?$", "vrt", OUT_VALID)
    gdal.BuildVRT(valid_vrt, OUT_VALID)
    
    # read vrt XML
    valid = ET.parse(valid_vrt)
    x = valid.findall("VRTRasterBand")[0]
    
    # insert and modify
    root.insert(j, x)
    root[j].attrib['band'] = str(j-1)
    root[j].remove(root[j][0])
    j += 1
    
    
    ## build vrt for energy
    # build vrt 
    energy_vrt = re.sub("tiff?$", "vrt", OUT_ENERGY)
    gdal.BuildVRT(energy_vrt, OUT_ENERGY)
    
    # read XML
    eng = ET.parse(energy_vrt)
    engbands = eng.findall("VRTRasterBand")
    
    # insert and modify
    for (i, x) in enumerate(engbands):
        root.insert(j, x)
        root[j].attrib['band'] = str(j-1)
        if i==0:
            root[j].remove(root[j][0])
        j += 1
        
    # write
    orth.write(OUT_FINAL)
    
    print("{:#^84}".format('  VRT  Complete '))
    
    ## LABEL BANDS
    #=============================
    print("{:#^84}".format('  Assign band metadata  '))
    
    img = gdal.Open(OUT_FINAL, gdal.GA_Update)
    # for (i, description) in enumerate(pol + ['Valid Data Pixels', 'Slope', 'TPI']):
    enpol = ['energy_'  + p for p in pol]
    for (i, description) in enumerate(pol  + ['Valid Data Pixels'] + enpol):
        img.GetRasterBand(i + 1).SetDescription(description)
    del img
    print("{:#^84}".format('   Band metadata complete  '))
    
    ## ZIP
    #==============================
    
    print("{:#^84}".format('  Zipping output  '))
    
    zip_out = re.sub("vrt", "tar", OUT_FINAL)
    with tarfile.open(zip_out, 'a') as myzip:
        for file in [OUT_ORTHO, OUT_VALID, OUT_ENERGY, OUT_FINAL]:
            print("Adding file to archive: {}".format(path.basename(file)))
            myzip.add(file, arcname=path.basename(file))
    print("{:#^84}".format('   Zipping complete  '))
    
    ## CLEAN UP
    #================= 
    if cleanup:
        print("{:#^84}".format('  Begin File cleanup  '))
        
        for file in [TMP_DEM, OUT_ORTHO, OUT_VALID, 
                    valid_vrt, ortho_vrt, energy_vrt, OUT_FINAL]:
            if os.path.isfile(file):
                os.remove(file)

        print("{:#^84}".format('  File cleanup complete  '))

    return(zip_out)


def preproRCM_bd(folder, DEM_dir, cleanup=True, product="CDED", filter=True):  
    """ Preprocess RCM scenes that have been converted to *.tif files 
    
    This assumes files have been converted to (amplitude * 20k) values and
    have embedded GCPs
    
    *Parameters*
    
    folder : str
        Path to data folder
    DEM_dir : str
        Path to directory containing DEM files in 
    cleanup : boolean
        Whether intermediate files should be deleted
    product : str
        Which DEM product to use. 
        
    *Returns*
    
    str
        Path to zipped output files
    """ 

    tif = None
    for item in os.listdir(folder):
        if item.endswith('.tif'):
            tif = path.join(folder, item)

    if tif is None:
        sys.exit(1)

    wd = path.dirname(tif)
    gdinfo = gdal.Info(tif)

    TMP_DEM     = path.join(wd, "TMP_DEM.tif")
    OUT_ORTHO   = path.join(wd, "OUT_ORTHO.tif")
    OUT_VALID   = path.join(wd, "OUT_VALID.tif")
    OUT_ENERGY  = path.join(wd, "OUT_ENERGY.tif")
    #OUT_FINAL  = path.splitext(tif)[0] + ".vrt"
    OUT_FINAL   = os.path.join(folder, os.path.basename(folder) + ".vrt")
    
    if filter:
        ## FILTER IMAGERY
        #=================
        print("{:#^84}".format('  Begin image filtering  '))
        
        backup_name = re.sub("\\.tiff?$", "_orig.tif", tif)
        os.rename(tif, backup_name)
        cloneRaster(backup_name, tif, ret=False, all_bands=True, coerce_dtype=6, copy_data=True)
        
        img = gdal.Open(tif , gdal.GA_Update)
        pol = [img.GetRasterBand(i + 1).GetDescription() for i in range(img.RasterCount)]
        
        filterbands = [i for i in range(img.RasterCount) if img.GetRasterBand(i+1).GetDescription() in bandnames.DETECTED_BANDS]
        if img.RasterCount == 1:
            amp2e4 = np.moveaxis(np.atleast_3d(img.ReadAsArray()), 2, 0)[filterbands,:,:] 
        else:
            amp2e4 = img.ReadAsArray()[filterbands,:,:]
         
        pow = np.square(amp2e4 / 2e4, dtype='float64')
        totpow = np.sum(pow, axis=0)
        filtered = psp(img=pow, P=totpow, numlook=1, winsize=5)
        amp2e4[:] = np.sqrt(filtered) * 2e4
        
        
        for i, bnd in enumerate(filterbands):
            img.GetRasterBand(bnd+1).WriteArray(amp2e4[i,:,:])
   
        del img
        ## End test filter block
        
        print("{:#^84}".format('  Filtering Complete  '))
    
    ## Get DEM and orthorectify RS2
    #===============================
    print("{:#^84}".format('  Begin Orthorectification  '))
            
    # load image
    img = gdal.Open(tif, gdal.GA_ReadOnly)
    
    # get image extent (xmin, xmax, ymin, ymax) check for existence of kml
    extent = de.get_spatial_extent(tif)
    
    # build dem
    de.create_DEM_mosaic_from_extent(extent, dstfile=TMP_DEM, 
                                       DEM_dir = DEM_dir, product=product)

    tmpReproj = path.join(folder, "TMP_REPROJDEM.tif")
    gdal.Warp(tmpReproj, TMP_DEM, dstSRS='EPSG:4326')
    os.remove(TMP_DEM)
    os.rename(tmpReproj, TMP_DEM)
                                        
    orthorectify_dem_rpc(img, OUT_ORTHO, DEM = TMP_DEM)
    del(img)
    print("{:#^84}".format('  Orthorectification Complete '))
    
    
    ## CREATE VALID BANDS GRID
    #============================ 
    print("{:#^84}".format('  Begin Valid pixel band creation  '))
    img = gdal.Open(OUT_ORTHO, gdal.GA_ReadOnly)
    createvalidpixrast(img, OUT_VALID, 1)
    del(img)
    print("{:#^84}".format('  Valid pixel band creation complete  '))
    

    ## CREATE ENERGY TEXTURE GRID
    #============================ 
    print("{:#^84}".format('  Begin texture band creation  '))
    orig = gdal.Open(OUT_ORTHO)
    textr = cloneRaster(orig, OUT_ENERGY, ret=True, all_bands = True, coerce_dtype=gdal.GDT_Float32)
    for band in range(1, orig.RasterCount + 1):
        if orig.GetRasterBand(band).GetDescription() in bandnames.DETECTED_BANDS:
            e = energy(orig.GetRasterBand(band).ReadAsArray(), 5)
            tb = textr.GetRasterBand(band)
            tb.WriteArray(e)
            tb.FlushCache()
            del tb
    del textr, orig
    print("{:#^84}".format('  texture band creation complete  '))
    
    
    ## CREATE MERGED VRT
    #============================
    # This requires some XML mumbo-jumobo because we can't easily make
    # VRTs with mixed data types.  One option would be to rescale the float32's
    # (slope and TPI) but then we would have to keep track of the fact that the
    # downstream processes are not training on 'slope' but on '(slope * 1000)'
    # or (TPI + 100) * 1000.
    
    # The general trick is to make VRTs for each raster, then brute-force
    # paste them together and edit metadata afterwards
    
    print("{:#^84}".format('  Creating VRT  '))
    
    # build vrt for ortho
    ortho_vrt = re.sub("tiff?$", "vrt", OUT_ORTHO)
    gdal.BuildVRT(ortho_vrt, OUT_ORTHO)
    orth = ET.parse(ortho_vrt)
    j = len(orth.findall("VRTRasterBand")) + 2
    root = orth.getroot()
    
    ## build vrt for valid
    valid_vrt = re.sub("tiff?$", "vrt", OUT_VALID)
    gdal.BuildVRT(valid_vrt, OUT_VALID)
    
    # read vrt XML
    valid = ET.parse(valid_vrt)
    x = valid.findall("VRTRasterBand")[0]
    
    # insert and modify
    root.insert(j, x)
    root[j].attrib['band'] = str(j-1)
    root[j].remove(root[j][0])
    j += 1
    
    ## build vrt for energy
    # build vrt 
    energy_vrt = re.sub("tiff?$", "vrt", OUT_ENERGY)
    gdal.BuildVRT(energy_vrt, OUT_ENERGY)
    
    # read XML
    eng = ET.parse(energy_vrt)
    engbands = eng.findall("VRTRasterBand")
    
    # insert and modify
    for (i, x) in enumerate(engbands):
        root.insert(j, x)
        root[j].attrib['band'] = str(j-1)
        if i==0:
            root[j].remove(root[j][0])
        j += 1
        
    # write
    orth.write(OUT_FINAL)
    
    print("{:#^84}".format('  VRT  Complete '))
    
    ## LABEL BANDS
    #=============================
    print("{:#^84}".format('  Assign band metadata  '))
    
    img = gdal.Open(OUT_FINAL, gdal.GA_Update)
    # for (i, description) in enumerate(pol + ['Valid Data Pixels', 'Slope', 'TPI']):
    enpol = ['energy_'  + p for p in pol]
    for (i, description) in enumerate(pol  + ['Valid Data Pixels'] + enpol):
        img.GetRasterBand(i + 1).SetDescription(description)
    del img
    print("{:#^84}".format('   Band metadata complete  '))
    
    ## ZIP
    #==============================
    
    print("{:#^84}".format('  Zipping output  '))
    
    zip_out = re.sub("vrt", "tar", OUT_FINAL)
    with tarfile.open(zip_out, 'a') as myzip:
        for file in [OUT_ORTHO, OUT_VALID, OUT_ENERGY, OUT_FINAL]:
            print("Adding file to archive: {}".format(path.basename(file)))
            myzip.add(file, arcname=path.basename(file))
    print("{:#^84}".format('   Zipping complete  '))
    
    ## CLEAN UP
    #================= 
    if cleanup:
        print("{:#^84}".format('  Begin File cleanup  '))
        
        for file in [TMP_DEM, OUT_ORTHO, OUT_VALID, OUT_ENERGY,
                    valid_vrt, ortho_vrt, energy_vrt, OUT_FINAL]:
            if os.path.isfile(file):
                os.remove(file)
            
        os.remove(tif)
        os.rename(backup_name, tif)
        
        print("{:#^84}".format('  File cleanup complete  '))

    return(zip_out)

def preproS1(folder, DEM_dir, cleanup=True, product="CDED"):
    """ Preprocess Radarsat-2 file in preparation for classification

    *Parameters*

    product_xml : str
        Path to data folder
    DEM_dir : str
        Path to directory containing DEM files in appropriate folder hierarchy
    cleanup : boolean
        Whether intermediate files should be deleted
    product : str
        Which DEM product to use.

    *Returns*

    str
        Path to zipped output files
    """

    #wd = path.dirname(folder)
    manifest = os.path.join(folder, 'manifest.safe')

    OUT_ENERGY = path.join(folder, "OUT_ENERGY.tif")
    TMP_DEM = path.join(folder, "TMP_DEM.tif")
    OUT_ORTHO = path.join(folder, "OUT_ORTHO.tif")
    OUT_VALID = path.join(folder, "OUT_VALID.tif")
    OUT_TMP = path.join(folder, "OUT_TMP.tif")

    OUT_FINAL = path.join(folder, path.basename(folder) + ".vrt")

    merge_files = []
    imagery_files = [f.strip() for f in re.findall(".*tiff", gdal.Info(manifest))]

    ## CHECK FOR COMPLEX VALUES
    # =================
    #print("{:#^84}".format('  Check for any complex values (SLC) and convert  '))
    #complex = ProcessSLC(product_xml)
    #print("{:#^84}".format('  Done!  '))

    ## CALIBRATE & FILTER IMAGERY
    # =================

    img = gdal.Open(manifest)
    pol = [img.GetRasterBand(i+1).GetMetadata_Dict()['POLARISATION'] for i in range(img.RasterCount)]
    # filterbands = [i for i in range(img.RasterCount) if img.GetRasterBand(i+1).GetDescription() in bandnames.DETECTED_BANDS]
    if img.RasterCount == 1:
        amp2e4 = np.moveaxis(np.atleast_3d(img.ReadAsArray()), 2, 0)[:, :, :]
    else:
        amp2e4 = img.ReadAsArray()[:, :, :]

    pow = np.square(amp2e4 / 2e4, dtype='float64')
    totpow = np.sum(pow, axis=0)

    filtered = psp_mem(img=pow, P=totpow, numlook=1, winsize=5, pieces=5)
    amp2e4[:] = np.sqrt(filtered) * 2e4

    for i, f in enumerate(imagery_files):
        imf = gdal.Open(f, gdal.GA_Update)
        print("opened", f)
        imf.GetRasterBand(1).WriteArray(amp2e4[i, :, :])
        del imf

    del img
    # == end PSPOLFIL

    print("{:#^84}".format('  Calibration and Filtering Complete  '))

    ## Get DEM and orthorectify RS2
    # ===============================
    print("{:#^84}".format('  Begin Orthorectification  '))

    gdal.Warp(OUT_ORTHO, manifest, dstSRS='EPSG:4326')

    img = gdal.Open(OUT_ORTHO, gdal.GA_ReadOnly)
    for band in range(1, img.RasterCount + 1):
        img.GetRasterBand(band).SetDescription(img.GetRasterBand(band).GetMetadata_Dict()['POLARISATION'])

    del img

    '''
    # get image extent (xmin, xmax, ymin, ymax) check for existence of kml
    extent = de.get_spatial_extent(manifest)

    # build dem
    de.create_DEM_mosaic_from_extent(extent, dstfile=TMP_DEM,
                                     DEM_dir=DEM_dir, product=product)

    tmpReproj = path.join(folder, "TMP_REPROJDEM.tif")
    gdal.Warp(tmpReproj, TMP_DEM, dstSRS='EPSG:4326')
    os.remove(TMP_DEM)
    os.rename(tmpReproj, TMP_DEM)

    orthorectify_dem_rpc(img, OUT_ORTHO, DEM=TMP_DEM)
    '''

    merge_files.append(OUT_ORTHO)
    #del (img)
    print("{:#^84}".format('  Orthorectification Complete '))

    ## CREATE VALID BANDS GRID
    # ============================
    print("{:#^84}".format('  Begin Valid pixel band creation  '))
    img = gdal.Open(OUT_ORTHO, gdal.GA_ReadOnly)
    createvalidpixrast(img, OUT_VALID, 1)
    merge_files.append(OUT_VALID)
    del (img)
    print("{:#^84}".format('  Valid pixel band creation complete  '))

    ## CREATE ENERGY TEXTURE GRID
    # ============================
    print("{:#^84}".format('  Begin texture band creation  '))
    orig = gdal.Open(OUT_ORTHO)
    textr = cloneRaster(orig, OUT_ENERGY, ret=True, all_bands=True, coerce_dtype=gdal.GDT_Float32)
    for band in range(1, orig.RasterCount + 1):
        if orig.GetRasterBand(band).GetDescription() in bandnames.DETECTED_BANDS:
            e = energy(orig.GetRasterBand(band).ReadAsArray(), 5)
            tb = textr.GetRasterBand(band)
            tb.WriteArray(e)
            tb.FlushCache()
            del tb
    del textr, orig
    print("{:#^84}".format('  texture band creation complete  '))

    ## CREATE MERGED VRT
    # ============================
    # This requires some XML mumbo-jumobo because we can't easily make
    # VRTs with mixed data types.  One option would be to rescale the float32's
    # (slope and TPI) but then we would have to keep track of the fact that the
    # downstream processes are not training on 'slope' but on '(slope * 1000)'
    # or (TPI + 100) * 1000.

    # The general trick is to make VRTs for each raster, then brute-force
    # paste them together and edit metadata afterwards

    print("{:#^84}".format('  Creating VRT  '))

    # build vrt for ortho
    ortho_vrt = re.sub("tiff?$", "vrt", OUT_ORTHO)
    gdal.BuildVRT(ortho_vrt, OUT_ORTHO)
    orth = ET.parse(ortho_vrt)
    j = len(orth.findall("VRTRasterBand")) + 2
    root = orth.getroot()

    ## build vrt for valid
    valid_vrt = re.sub("tiff?$", "vrt", OUT_VALID)
    gdal.BuildVRT(valid_vrt, OUT_VALID)

    # read vrt XML
    valid = ET.parse(valid_vrt)
    x = valid.findall("VRTRasterBand")[0]

    # insert and modify
    root.insert(j, x)
    root[j].attrib['band'] = str(j - 1)
    root[j].remove(root[j][0])
    j += 1

    ## build vrt for energy
    # build vrt
    energy_vrt = re.sub("tiff?$", "vrt", OUT_ENERGY)
    gdal.BuildVRT(energy_vrt, OUT_ENERGY)

    # read XML
    eng = ET.parse(energy_vrt)
    engbands = eng.findall("VRTRasterBand")

    # insert and modify
    for (i, x) in enumerate(engbands):
        root.insert(j, x)
        root[j].attrib['band'] = str(j - 1)
        if i == 0:
            root[j].remove(root[j][0])
        j += 1

    # write
    orth.write(OUT_FINAL)

    print("{:#^84}".format('  VRT  Complete '))

    ## LABEL BANDS
    # =============================
    print("{:#^84}".format('  Assign band metadata  '))

    img = gdal.Open(OUT_FINAL, gdal.GA_Update)
    # for (i, description) in enumerate(pol + ['Valid Data Pixels', 'Slope', 'TPI']):
    enpol = ['energy_' + p for p in pol]
    for (i, description) in enumerate(pol + ['Valid Data Pixels'] + enpol):
        img.GetRasterBand(i + 1).SetDescription(description)
    del img
    print("{:#^84}".format('   Band metadata complete  '))

    ## ZIP
    # ==============================

    print("{:#^84}".format('  Zipping output  '))

    zip_out = re.sub("vrt", "tar", OUT_FINAL)
    with tarfile.open(zip_out, 'a') as myzip:
        for file in [OUT_ORTHO, OUT_VALID, OUT_ENERGY, OUT_FINAL]:
            print("Adding file to archive: {}".format(path.basename(file)))
            myzip.add(file, arcname=path.basename(file))
    print("{:#^84}".format('   Zipping complete  '))

    ## CLEAN UP
    # =================
    if cleanup:
        print("{:#^84}".format('  Begin File cleanup  '))

        for file in [TMP_DEM, OUT_ORTHO, OUT_VALID,
                     valid_vrt, ortho_vrt, energy_vrt, OUT_FINAL]:
            if os.path.isfile(file):
                os.remove(file)

        print("{:#^84}".format('  File cleanup complete  '))

    return (zip_out)