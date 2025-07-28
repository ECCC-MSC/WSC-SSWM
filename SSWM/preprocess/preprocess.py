# preprocessing for radar images
# note that one cannot create VRTs with heterogenous data types (e.g. byte + UInt16.)
# It is necessary to go into the XML and edit it manually (or use a script)
# this complicates gathering the output significantly.
# Once the mixed-type VRT exists, one cannot use gdal_translate to change it into
# e.g. a PIX file because the intermediate step requires gdalbuildVRT.
#


from osgeo import gdal
import numpy as np
import os
import re
import shutil
import sys
import tarfile
import xml.etree.ElementTree as ET
import zipfile

from os import path

from SSWM.preprocess.orthorectify import orthorectify_dem_rpc, orthorectify_otb
from SSWM.preprocess.preutils import reproject_image_to_master, createvalidpixrast, RS2, ProcessSLC, incidence_angle_from_xml, cloneRaster, RS2, calibrate_in_place, calibrateS1
from SSWM.preprocess.filters import lee_filter2
import SSWM.preprocess.DEM as de
from SSWM.utils import bandnames

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

    for band_i in range(0, img.RasterCount):
        print("band: {}".format(band_i + 1))
        # create filtered data and write to file
        filtered = lee_filter2(amp2e4[band_i, :, :], window=(3, 3))
        # amp2e4[:] = np.sqrt(filtered) * 2e4
        img.GetRasterBand(band_i + 1).WriteArray(filtered[:, :])

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
        for file in [OUT_ORTHO, OUT_VALID, OUT_FINAL]:
            print("Adding file to archive: {}".format(path.basename(file)))
            myzip.add(file, arcname=path.basename(file))
    print("{:#^84}".format('   Zipping complete  '))
    
    ## CLEAN UP
    #================= 
    if cleanup:
        print("{:#^84}".format('  Begin File cleanup  '))
        
        for file in [TMP_DEM, OUT_ORTHO, OUT_VALID, 
                    valid_vrt, ortho_vrt, OUT_FINAL]:
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

    TMP_DEM     = path.join(wd, "TMP_DEM.tif")
    OUT_ORTHO   = path.join(wd, "OUT_ORTHO.tif")
    OUT_VALID   = path.join(wd, "OUT_VALID.tif")
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
        
        filterbands = [i for i in range(img.RasterCount) if img.GetRasterBand(i+1).GetDescription() in bandnames.DATA_BANDS]
        if img.RasterCount == 1:
            amp2e4 = np.moveaxis(np.atleast_3d(img.ReadAsArray()), 2, 0)[filterbands,:,:] 
        else:
            amp2e4 = img.ReadAsArray()[filterbands,:,:]
         
        #pow = np.square(amp2e4 / 2e4, dtype='float64')

        for band_i in range(0, img.RasterCount):
            print("band: {}".format(band_i + 1))
            # create filtered data and write to file
            filtered = lee_filter2(amp2e4[band_i, :, :], window=(3,3))
            #amp2e4[:] = np.sqrt(filtered) * 2e4
            img.GetRasterBand(band_i + 1).WriteArray(filtered[:, :])
   
        del img
        ## End test filter block
        
        print("{:#^84}".format('  Filtering Complete  '))
    
    ## Get DEM and orthorectify RS2
    #===============================
    print("{:#^84}".format('  Begin Orthorectification  '))
            
    # load image
    img = gdal.Open(tif, gdal.GA_ReadOnly)
    pol = [img.GetRasterBand(i + 1).GetDescription() for i in range(img.RasterCount)]
    
    # get image extent (xmin, xmax, ymin, ymax) check for existence of kml
    extent = de.get_spatial_extent(tif)

    if extent['ymax'] > 58:
        product = 'CDED'

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
        
    # write
    orth.write(OUT_FINAL)
    
    print("{:#^84}".format('  VRT  Complete '))

    ## LABEL BANDS
    # =============================
    print("{:#^84}".format('  Assign band metadata  '))

    img = gdal.Open(OUT_FINAL, gdal.GA_Update)
    # for (i, description) in enumerate(pol + ['Valid Data Pixels', 'Slope', 'TPI']):
    for (i, description) in enumerate(pol + ['Valid Data Pixels']):
        img.GetRasterBand(i + 1).SetDescription(description)
    del img
    print("{:#^84}".format('   Band metadata complete  '))

    ## ZIP
    #==============================
    
    print("{:#^84}".format('  Zipping output  '))
    
    zip_out = re.sub("vrt", "tar", OUT_FINAL)
    with tarfile.open(zip_out, 'a') as myzip:
        for file in [OUT_ORTHO, OUT_VALID, OUT_FINAL]:
            print("Adding file to archive: {}".format(path.basename(file)))
            myzip.add(file, arcname=path.basename(file))
    print("{:#^84}".format('   Zipping complete  '))
    
    ## CLEAN UP
    #================= 
    if cleanup:
        print("{:#^84}".format('  Begin File cleanup  '))
        
        for file in [TMP_DEM, OUT_ORTHO, OUT_VALID,
                    valid_vrt, ortho_vrt, OUT_FINAL]:
            if os.path.isfile(file):
                os.remove(file)
            
        os.remove(tif)
        os.rename(backup_name, tif)
        
        print("{:#^84}".format('  File cleanup complete  '))

    return(zip_out)

def preproS1(folder, DEM_dir, cleanup=True, product="CDED"):
    """ Preprocess Sentinel-1 file in preparation for classification

    *Parameters*

    folder : str
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

    os.environ['OTB_MAX_RAM_HINT'] = '2000'

    TMP_DEM = path.join(folder, "TMP_DEM.tif")
    OUT_ORTHO = path.join(folder, "OUT_ORTHO.tif")
    OUT_VALID = path.join(folder, "OUT_VALID.tif")
    OUT_TMP = path.join(folder, "OUT_TMP.tif")
    DEM_FOLDER = path.join(folder, "DEM_FOLDER")

    OUT_FINAL = path.join(folder, path.basename(folder) + ".vrt")

    merge_files = []
    imagery_files = [f.strip() for f in re.findall(".*tiff", gdal.Info(manifest))]

    img = gdal.Open(manifest, gdal.GA_ReadOnly)
    pol = [img.GetRasterBand(i + 1).GetDescription() for i in range(img.RasterCount)]
    del img

    ## CHECK FOR COMPLEX VALUES
    # =================
    #print("{:#^84}".format('  Check for any complex values (SLC) and convert  '))
    #complex = ProcessSLC(product_xml)
    #print("{:#^84}".format('  Done!  '))

    ## CALIBRATE & FILTER IMAGERY
    # =================

    for i, datafile in enumerate(imagery_files):
        print(f"Calibrating Band {i+1}")
        calibrateS1(datafile)

        img = gdal.Open(datafile, gdal.GA_Update)

        amp2e4 = np.moveaxis(np.atleast_3d(img.ReadAsArray()), 2, 0)[:, :, :]

        print(f"Filtering Band {i+1}")
        # create filtered data and write to file
        filtered = lee_filter2(amp2e4[0, :, :], window=(3, 3))
        # amp2e4[:] = np.sqrt(filtered) * 2e4
        img.GetRasterBand(1).WriteArray(filtered[:, :])

        del img
    # == end PSPOLFIL

    command = f'otbcli_ConcatenateImages -il {" ".join(imagery_files)} -out {OUT_TMP}'
    print(command)
    os.system(command)

    print("{:#^84}".format('  Calibration and Filtering Complete  '))

    ## Get DEM and orthorectify RS2
    # ===============================
    print("{:#^84}".format('  Begin Orthorectification  '))


    # get image extent (xmin, xmax, ymin, ymax) check for existence of kml
    extent = de.get_spatial_extent(manifest)

    print("spatial extent of the images " + str(extent['xmax']))

    if extent['ymax'] > 58:
        product = 'CDED'

    # build dem
    de.create_DEM_mosaic_from_extent(extent, dstfile=TMP_DEM,
                                     DEM_dir=DEM_dir, product=product)

    tmpReproj = path.join(folder, "TMP_REPROJDEM.tif")
    gdal.Warp(tmpReproj, TMP_DEM, dstSRS='EPSG:4326')
    os.remove(TMP_DEM)
    os.rename(tmpReproj, TMP_DEM)

    gdal.Warp(OUT_ORTHO, manifest, dstSRS='EPSG:4326')

    gr = gdal.Open(OUT_ORTHO)  # Grap output pixel spacing, will be gridspacing for ortho
    gt = gr.GetGeoTransform()
    gsx = gt[1]
    gsy = gt[5]

    del (gr)
    os.remove(OUT_ORTHO)

    print(gsx)

    os.makedirs(DEM_FOLDER)
    shutil.move(TMP_DEM, DEM_FOLDER)

    """
    TODO: Depreciated! 
    """
    orthorectify_otb(OUT_TMP, OUT_ORTHO, DEM_FOLDER, gsx)


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


    # write
    orth.write(OUT_FINAL)

    print("{:#^84}".format('  VRT  Complete '))

    ## LABEL BANDS
    # =============================
    print("{:#^84}".format('  Assign band metadata  '))

    img = gdal.Open(OUT_FINAL, gdal.GA_Update)
    # for (i, description) in enumerate(pol + ['Valid Data Pixels', 'Slope', 'TPI']):
    for (i, description) in enumerate(pol + ['Valid Data Pixels']):
        img.GetRasterBand(i + 1).SetDescription(description)
    del img
    print("{:#^84}".format('   Band metadata complete  '))

    ## ZIP
    # ==============================

    print("{:#^84}".format('  Zipping output  '))

    zip_out = re.sub("vrt", "tar", OUT_FINAL)
    with tarfile.open(zip_out, 'a') as myzip:
        for file in [OUT_ORTHO, OUT_VALID, OUT_FINAL]:
            print("Adding file to archive: {}".format(path.basename(file)))
            myzip.add(file, arcname=path.basename(file))
    print("{:#^84}".format('   Zipping complete  '))

    ## CLEAN UP
    # =================
    if cleanup:
        print("{:#^84}".format('  Begin File cleanup  '))

        for file in [TMP_DEM, OUT_ORTHO, OUT_VALID,
                     valid_vrt, ortho_vrt, OUT_FINAL]:
            if os.path.isfile(file):
                os.remove(file)

        print("{:#^84}".format('  File cleanup complete  '))

    return (zip_out)