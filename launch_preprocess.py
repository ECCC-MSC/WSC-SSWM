"""

"""
import configparser
import re
import shutil
import sys
import zipfile
import logging
import os
import tarfile

from SSWM.utils import filedaemon
from SSWM.preprocess.preutils import RS2, RCM, S1
from SSWM.preprocess.preprocess import preproRS2, preproRCM_bd, preproS1

def untar(cur_file, folder, s1=False):
    """ Extract files from archive

    *Returns*

    tuple
        (1) Path to extracted VRT file
        (2) Path to working directory containing VRT and image files
    """
    wkdir, ext = os.path.splitext(cur_file)
    logging.info(f"using {cur_file}")
    if ext == '.tar':
        fh = tarfile.open(cur_file, 'r')
        folder = wkdir
    elif ext == '.zip':
        fh = zipfile.ZipFile(cur_file)

    fh.extractall(path=folder)

    if len(os.listdir(folder))>1:
        if s1:
            wkdir = wkdir + '.SAFE'
            nwkdir = os.path.join(folder, os.path.basename(wkdir).split('.')[0])
            os.rename(wkdir, nwkdir)
            wkdir = nwkdir
        file = wkdir
    else:    #No folder inside extracted dir
        file = os.path.join(wkdir, os.path.splitext(os.path.basename(cur_file))[0])

    return file


def preprocess(cur_file, folder, DEM_directory, finished_result, DEMType, logger, satellite = 'RS2'):
    """ Run preprocessing on next file and move results to correct directory
    
    Gets the next file from the manifest, processes it and moves the results
    to the target folders.
    
    *Parameters*

    folder : str
        
    DEM_directory : str
        Path to directory containing DEM files in appropriate folder hierarchy
    finished_raw : str
        Path to folder where original file should be moved after processing
    finished_result : str
        Path to folder where results should be saved
    DEMType:
        DEM type the preprocessor will use (See documentation for options)
    satellite : str
        Which satellite profile should be used. One of "RS2" or "RCM" 
        
    """
    
    sat_dict = {'RS2': RS2, 
                'RCM': RCM,
                'S1': S1}
                
    R = sat_dict[satellite]

    if cur_file.endswith('.tar') or cur_file.endswith('.zip'):
        cur_zip = cur_file
        if satellite == 'S1':
            cur_file = untar(cur_file, folder, s1=True)
        else:
            cur_file = untar(cur_file, folder)
        os.remove(cur_zip)
    
    if satellite == 'RS2':
        logger.info("Starting RS2 preprocess for {}".format(cur_file))
        # unzip
        wkdir = os.path.splitext(cur_file)[0]
        zipfile.ZipFile(cur_file).extractall(path=wkdir)
    
        # find product.xml
        product_xml = R.path_to_xml(wkdir)
    
        # do preprocessing
        zip_out = preproRS2(product_xml, DEM_directory, product=DEMType)
        clean = lambda : shutil.rmtree(wkdir)

    elif satellite == 'RCM':
        logger.info("Starting RCM preprocess for {}".format(cur_file))
        zip_out = preproRCM_bd(cur_file, DEM_directory, product=DEMType)
        clean = lambda *args: None

    elif satellite == 'S1':
        logger.info("Starting S1 preprocess for {}".format(cur_file))
        zip_out = preproS1(cur_file, DEM_directory, product=DEMType)
        clean = lambda *args: None
    
    # move zipfiles and clean up
    shutil.move(zip_out, os.path.join(finished_result, os.path.basename(zip_out)))
    shutil.rmtree(cur_file)
    clean()

    return os.path.join(finished_result, os.path.basename(zip_out))


def preParamConfig(config, cur_file):
    Config = configparser.ConfigParser()
    Config.read(config)

    # preprocessor keywords
    folder          = Config.get('Directories', 'watch_folder')
    DEM_dir         = Config.get('Directories', 'DEM_dir')
    finished_result = Config.get('Directories', 'TMP')
    sat             = Config.get('Params', 'satellite_profile')
    DEMType         = Config.get('Params', 'DEMType')

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logfile = os.path.join(Config.get('Generic', 'log_dir'), "preprocess.log")
    logging.basicConfig(filename=logfile, level=logging.INFO,
                        format='%(asctime)s - %(name)s - [%(levelname)s] %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())
    logger = logging.getLogger()

    zip_out = preprocess(cur_file, folder, DEM_dir, finished_result, DEMType, logger,  satellite=sat)

    return zip_out