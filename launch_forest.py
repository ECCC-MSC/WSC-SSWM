"""
This script is used to create a hdf5 file from an image, train a random
forest clasifier and then classify the image
"""

import argparse
import configparser
import logging
import os
import shutil
import sys
import tarfile
import zipfile

from SSWM.trainingTesting.PixStats import PixStats
from SSWM.utils import filedaemon, bandnames
from SSWM.forest import forest, postprocess


def get_cur_file(folder):
    """ Read current file from textfile manifest
    
    *Parameters*

    folder : str
        Path to job directory containing manifest and preprocessed image archives
    
    *Returns*

    tuple
        (1) Path at which to create next file to process
        (2) Directory to which the archived files should be extracted
    """
    manifest = os.path.join(folder, 'manifest.txt')
    zip_file = filedaemon.manifest_get_next(manifest)

    exdir = os.path.splitext(zip_file)[0]
    fname = os.path.splitext(os.path.basename(zip_file))[0]
    cur_file = os.path.join(exdir, fname + '.vrt')    
    
    return cur_file, exdir
    
def untar_VRT(cur_file):
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
    elif ext == '.zip':
        fh = zipfile.ZipFile(cur_file)
        
    fh.extractall(path=wkdir)
    vrt = os.path.join(wkdir, os.path.splitext(os.path.basename(cur_file))[0] + ".vrt")
    
    return vrt, wkdir

def clean_up(extracted_directory):
    """ Remove extracted files and move archive to backup directory """
    shutil.rmtree(extracted_directory) #  folder
    archive = extracted_directory + '.tar'
    os.remove(archive)

def failure(exdir):
    clean_up(exdir)
    sys.exit(0)

            
def forestClassifier(config):
    # Load configuration file
    Config = configparser.ConfigParser()
    Config.read(config)
    
    # Set up message logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logfile = os.path.join(Config.get('Generic', 'log_dir'), "forest.log")
    logging.basicConfig(filename=logfile, level=logging.INFO,
                        format='%(asctime)s - %(name)s - [%(levelname)s] %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())
    logger = logging.getLogger()
    
    # Classifier keywords
    gsw_path          = Config.get('Classification', 'gsw_path')
    training_data     = Config.get('Classification', 'training_data')
    images_output_dir = Config.get('Classification', 'output')
    num_procs         = Config.getint('Classification', 'num_procs')
    npz_dir           = Config.get('Classification', 'tmp')
       
    # Get current file


    folder = Config.get('LaunchClassifier', 'watch_folder')
    vrt, exdir = get_cur_file(folder)
    if not vrt:
        sys.exit(0)
    vrt = os.path.join(folder, vrt)
    exdir = os.path.join(folder, exdir)

    archive = exdir + '.tar'
    cur_file, exdir = untar_VRT(archive)
    logging.info(f"opening archive from manifest: {cur_file}")
    
    scene_id = os.path.splitext(os.path.basename(cur_file))[0]
    
    """
    # Create HDF5 for training, skip if it already exists 
    #====================================================
    output_h5 = scene_id + ".h5"
    output_h5 = os.path.join(training_data, output_h5)
    
    P = PixStats(cur_file, 
                 output_dir=training_data, 
                 gsw_path=gsw_path, 
                 images_output_dir=images_output_dir, 
                 fst_converter_path=npz_dir)
                 
    if not os.path.isfile(output_h5):   
        for band in bandnames.DATA_BANDS:
            if not band in P.valid_bands:
                P.valid_bands.append(band)
        P.available_bands = P.get_valid_bands()
        valid = P.get_stats(write_water_mask=False)
            
    else:
        logging.info("h5 file already exists - skipping creation of new")
    
  
    # Select training data, fit model and get stats
    #==============================================
    if os.path.isdir(Config.get('Classification', 'train_on')):
        pass
        # get file list from dir and 
    elif os.path.isfile(Config.get('Classification', 'train_on')):
        training_file = Config.get('Classification', 'train_on')
    else:
        training_file = output_h5
    """


        
    output_basename = os.path.join(images_output_dir, scene_id)
    output_report = os.path.join(images_output_dir, scene_id + '.txt')
    test_report = os.path.join(images_output_dir, scene_id + '_testreport.txt')
    
    RF = forest.waterclass_RF(n_estimators=500, criterion='entropy', oob_score=True, n_jobs=-1)
    
    try:
        #RF.train_from_h5(training_file, nland=7500, nwater=2500, eval_frac=0.25)
        RF.train_from_image(cur_file, exdir, gsw_path, nland=750, nwater=2500, eval_frac=0.25)

    except ZeroDivisionError as e:
        msg = ("No overlapping water pixels were found in this scene."
                "Classification for this image was not performed.")
        logging.error(msg)
        failure(exdir)
 
    RF.rf.num_procs = num_procs
    RF.save_evaluation(output_report)
    #RF.test_from_h5(output_h5, nwater=625, output=test_report)
    
    if RF.results['m']['F1'] < bandnames.MIN_F1:
        msg = ("Poor classification quality found during model fitting"
                " (F1 < {}). "
                "Classification for this image was not performed. Change F1 threshold in the "
                "'bandnames' class (DUAP/utils.py)".format(bandnames.MIN_F1))
        logging.warning(msg)
        failure(exdir)
        
    # Classify image
    #================
    output_img = output_basename + '.tif'
    RF.predict_chunked(cur_file, output_img, 3000) 

    
    # Postprocess to remove false positives
    #=======================================
    pythonexe = Config.get('Postprocess', 'python3')
    gdalpolypath = Config.get('Postprocess', 'polygonize')
    output_polygon = output_basename + "_classified_filt.gpkg"
    low_estimate = output_basename + "_classified_filt.tif" # created by .postprocess()
    
    postprocess.postprocess(output_img, output_polygon, pythonexe, gdalpolypath)
    postprocess.rasterize_inplace(low_estimate, output_polygon)
    postprocess.max_filter_inplace(low_estimate, band=1, size=3) # testing
    
    
    # Clean up
    #=========
    logger.info("cleaning up")
    clean_up(exdir)
 
