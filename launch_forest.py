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

from SSWM.utils import filedaemon, bandnames
from SSWM.forest import forest, postprocess

    
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

def failure(output_h5, exdir, cur_file, images_output_dir, msg):
    """ Create a flag file to indicate that the processing was aborted."""
    os.remove(output_h5)
    clean_up(exdir)
    errfile = os.path.join(images_output_dir, cur_file + ".failed")
    with open(errfile, 'w') as f:
        f.writelines(msg)
    sys.exit(0)
            
def forestClassifier(config, archive):
    # Load configuration file
    Config = configparser.ConfigParser()
    Config.read(config)
    
    # Set up message logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logfile = os.path.join(Config.get('Directories', 'log_dir'), "forest.log")
    logging.basicConfig(filename=logfile, level=logging.INFO,
                        format='%(asctime)s - %(name)s - [%(levelname)s] %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())
    logger = logging.getLogger()
    
    # Classifier keywords
    gsw_path          = Config.get('Directories', 'gsw_path')
    images_output_dir = Config.get('Directories', 'output')
    num_procs         = Config.getint('Params', 'num_procs')
       
    # Get current file

    cur_file, exdir = untar_VRT(archive)
    logging.info(f"opening archive from manifest: {cur_file}")
    
    scene_id = os.path.splitext(os.path.basename(cur_file))[0]
        
    output_basename = os.path.join(images_output_dir, scene_id)
    output_report = os.path.join(images_output_dir, scene_id + '.txt')

    seed = 12345
    
    RF = forest.waterclass_RF(random_state=seed, n_estimators=250, criterion='entropy', oob_score=True, n_jobs=-1)
    
    try:
        #RF.train_from_h5(training_file, nland=7500, nwater=2500, eval_frac=0.25)
        #A max land-water ratio of 10 is hardcoded here, nland doesn't mean anything
        RF.train_from_image(cur_file, exdir, gsw_path, seed, nland=750, nwater=5000, eval_frac=0.25)

    except ZeroDivisionError as e:
        logging.error("No water pixels found in scene. Skipping image.")
        msg = ("No overlapping water pixels were found in this scene."
                "Classification for this image was not performed.")
        logging.error(msg)
        failure(exdir)
 
    RF.rf.num_procs = num_procs
    RF.save_evaluation(output_report)
    
    if RF.results['m']['F1'] < bandnames.MIN_F1:
        msg = ("Poor classification quality found during model fitting"
                " (F1 < {}). "
                "Classification for this image was not performed. Change F1 threshold in the "
                "'bandnames' class (utils.py)".format(bandnames.MIN_F1))
        logging.error(msg)
        
    # Classify image
    #================
    output_img = output_basename + '.tif'
    RF.predict_chunked(cur_file, output_img, 1000)

    del RF
    # Postprocess to remove false positives
    #=======================================
    output_polygon = output_basename + "_classified_filt.gpkg"
    low_estimate = output_basename + "_classified_filt.tif" # created by .postprocess()
    
    postprocess.postprocess(output_img, output_polygon, output_report)
    postprocess.rasterize_inplace(low_estimate, output_polygon)


    # Clean up
    #=========
    logger.info("cleaning up")
    clean_up(exdir)
 
