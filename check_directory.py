#!/usr/bin/env python
""" 
Checks a directory for the existence of certain files. 
"""

import configparser
import os

from SSWM import utils
from launch_preprocess import preParamConfig
from launch_forest import forestClassifier

if __name__ == "__main__":
    Config = configparser.ConfigParser()
    ##User input##
    # Get keywords from an INI file
    config = '' #Full path to classify.ini
    task = ''  #Either LaunchPreprocessor or LaunchClassifier as string
    ##User Input End##

    Config.read(config)
    # Directory monitor keywords
    folder = Config.get(task, 'watch_folder')

    DIRMON = utils.filedaemon(folder=folder)

    """
    Write a manifest with all the files that are there.  The maestro job will 
    get the next file to process from this manifest, and terminate when the 
    manifest is empty. If a manifest exists already, the job quits so that we
    don't get multiple maestro jobs operating on the same files.
    This technique can probably be streamlined
    """
    manifest = os.path.join(DIRMON.folder, "manifest.txt")
    DIRMON.check_completion(DIRMON.folder)
    
    if os.path.isfile(manifest):
        DIRMON.check_completion(DIRMON.folder)  #clear out the manifest file if its empty
        print('Previous job not yet complete')
        exit(0)
    else:
        result = DIRMON.check()
        if not result:
            print("No items to process!")
            exit(0)

        DIRMON.write_manifest(manifest)

        while True:
            if task == 'LaunchPreprocessor':
                preParamConfig(config)
            elif task == 'LaunchClassifier':
                forestClassifier(config)
            else:
                exit(0)
