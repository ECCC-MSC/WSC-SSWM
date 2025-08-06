#!/usr/bin/env python
""" 
Checks a directory for the existence of certain files. 
"""

import configparser
import os

from launch_preprocess import preParamConfig
from launch_forest import forestClassifier

if __name__ == "__main__":
    Config = configparser.ConfigParser()
    ##User input##
    # Get keywords from an INI file
    config = r'PATH TO CONFIG' #Full path to classify.ini
    ##User Input End##

    Config.read(config)
    # Directory monitor keywords
    folder = Config.get('Directories', 'watch_folder')
    print(folder)

    for product in os.listdir(folder):
        #First, preprocess
        result = preParamConfig(config, os.path.join(folder, product))

        #Then, classify
        #forestClassifier(config, os.path.join(folder, product))
        forestClassifier(config, result)

