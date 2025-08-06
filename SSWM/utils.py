import glob
import logging
import numpy as np
import os
import re

logger = logging.getLogger(__name__)


class filedaemon:
    """ Check a directory for files and execute a function if any files exist 
    
    *Parameters*

    folder : str
        Path to folder which will be checked for files
    action : function 
        python function that acccepts a folder as its first positional argument 
        and any number of keyword arguments.
    ext :str, optional
        regex-style pattern which file names must match to be considered 
        (e.g. "\\.tif" would include only tif files. 
    """

    def __init__(self, folder):
        self.folder = folder

    def check(self):
        """ Test whether or not files are present in the directory
        
        *Returns*

            True if files exist, False otherwise.
        """
        file_list = glob.glob(os.path.join(self.folder, "*"))
        file_list = [os.path.basename(f) for f in file_list]

        '''
        if self.ext is not None:
            reg = re.compile(self.ext)
            file_list = filter(reg.search, file_list)
        '''
        if file_list:
            self.file_list = [os.path.join(self.folder, f) for f in file_list]
            if len(self.file_list) > 0:
                return(True)
    
        return(False)
            
    def write_manifest(self, file):
        """ writes file list to a manifest.txt file """
        
        print(self.file_list)
        if len(self.file_list) != 0:
            self.__writemf(file, self.file_list)
        else:
            logger.info("No valid files in directory")
        #lst = [i + ',0' for i in self.file_list]
        #self.__writemf(file, lst)
        
    @ staticmethod
    def __writemf(file, list_obj):
        """ helper function for write_maifest """
        np.savetxt(file, list_obj, delimiter=",", fmt='%s', header="")
        
    @classmethod
    def manifest_get_next(cls, file):
        """ gets first item in manifest and updates the manifest """
        if not os.path.isfile(file):
            return(None)
            
        data = np.genfromtxt(file, dtype=str, delimiter=",")
        data = np.atleast_1d(data)
        if len(data) == 0:
            os.remove(file)
            return(None)
        
        else:
            next_item, data = data[0], data[1:]
            cls.__writemf(file, data)
        
        return(next_item)

    @classmethod
    def manifest_get_index(cls, file, index):
        """ gets item in manifest from index """
        if not os.path.isfile(file):
            return(None)
        
        data = np.genfromtxt(file, dtype=str, delimiter=",")
        data = np.atleast_1d(data)
        next_item = data[index]
        
        return(next_item)
    
    @classmethod
    def check_completion(cls, folder):
        """ Delete manifest if it is empty """
        m_file = os.path.join(folder, 'manifest.txt')
        
        if not os.path.isfile(m_file):
            return
            
        manifest = np.atleast_1d(np.genfromtxt(m_file, dtype=str, delimiter=","))
        manifest = [os.path.basename(x) for x in manifest]
        files_remaining = set(os.listdir(folder)).intersection(set(manifest))
        logging.info(f'files remaining : {files_remaining}')
        
        if not files_remaining:
            os.remove(m_file)


class bandnames:
    """ Base class for classification tasks to standardize band names. 
    
    This is used to set the names of the bands that are used in the 
    classifications.  It can also be used to turn various data bands on or off
    in the classification. Bands will be written in the order they are specified 
    in the DATA_BANDS attribute. This also controls the order of terms in the vector
    that is passed to the classifier for each pixel.
    
    *Attributes*

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
    
    DATA_BANDS = ['HH', 'HV','VV', 'VH', 'RH', 'RV', 'SE_I', 'SE_P']
    
    VALID_PIX_BAND = ['Valid Data Pixels']
    
    MIN_F1 = 0.6
    
