"""
For RandomForest water classification of radar images
"""

import csv
from osgeo import gdal
import numpy as np
import pandas as pd
import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from SSWM.trainingTesting.PixStats import PixStats
from SSWM.preprocess.preutils import write_array_like, cloneRaster
from SSWM.utils import bandnames

logger = logging.getLogger(__name__)


class waterclass_RF(bandnames):
    """ RandomForest classifier for open-water classification of (radar) images 
    
    *Parameters*
    
    **rfargs : 
        keyword arguments passed to sklearn.ensemble.RandomForestClassifier
    """

    def __init__(self, **rfargs):
        self.rf = self._make_tree(**rfargs)
        self.results = dict()

    def train_from_image(self, cur_file, exdir, gsw_path, valseed, nland=1000, nwater=1000, eval_frac=0.2):
        """ Train a random forest by sampling directly from an image
        Optionally set aside some of the sample for evaluation

        **Parameters**

        cur_file : str
            path to image file
        exdir: str
            path to dataset directory
        gsw_path : str
            path to gsw files needed to create mask for training
        valseed : int
            seed for random sampling
        nland : int
            number of land pixels to sample
        nwater : int
            number of water pixels to sample
        eval_frac : float
            fraction between 0 and 1 that should be set aside for evaluation
        """

        logger.info("Beginning train from image")
        trainingdataset = training_dataset()

        trainingdataset.sample_from_image(cur_file, exdir, gsw_path, valseed, nland, nwater, eval_frac,
                                          max_L2W_ratio=10)
        logger.info("Creation of training data complete!")

        self.training_data = trainingdataset.training_data
        self.training_targets = trainingdataset.training_targets
        self.testing_data = trainingdataset.testing_data
        self.testing_targets = trainingdataset.testing_targets

        del trainingdataset

        self.rf.fit(self.training_data, self.training_targets)
        
    def evaluate(self):
        """ Evaluate the current random forest model using current test data """
        
        # confusion matrix
        predictions = self.rf.predict(self.testing_data)
        cm = pd.crosstab(self.testing_targets, predictions, 
                         rownames=['Actual Water'], 
                         colnames=['Predicted Water'])
        
        self.results['cm'] = cm
        
        # metrics
        M = metric(self.testing_targets, predictions)
        self.results['m'] = M
        
        # feature importance
        features = list(self.testing_data.columns)
        fi = {f:i for (f,i) in zip(features, np.round(self.rf.feature_importances_, 2))}
        self.results['fi'] = fi
        
    def save_evaluation(self, file):
        """ Save current evaluation statistics to a text file"""
        self.evaluate()
        self.results['m'].add_dict(self.results['fi'], 'importances')
        self.results['m'].save_report(file)
        
    def predict_features(self, imfile, outfile):
        """ Use current RF model to produce binary classification of an image """
        ch = imgchunker(imfile, -1)
        chu = ch.chunkerator()
        array, offset = next(chu)
        features = ch.reshape_chunk(array)
        predictions = self.rf.predict(features[:,:-1])
        predictions = predictions.reshape(ch.last_chunk_shape)

        write_array_like(imfile, outfile, predictions, dtype=gdal.GDT_Byte)
        
    def predict_probabilities(self, imfile, outfile):
        """ Use current RF model to produce probabilistic classification of an image """
        ch = imgchunker(imfile, -1)
        chu = ch.chunkerator()
        array, offset = next(chu)
        features = ch.reshape_chunk(array)
        true_ix = np.argmax(self.rf.classes_) #which ix is probability of 'True'
        predictions = self.rf.predict_proba(features[:,:-1])[:,true_ix]
        predictions = predictions.reshape(ch.last_chunk_shape)
        predictions *= 100
        predictions = np.array(predictions, dtype='int16')

        write_array_like(imfile, outfile, predictions, dtype=gdal.GDT_UInt16)
    
    def predict_chunked(self, imfile, outfile, chunksize=5000):
        """ Classify an image piece-by-piece to avoid running out of RAM
        
        *Parameters*
        
        imfile : str
            Path to input image file
        outfile : str
            Path to output raster containing probability of water
        chunksize : int
            How many rows to process at once during classification
            
        """
        logger.info("Beginning image classification in chunks")
        ch = imgchunker(imfile, chunksize)
        chu = ch.chunkerator()
        true_ix = np.argmax(self.rf.classes_) # which results are True (i.e. water)
        tif = cloneRaster(imfile, outfile, ret=True, all_bands=False, coerce_dtype=2)
        bnd = tif.GetRasterBand(1)
        
        for array, offset in chu:
            features = ch.reshape_chunk(array)
            # TEMP FIX TO NAN ISSUE!!!
            ################
            np.nan_to_num(features, copy=False, posinf=0, neginf=0)
            ################
            logger.info("Classifying chunk")
            # only classify valid pixels (last column is valid pixels)
            valid_ix = features[:,-1] >= 1 
            predictions = np.empty_like(valid_ix, dtype='float32')
            predictions[~valid_ix] = -1
            if any(valid_ix):
                predictions[valid_ix] = self.rf.predict_proba(features[valid_ix,:-1])[:,true_ix]
            predictions = predictions.reshape(ch.last_chunk_shape)
            predictions *= 100
            predictions = np.array(predictions, dtype='int16')
            bnd.WriteArray(predictions, yoff=offset)
            bnd.FlushCache()
        
        del bnd, tif
            
    def _make_tree(self, n_estimators=1000, n_jobs=-1, criterion='entropy', **rfargs):
        """" Initialize random forest classifier """
        
        rf = RandomForestClassifier(n_estimators=n_estimators, 
                                         criterion=criterion, 
                                         n_jobs=n_jobs, 
                                         **rfargs)
        return(rf)
                                         
        
        
    
class training_dataset(bandnames):
    """ A class to hold data for random forest training and evaluation.  
    
    Makes use of hdf5 files to store training data. 
    
    *Attributes*
    
    training_data : array-like
        features for training samples
    testing_data : array-like
        features for testing samples
    training_targets : array-like
        1-d vector of feature labels for training samples
    testing_targets : array-like
        1-d vector of feature labels for testing samples  
        
    """
    
    def __init__(self):
        self.training_data    = None
        self.testing_data     = None 
        self.training_targets = None
        self.testing_targets  = None

    def sample_from_image(self, cur_file, exdir, gsw_path, valseed, nland=1000, nwater=1000, eval_frac=0.2,
                          max_L2W_ratio=10):
        """
        sample n & m pixels water and land pixels respectivally from the scene,
        split into training and testing sets needed for RF

        **Parameters**

        cur_file : str
            path to image file
        exdir: str
            path to dataset directory
        gsw_path : str
            path to gsw files needed to create mask for training
        valseed : int
            seed for random sampling
        nland : int
            number of land pixels to sample
        nwater : int
            number of water pixels to sample
        eval_frac : float
            fraction between 0 and 1 that should be set aside for evaluation
        max_l2w_ratio : int
            maximum allowed ratio of land pixels to water pixels in the training dataset
        """
        logger.info("Cur_file {}".format(cur_file))

        P = PixStats(cur_file,
                     output_dir=exdir,
                     gsw_path=gsw_path)

        for band in bandnames.DATA_BANDS:
            if not band in P.valid_bands:
                P.valid_bands.append(band)
        P.available_bands = P.get_valid_bands()

        water_sample, land_sample = P.get_stats_and_sample(valseed, nwater, nland, max_L2W_ratio)

        # split sample into training points and test points
        training_wat, test_wat = self.split_sample(water_sample, eval_frac=eval_frac)
        training_land, test_land = self.split_sample(land_sample, eval_frac=eval_frac)

        TRAIN = np.concatenate((training_wat, training_land), axis=0)
        TEST = np.concatenate((test_wat, test_land), axis=0)

        self.training_data = pd.DataFrame(TRAIN).drop(self.MASK_LABEL[0], axis=1)
        self.testing_data = pd.DataFrame(TEST).drop(self.MASK_LABEL[0], axis=1)
        self.training_targets = TRAIN[self.MASK_LABEL[0]]
        self.testing_targets = TEST[self.MASK_LABEL[0]]

        del P

        
    def split_sample(self, sample, eval_frac=0.2):
        """ Randomly split sample into training and test subsamples 
       
        Returned samples are shuffled relative to the input sample
        
        *Parameters*
        
        sample : array-like
            an array of samples with dimension (m, n) 
        eval_frac : numeric
            fraction of rows to allocate to testing data
        
        *Returns*
        
        tuple
            Two arrays with dimension (m - j, n) and (j, n) where j~=eval_frac*m
            
        """
        c = int(eval_frac * sample.shape[0])
        np.random.shuffle(sample)
        test, training = sample[:c, ], sample[c:, ]
        
        return (training, test)
        
class imgchunker(bandnames):
    """ Splits image into chunks for memory-safer processing
    
    This object takes raster arrays of dimension (m,n,p)
    and yields 'chunks' with dimension (i, j) with 1 < i < m*n and 1 < j < p.
    The smaller chunks can then be classified without running out of memory.
    
    The last chunk is usually smaller than the rest unless by_y is chosen to
    evenly divide the number of image rows.
    
    *Parameters*
    
    img : str 
        Path to gdal-compatible raster image
    by_y : int
        How many rows should be returned during each iteration
    
    """
    
    def __init__(self, img, by_y=5000):
        self.by_y = by_y 
        self.open(img)    
    
    def build_band_dict(self, img):
        """ Get indices of image bands that will be used in classification """
        
        self.band_dict = {img.GetRasterBand(i).GetDescription():i for i in range(1, img.RasterCount + 1)}
        self.DATA_BANDS = [b for b in self.DATA_BANDS if b in list(self.band_dict.keys())]
        self.valid_data_bands = [self.band_dict[i] for i in self.DATA_BANDS + self.VALID_PIX_BAND]
        
    def open(self, img):
        """ Open an image and collect some parameters """
        
        self.img = gdal.Open(img)    
        
        if self.by_y == -1:
            self.by_y = self.img.RasterYSize
        
        self.nchunks = -(-self.img.RasterYSize // self.by_y) # upside-down floor division
        self.last_dy = self.img.RasterYSize - self.by_y * (self.nchunks - 1)
        self.imwdth = self.img.RasterXSize
        self.build_band_dict(self.img)
        self.original_shape = (self.img.RasterYSize, self.img.RasterXSize)
    
    @staticmethod    
    def get_chunk(img, ix, offx, offy, lnx, lny):
        """ Get a slice of an image for classification.
        
        Images are classified in pieces to prevent memory overflow
        
        *Parameters*
        
        ix : list 
            indices (1-based) for image bands that are to be used. 
        offx : int
            X offset from which to begin reading image. Referenced to upper left
            corner
        offy : int
            Y offset from which to begin reading image. Referenced to upper left
            corner. 
        lnx : int
            How many columns to read beginning from x offset
        lny : int
            How many rows to read beginning from y offset
            
        *Returns*
        
        array
            an array corresponding to a slice of the raster array with 
            dimensions (m,n,p) where m=lnx, n=lny and p=len(ix)
                
        """
        
        nrow = img.RasterYSize
        nchunk = (nrow // lny) + 1
        cur_chunk = offy // lny + 1
        logging.info(f"getting chunk {cur_chunk} of {nchunk} with shape ({lny}, {lnx})")
        chunk = np.empty((lny, lnx, len(ix)), dtype='float32')
        
        for i, j in enumerate(ix):
            chunk[:,:,i] = img.GetRasterBand(j).ReadAsArray(offx, offy, lnx, lny)
        
        return(chunk)
        
    def chunkerator(self):
        """ Generate image chunks for classification
        
        During the classification process, this function 'feeds' the classifier
        pieces of the input image. The last piece of the image is usually smaller
        than the rest.
        
        *Yields*
        
        tuple
            A tuple containing (1) An array corresponding to a chunk of the input
            image, and (2) the y-offset of the chunk relative to the upper-left
            corner of the original image.
        """
        c = 0

        while c < self.nchunks:
            if c == self.nchunks - 1:
                dy = self.last_dy
            else:
                dy = self.by_y
            offy = c * self.by_y
            
            chunk = self.get_chunk(img = self.img, 
                                   ix = self.valid_data_bands, 
                                   offx = 0,
                                   offy = offy,
                                   lnx = self.imwdth,
                                   lny = dy)
            c += 1
            
            yield((chunk, offy))
                 
    def reshape_chunk(self, chunk):
        """ Flatten a 3-d array so it can be fed into a random forest classifier
        
        *Parameters*
        
        chunk : array-like 
             3-d array with dimensions (m, n, p) 

        *Returns*
        
        array-like
            2-d array with dimensions (m*n, p). Each row corresponds to a pixel 
            and each column corresponds to an image band
        """
        
        logging.info(f"reshaping chunk with shape {chunk.shape}")
        out = chunk.reshape((np.prod(chunk.shape[0:2]), chunk.shape[2]))
        self.last_chunk_shape = chunk.shape[0:2]
        
        return out
        

class metric():
    """ A class to hold various statistics about a binary classification
    
    *Parameters*
    
    labels : array-like (1-d)
        vector of feature labels
    predictions : array-like (1-d)
        vector of predicted categories equal in length to labels
        
    *Example*
    
    labels = np.array([True, True, True, True, True, False, False, False])
    predictions = np.array([True, False, False, True, True, True, False, False])
    M = metric(labels, predictions) 
    print(M)
    """
    
    def __init__(self, labels, predictions):
        self.metrics = {}
        self.extras = {}
        self.calculate_metrics(labels, predictions)
    
    def __getitem__(self, key):
        return(self.metrics[key])
        
    def __repr__(self):
        s = "\n".join(["{}: {}".format(key, np.round(self.metrics[key],2)) for key in self.metrics]) 
        return(s)
        
    def calculate_metrics(self, labels, predictions):
        """ Calculates accuracy, precision, F1, recall and specificity """
        
        self.confusion_matrix(labels, predictions)
        self.metrics['ACC']  = metrics.accuracy_score(labels, predictions)
        self.metrics['PREC'] = metrics.precision_score(labels, predictions)
        self.metrics['F1']   = metrics.f1_score(labels, predictions)
        self.metrics['RECAL']   = metrics.recall_score(labels, predictions)
        self.metrics['SPEC'] = self.cm['_FN'] / (self.cm['_FP'] + self.cm['_TN'] )
        
    def confusion_matrix(self, labels, predictions):
        """ Build confusion matrix for labels and predictions """
        
        cm = pd.crosstab(labels, predictions, rownames=['Actual Water'], colnames=['Predicted Water'])
        self.cm = {}
        self.cm['_TP'] = cm[True][True]
        self.cm['_TN'] = cm[False][False]
        self.cm['_FP'] = cm[True][False]
        self.cm['_FN'] = cm[False][True]
        
    def add_dict(self, dct, name):
        """ Add custom statistics 
        
        Dictionaries are added to the 'extras' attribute are written to the 
        output file when save_report() is called.
        
        *Parameters*
        
        dct : dict
            Dictionary of statistics to add
        name : str  
            Header for set of statistics when it is written to a file
        """
        
        for key in dct:
            self.extras[name + "_" + key] = dct[key]
        
    def save_report(self, txtfile):
        """ Saves all calculated statistics to a textfile. 
        
        Includes confusion matrix, derived statistics (F1, Accuracy etc..) 
        and any custom statistics that were added.
            
        *Parameters*
       
        txtfile : str
            path to output file       
        """
        with open(txtfile, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter="=")
            for key, value in self.cm.items():
                writer.writerow(["{:5}".format(key), "{:3d}".format(value)])
            for key, value in self.metrics.items():
                writer.writerow(["{:5}".format(key), "{:6.3f}".format(value)])
            if len(self.extras.keys()) != 0: 
                for key, value in self.extras.items():
                    writer.writerow([key, value])
        
