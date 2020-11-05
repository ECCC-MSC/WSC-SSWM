
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import h5py
import shutil
import sys
import tensorflow as tf
import time

from osgeo import gdal,osr
from collections import deque
from datetime import datetime
from SSWM.trainingTesting.TrainingTestingutils import ModelModes

matplotlib.use('agg')

tf.logging.set_verbosity(tf.logging.INFO)

printt = print
def print(*args,**kwargs):
    printt(*args,flush=True,**kwargs)

def consume(iterator, n=None):
    """"Advance the iterator n-steps ahead.
        If n is none, consume entirely.
        From python.org manual 9.7.2
    """
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(slice(iterator, n, n), None)

def rebin(a, new_shape):
    """
    Resizes a 2d array by averaging or repeating elements, 
    new dimensions must be integral factors of original dimensions
    
    *Parameters*
    
    a : array_like
        Input array.
    new_shape : tuple of int
        Shape of the output array
    
    *Returns*
    
    rebinned_array : ndarray
        If the new shape is smaller of the input array, the data are averaged, 
        if the new shape is bigger array elements are repeated
    
    *See Also*
    
    resize : Return a new array with the specified shape.
    
    *Examples*
    
    >>> a = np.array([[0, 1], [2, 3]])
    >>> b = rebin(a, (4, 6)) #upsize
    >>> b
    array([[0, 0, 0, 1, 1, 1],
           [0, 0, 0, 1, 1, 1],
           [2, 2, 2, 3, 3, 3],
           [2, 2, 2, 3, 3, 3]])
    >>> c = rebin(b, (2, 3)) #downsize
    >>> c
    array([[ 0. ,  0.5,  1. ],
           [ 2. ,  2.5,  3. ]])
    """
    M, N = a.shape
    m, n = new_shape
    if m<M:
        return a.reshape((m,M/m,n,N/n)).mean(3).mean(1)
    else:
        return np.repeat(np.repeat(a, m/M, axis=0), n/N, axis=1)

def bin_ndarray(ndarray, new_shape, operation='sum'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.
    Keeps the dtype of the input array
    Number of output dimensions must match number of input dimensions.
    
    *Example*
    
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)
    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]
    """
    
    if not operation.lower() in ['sum', 'mean', 'average', 'avg']:
        raise ValueError("Operation {} not supported.".format(operation))
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(ndarray.shape,
                                                           new_shape))
    compression_pairs = [(d, c//d) for d, c in zip(new_shape,
                                                   ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    print(f'Flattened=>{flattened}')
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        if operation.lower() == "sum":
            ndarray = ndarray.sum(-1*(i+1))
        elif operation.lower() in ["mean", "average", "avg"]:
            ndarray = ndarray.mean(-1*(i+1))
    return ndarray
    

class HDF5Reader:
    def __init__(self,file_path):
        self.file_path = file_path
        self.group_basename = os.path.splitext(os.path.basename(file_path))[0].replace('_eval','').replace('_test','')
        self.water_presence = None
        self.land_presence = None
        self.water_pixels = {}
        self.land_pixels_start_index = 0
        self.pol_one_hot = {}
        
        
    def open(self,file_path=None):
        if not file_path:
            file_path = self.file_path
        self.hfh = h5py.File(file_path,'r')
        
    def get_beam_mode(self,group=None):
        if not group:
            group = self.group_basename
        self.beam_mode = np.array(self.hfh[group].attrs['beam mode'],dtype=np.int8)
        return self.beam_mode
    
    def get_classified_pixels_idx(self,group=None):
        if not group:
            group = self.group_basename
        self.water_presence_mask = np.array(self.hfh[f'/{group}/water_mask'],dtype=bool)
        #Get the indexes and shuffle them:
        self.water_presence = np.nonzero(self.water_presence_mask)
        np.random.shuffle(self.water_presence[0])
        self.land_presence = np.nonzero(~self.water_presence_mask)
        np.random.shuffle(self.land_presence[0])
        self.nb_water_pixels = self.water_presence[0].shape[0]
        
    def get_h5_data(self,group,dataset,idx=None):
        return self.hfh[group][dataset][:][idx][:,np.newaxis]
        
        
    def get_pol_data(self,group,pol,nb_land_pixels):
        self.nb_array_rows = nb_land_pixels+self.nb_water_pixels
        # incidence_angle,value,classification
        sar_array = np.zeros((self.nb_array_rows,1),dtype=[('incidence_angle', '<f4'), ('value', '<f4'), ('classification', 'i1')])
        #These will always be returned:
        sar_array[:self.nb_water_pixels]['incidence_angle'] = self.water_pixels.setdefault('incidence_angle',self.get_h5_data(group,'incidence_angle',self.water_presence))
        sar_array[:self.nb_water_pixels]['value'] = self.water_pixels.setdefault(pol,self.get_h5_data(group,pol,self.water_presence))
        sar_array[:self.nb_water_pixels]['classification'] = True
        
        # Pick random land pixels
        land_pixels_idx = self.land_presence[0][self.land_pixels_start_index:self.land_pixels_start_index+nb_land_pixels]
        
        sar_array[self.nb_water_pixels:]['incidence_angle'] = self.get_h5_data(group,'incidence_angle',land_pixels_idx)
        sar_array[self.nb_water_pixels:]['value'] = self.get_h5_data(group,pol,land_pixels_idx)
        sar_array[self.nb_water_pixels:]['classification'] = False
        self.land_pixels_start_index+=nb_land_pixels
        
        return sar_array  
        
    def pick_random_data(self,group=None,min_water_ratio=15,max_water_ratio=60):
        """
        *Parameters*
        	group: name of the hdf group where the data is
        	min_water_ratio: minimum water/total pixels ratio in percent
        	max_water_ratio: maximum water/total pixels ratio in percent 
        """
        self.open()
        if not group:
            group = self.group_basename
        if self.water_presence is None:
            self.get_classified_pixels_idx(group)
        ratio = random.randint(min_water_ratio,max_water_ratio)/100.
        print(f'Ratio: {ratio}')
        total_pixels = int(self.nb_water_pixels/ratio)
        print(f'Total pixels in file random sample: {total_pixels} for each polarizations available')
        nb_land_pixels = total_pixels-self.nb_water_pixels

        sar_data=None
        pol_data=None
        for pol in ('HH','VV','HV','VH'):
            if pol in self.hfh[group].keys():
                #beam mode one hot, pol one hot , incidence angle , value , class
                pol_one_hot=np.array(self.hfh[group][pol].attrs['one_hot'],dtype=np.int8)
                if sar_data is not None:
                    s_data = self.get_pol_data(group,pol,nb_land_pixels)
                    p_data = np.tile(pol_one_hot,(s_data.shape[0],1))
                    pol_data = np.vstack((pol_data,p_data))
                    sar_data = np.vstack((sar_data,s_data))
                else:
                    sar_data = self.get_pol_data(group,pol,nb_land_pixels)
                    pol_data = np.tile(pol_one_hot,(sar_data.shape[0],1))
        return self.get_beam_mode(),pol_data,sar_data.view(np.recarray)

class HDF5Feeder:
    
    def __init__(self,hdf_directory,global_time=None):
        self.files_path=hdf_directory
        available_files = [f.path for f in os.scandir(hdf_directory) if f.name.lower().endswith('.h5')]
        self.source_files = [f for f in available_files if 'eval' not in f.lower() and 'test' not in f.lower()]
        self.eval_file = [f for f in available_files if 'eval' in f.lower()][0]
        self.test_file = [f for f in available_files if 'test' in f.lower()][0]
        self.groups=None
        self.batch=None
        self.data_renew=0
        if global_time is None:
            self.global_time = time.time()
        else:
            self.global_time=global_time
        
    def pick_random_files(self):
        print('Getting data using random files,water/land ratios and pixels')
        print('This will take a while...')
        t=time.time()
        random_files = random.sample(self.source_files,random.randint(3,6))
        #random_files = random.sample(self.source_files,1)
        random_sar_data = []
        print('Data will come from following files:')
        print('\n'.join(random_files))
        for f in random_files:
            print(80*'-')
            print(f'Getting data from {f}')
            print(80*'-')
            reader = HDF5Reader(f)
            random_sar_data.append(reader.pick_random_data())
        print(f'Done getting the random training data [time to process: {time.time()-t}s ; global time = {time.time()-self.global_time}]')
        return random_sar_data
    
    def get_batch_generator(self,yielded=False,water_weight=1.0):
        print(80*'-')
        print('Sending train data')
        print(80*'-')
        random_sar_data = self.pick_random_files()
        while (time.time() - self.global_time) <= 11500:
            source = list(random.choice(random_sar_data))
            samples = random.sample(range(source[1].shape[0]),1000)
            # source[0]: beam_mode <scalar>
            # source[1]: pol one hot , <col>
            # source[2]: sar_data <recarray> 'incidence_angle','value','classification'
            # int8,
            source[1]=source[1][samples]
            source[2]=source[2][samples]
            try:
                yield (((source[0],source[1][j],source[2].incidence_angle[j][0],source[2].value[j][0],water_weight),source[2].classification[j][0]) for j in range(1000))
                yielded = True
            except Exception as exc:
                if yielded and (time.time() - self.global_time) <= 10000:
                    # We have gone through the whole generator and still have time to load another loop
                    del random_sar_data
                    print('Generator exhausted, calling new batches of random train data')
                    random_sar_data = self.pick_random_files()
                else:
                    print('Unknown Exception')
                    print(type(exc))
                    print(exc)
                    return
            if ((time.time() - self.global_time) // 3600) > self.data_renew:
                print('Renewing data samples after 1 hour of training')
                self.data_renew+=1
                del random_sar_data
                random_sar_data = self.pick_random_files()
        return
                
    def get_test_or_eval_data(self,data_file=None,flag='eval'):
        print(80*'-')
        print(f'Sending random {flag} data')
        print(f'Eval file: {self.eval_file}')
        print(80*'-')
        # pick random data from eval file:
        reader = HDF5Reader(data_file)
        source = list(reader.pick_random_data())
        #self.get_beam_mode(),pol_data,sar_data.view(np.recarray)
        source[0] = np.tile(source[0],(source[1].shape[0],1))
        return source
                
    def get_random_eval_data(self):
        return self.get_test_or_eval_data(data_file=self.eval_file,flag='eval')
    
    def get_random_test_data(self):
        return self.get_test_or_eval_data(data_file=self.test_file,flag='test')
    

class TmpCorrector:
    # In case needed in the future
    # Name : (uniqueid,(min_angle,max_angle))
    beam_modes = {'S1':(100,(19,53)),
             'S2':(102,(19,53)),
             'S3':(104,(19,53)),
             'S4':(106,(19,53)),
             'S5':(108,(19,53)),
             'S6':(110,(19,53)),
             'S7':(112,(19,53)),
             'W1':(200,(19,45)),
             'W2':(202,(19,45)),
             'W3':(204,(19,45)),
             'W4':(206,(19,45)),
             'F0W1':(208,(19,45)),
             'F0W2':(210,(19,45)),
             'F0W3':(212,(19,45)) } 
    
    def __init__(self,file_path,group_name=None):
        self.file_path = file_path
        if not group_name:
            self.group_basename=os.path.splitext(os.path.basename(file_path))[0].replace('_new','')
        else:
            self.group_basename=group_name
        
    def open(self):
        try:
            open(self.file_path)
            self.hfh = h5py.File(self.file_path,'r+')
        except:
            self.hfh = h5py.File(self.file_path, 'w')
        
    def get_beam_mode(self):
        base_name= os.path.basename(self.file_path)
        beam_mode_str = base_name.split('_')[1]
        #beam_mode_id = self.beam_modes[beam_mode_str][0]
        #one_hot_beam_mode=self.to_one_hot_beam(beam_mode_id)
        one_hot_beam_mode = np.array([0 if b != beam_mode_str else 1 for b in self.beam_modes])
        return one_hot_beam_mode
    
    def read_beam_mode(self):
        self.open()
        return self.hfh[self.group_basename].attrs['beam mode']
        #/fs/site2/dev/eccc/oth/nhs/jfi007/suites/General_Loop_Task/run/output/
        #oSK_F0W3_20150804_130935_HH_HV_SLC_pspolfil_real_qa_new.h5
        #/fs/site2/dev/eccc/oth/nhs/jfi007/suites/General_Loop_Task/run/output/oSK_F0W2_20140601_132220_HH_HV_SLC_pspolfil_real_qa_new.h5
    
    def correct_beam_mode(self):
        required_beam_mode = self.get_beam_mode()
        actual_beam_mode = self.read_beam_mode()
        print(self.hfh)
        print(f'Checking beam mode')
        print(f'File: {self.file_path}')
        if (required_beam_mode != actual_beam_mode).any():
            print(f'Changing beam mode from {actual_beam_mode} to {required_beam_mode}')
            self.hfh[self.group_basename].attrs['beam mode'] = required_beam_mode
            
    

class PixStats:
    # <ID>, <(min,max) incidence angle>
    beam_modes = {'S1':(100,(19,53)),
                 'S2':(102,(19,53)),
                 'S3':(104,(19,53)),
                 'S4':(106,(19,53)),
                 'S5':(108,(19,53)),
                 'S6':(110,(19,53)),
                 'S7':(112,(19,53)),
                 'W1':(200,(19,45)),
                 'W2':(202,(19,45)),
                 'W3':(204,(19,45)),
                 'W4':(206,(19,45)),
                 'F0W1':(208,(19,45)),
                 'F0W2':(210,(19,45)),
                 'F0W3':(212,(19,45)) } 
    
    polarization = {'HH': np.array([1,0,0,0]),
                    'VV': np.array([0,1,0,0]),
                    'HV': np.array([0,0,1,0]),
                    'VH': np.array([0,0,0,1])
                    }
    
    available_bands ={ 'incidence_angle': 1,
                           'HH': 2,
                           'HV': 3,
                           'Valid Data Pixels':4,
                           'Unfiltered Seeds':5,
                           'Unfilt.Ext from energy HV':6,
                           'Final HV+TexEn Ext Mask BLO':7,
                           'Filtered Seeds':8,
                           'Filtered Extended':9
                         }
    
    def __init__(self,f_path,output_dir=None,gsw_path=None,images_output_dir=None,fst_converter_path=None):
        self.f_path=f_path
        self.output_dir=output_dir
        self.gsw_path = gsw_path
        self.images_output_dir=images_output_dir
        self.fst_converter_path=fst_converter_path
    
    def get_bands_infos(self):
        f_name=self.f_path
        self.base_name= os.path.basename(f_name)
        ds = gdal.Open(f_name)
        self.original_dataset = ds
        srs = osr.SpatialReference(gdal.Info(f_name, format='json')['coordinateSystem']['wkt'])
        src_srs=srs.ExportToProj4()
        invert_xy=srs.IsProjected()
        return ds,src_srs,invert_xy
    
    def get_bbox_coords(self,coords):
        min_lat = np.amin(coords[:,0])
        min_lon = np.amin(coords[:,1])
        max_lat = np.amax(coords[:,0])
        max_lon = np.amax(coords[:,1])

        return min_lat,min_lon,max_lat,max_lon
    
    def get_geotransform(self,lat0,lon0,dlat,dlon):
        print(f'Geotransform:',lat0,lon0,dlat,dlon)
        geotransform=[]
        geotransform.append(lon0)           # top left x
        geotransform.append(dlon)           # w-e pixel resolution
        geotransform.append(0)              # 0
        geotransform.append(round(lat0,3))  # top left y
        geotransform.append(0)              # 0
        geotransform.append(-dlat)          # n-s pixel resolution (negative value)
        return geotransform
    
    def to_geotiff(self,array,mask=None,grid_dims=None,f_name=None,geotransform=None,srs=None,gdal_type=gdal.GDT_Byte):
        if not f_name:
            f_name = f'{os.path.splitext(self.base_name)[0]}_a_priori_classification.tiff'
        print(self.images_output_dir,f_name)
        f_path = os.path.join(self.images_output_dir,f_name)
        print(f'File path: {f_path}')
        driver = gdal.GetDriverByName("GTiff")
        if not grid_dims:
            grid_dims=self.grid_dims
        output_file = driver.Create(f_path,grid_dims[1], grid_dims[0], 1, gdal_type)
        #output_file = driver.Create(f_path, self.grid_dims[1], self.grid_dims[0], 1, gdal.GDT_Byte)
        
        if not geotransform:
            geotransform = self.original_dataset.GetGeoTransform()
        output_file.SetGeoTransform(geotransform)
        
        if not srs:
            srs = self.original_dataset.GetProjection()
        output_file.SetProjection(srs)
        
        output_file.GetRasterBand(1).SetNoDataValue(255)
        if mask is not None:
            print('Setting invalid data to 255')
            print(f'Shapes\nMask : {mask.shape}\nData : {array.shape}')
            array[~mask]=255
            print('Done')
        print('Writing array')
        output_file.GetRasterBand(1).WriteArray(array)
        print('Flushing cache')
        output_file.FlushCache()
        output_file=None
        print('Done')
        
    def to_one_hot_beam(self,beam_mode_id):
        beams=list(self.beam_modes.values())
        beams.sort()
        return np.array([0 if b != beam_mode_id else 1 for b in beams])
    
    def get_beam_mode(self):
        base_name= os.path.basename(self.f_path)
        beam_mode_str = base_name.split('_')[1]
        #beam_mode_id = self.beam_modes[beam_mode_str][0]
        #one_hot_beam_mode=self.to_one_hot_beam(beam_mode_id)
        one_hot_beam_mode = np.array([0 if b != beam_mode_str else 1 for b in self.beam_modes])
        return one_hot_beam_mode
    
    def prepare_fst_info(self,pol_fst_array,target_resolution=200):
        """
        Taken from https://wiki.cmc.ec.gc.ca/wiki/Python-RPN/2.0/tutorial
        """
        ds,src_srs,invert_xy = self.get_bands_infos()
        coords,nb_pixels = self.get_coords_for_file(ds,invert_xy)
        gt = ds.GetGeoTransform()
        pixelSizeX = gt[1]
        pixelSizeY =-gt[5]
        resize_factor=target_resolution//pixelSizeX
        print(f'Coords size: {coords.shape[0]}')
        wgs84_coords=np.zeros((coords.shape[0],2),order='F')
        wgs84_coords[:,1],wgs84_coords[:,0]=SRIDConverter.convert_from_coordinates_check_geo(coords,src_srs)
        #min_lat,min_lon,max_lat,max_lon = self.get_bbox_coords(wgs84_coords)
        lat=wgs84_coords[:,0]
        lon=wgs84_coords[:,1]
        # Get data from the geotiff
        # mask => data == 255
        # add water mask
        mask = np.array(ds.GetRasterBand(self.available_bands['Valid Data Pixels']).ReadAsArray(),dtype=np.int8, order='FORTRAN')
        print(f'Mask shape: {mask.shape}')
        print(f'Lat shape: {lat.shape}')
        lat=lat.reshape(mask.shape)
        lon=lon.reshape(mask.shape)
        print(f'Lat shape: {lat.shape}')
        final_shape=(int(mask.shape[0]//resize_factor),int(mask.shape[1]//resize_factor))
        print(f'Final shape: {final_shape}\nResize Factor: {resize_factor}')
        print(type(final_shape[0]))
        rows_to_clip=mask.shape[0]%resize_factor
        cols_to_clip=mask.shape[1]%resize_factor
        start_offset_rows=int(math.ceil(rows_to_clip/2))
        end_offset_rows=int(rows_to_clip-start_offset_rows)
        start_offset_cols=int(math.ceil(cols_to_clip/2))
        end_offset_cols=int(cols_to_clip-start_offset_cols)
        #Get the lat,lon and final data based on target_resolution
        slice=np.s_[start_offset_rows:mask.shape[0]-end_offset_rows,start_offset_cols:mask.shape[1]-end_offset_cols]
        print('rebinning lat')
        print(f'Cropped lat shape: {lat[slice].shape}')
        print(lat)
        lat = bin_ndarray(lat[slice], final_shape, operation='avg')
        print('rebinning lon')
        lon = bin_ndarray(lon[slice], final_shape, operation='avg')
        print('rebinning mask')
        mask_=bin_ndarray(mask[slice], final_shape, operation='avg')
        mask_true=mask_>=0.5
        mask_false=mask_<0.5
        mask_[mask_true]=1
        mask_[mask_false]=0
        new_coords = np.hstack((lat.ravel()[:,np.newaxis],lon.ravel()[:,np.newaxis]))
        mask=np.array(mask,dtype=bool,order='FORTRAN')
        
        min_lat,min_lon,max_lat,max_lon = self.get_bbox_coords(new_coords)
        min_lat = min_lat
        min_lon = min_lon
        dlon = (max_lon-min_lon)/mask_.shape[1]
        dlat = (max_lat-min_lat)/mask_.shape[0]
        fst_dict = {'dlat':dlat,
                    'dlon':dlon,
                    'min_lat':min_lat,
                    'min_lon':min_lon,
                    'ni':mask_.shape[1],
                    'nj':mask_.shape[0],
                    'classification_mask':mask_,
                    'lats':lat,
                    'lons':lon
                    }
        
        for pol in pol_fst_array:
            print(f'Zeros array for {pol} [{mask.shape}]')
            pol_arr=np.zeros(mask.shape,dtype=np.float32)
            #print('Setting nodata')
            #pol_arr[~mask]=255
            print(f'Setting classification data\nMask:{mask.shape}\npol_fst_array: {pol_fst_array[pol].shape}')
            print(mask)
            pol_arr[mask]=np.array(pol_fst_array[pol],dtype=np.float32)
            print(f'Rebinning {pol}')
            pol_arr=bin_ndarray(pol_arr[slice],final_shape,operation='sum')
            pol_arr=np.array(np.round((pol_arr/(resize_factor**2))*100.,4),order='FORTRAN',dtype=np.float32)
            pol_arr[pol_arr>100]=9999.
            fst_dict[pol]=pol_arr
            
        f_path = os.path.join(self.fst_converter_path,
                              os.path.splitext(os.path.basename(self.f_path))[0])
        
        np.savez_compressed(f_path,
                            **fst_dict
                            )
        
        print('Min,max:')
        print(min_lat,min_lon,max_lat,max_lon)
        geotransform = self.get_geotransform(max_lat+dlat, min_lon+dlon, dlat, dlon)
        
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        projection = srs.ExportToWkt()
        
        f_name = 'test_binning_classif_percentage_water_HH.tiff'
        print(self.images_output_dir,f_name)
        self.to_geotiff(array=fst_dict['HH'], 
                        mask=np.array(mask_,dtype=bool), 
                        grid_dims=fst_dict['HH'].shape, 
                        f_name=f_name, 
                        geotransform=geotransform,
                        srs=projection,
                        gdal_type=gdal.GDT_Float32)
        
        return f_path,fst_dict
        
    def prepare_predict(self):
        #f_name = f"{os.path.splitext(os.path.basename(self.f_path))[0]}_{os.environ['LOOP_INDEX']}_predict.h5"
        #hdf5_path = os.path.join(self.output_dir,f_name)
        #hdf_writer = Hdf5Writer2(hdf5_path)
        #hdf_writer.open()
        beam_mode = self.get_beam_mode()
        ds,src_srs,invert_xy = self.get_bands_infos()
        coords,nb_pixels = self.get_coords_for_file(ds,invert_xy)
        wgs84_coords=np.zeros((coords.shape[0],2),order='F')
        wgs84_coords[:,1],wgs84_coords[:,0]=SRIDConverter.convert_from_coordinates_check_geo(coords,src_srs)
        self.coords=wgs84_coords
        min_lat,min_lon,max_lat,max_lon,water_presence=self.get_water_pixels()
        mask = np.array(ds.GetRasterBand(self.available_bands['Valid Data Pixels']).ReadAsArray(),dtype=bool)
        #hdf_writer.add_beam_mode(self.f_path,beam_mode)
        water_presence=water_presence.reshape(self.grid_dims)
        # Set as ambiguous pixels (anything that is >0 and <90
        water_presence[water_presence==255]=2
        # Mask invalid pixels
        water_presence[~mask]=255
        # A priori water presence
        self.to_geotiff(water_presence.reshape(self.grid_dims),mask=mask)
        #hdf_writer.add_mask(self.f_path,mask.ravel())
        del mask
        """
        hdf_writer.add_rs2_coords(self.f_path,self.coords,min_lat,max_lat,min_lon,max_lon,shape=self.grid_dims)
        for band in ('incidence_angle','HH','HV'):
            band_array = np.array(ds.GetRasterBand(self.available_bands[band]),dtype=np.float32)
            print(f'Adding {band} values')
            if band == 'incidence_angle':
                hdf_writer.add_incidence_angle(self.f_path,band_array.ravel())
            else:
                hdf_writer.add_pol(self.f_path,band,band_array.ravel(),self.polarization[band])
            print('Done')
        """
            
    def get_predict_data(self,index=0,num_procs=8,polarization='HH',water_weight=1):
        beam_mode = self.get_beam_mode()
        ds,src_srs,invert_xy = self.get_bands_infos()
        mask = np.array(ds.GetRasterBand(self.available_bands['Valid Data Pixels']).ReadAsArray(),dtype=bool)
        shape = mask.shape
        nb_pixels=mask[mask].size
        nb_pixels_per_task = math.ceil(nb_pixels/num_procs)
        #hdf_writer.add_mask(self.f_path,mask.ravel())
        start=index*nb_pixels_per_task
        end=start+nb_pixels_per_task
        dict_predict={}
        
        for band in ('incidence_angle',polarization):
            band_array = np.array(ds.GetRasterBand(self.available_bands[band]).ReadAsArray()[mask].ravel()[start:end],dtype=np.float32)
            dict_predict['beam_mode']=np.tile(beam_mode,(band_array.shape[0],1))
            dict_predict['weight']=np.tile(water_weight,(band_array.shape[0],))
            if band != 'incidence_angle':
                pol=np.tile(self.polarization[band],(band_array.shape[0],1))
                dict_predict['polarization']=pol
                dict_predict['value']=band_array
            else:
                dict_predict['incidence_angle']=band_array
            
            """
            'beam_mode':eval_data[0],
            'polarization':eval_data[1],
            'incidence_angle':eval_data[2].incidence_angle.ravel(),
            'value':eval_data[2].value.ravel(),
            'weight':self.water_weight*np.ones(eval_data[1].shape[0])
            """
        del mask
        return dict_predict
            
    def get_stats(self):
        f_name = f'{os.path.splitext(os.path.basename(self.f_path))[0]}_new.h5'
        hdf5_path = os.path.join(self.output_dir,f_name)
        hdf_writer = Hdf5Writer2(hdf5_path)
        hdf_writer.open()
        beam_mode = self.get_beam_mode()
        ds,src_srs,invert_xy = self.get_bands_infos()
        coords,nb_pixels = self.get_coords_for_file(ds,invert_xy)
        print(f'Coords size: {coords.shape[0]}')
        wgs84_coords=np.zeros((coords.shape[0],2),order='F')
        wgs84_coords[:,1],wgs84_coords[:,0]=SRIDConverter.convert_from_coordinates_check_geo(coords,src_srs)
        self.coords=wgs84_coords
        min_lat,min_lon,max_lat,max_lon,water_presence=self.get_water_pixels()
        print(min_lat,min_lon,max_lat,max_lon)
        print(water_presence)
        mask = np.array(ds.GetRasterBand(self.available_bands['Valid Data Pixels']).ReadAsArray(),dtype=bool)
        hdf_writer.add_beam_mode(self.f_path,beam_mode)
        #hdf_writer.add_rs2_coords(self.f_path,self.coords.reshape(self.grid_dims)[~mask],min_lat,max_lat,min_lon,max_lon,shape=self.grid_dims)
        #hdf_writer.add_mask(self.f_path,~mask)
        water_presence=water_presence.reshape(self.grid_dims)
        print(f'MASK: {mask.shape}')
        print(mask)
        print(f'Water Presence: {water_presence.shape}')
        print(water_presence)
        water_presence[~mask]=255
        self.to_geotiff(water_presence.reshape(self.grid_dims),mask=mask)
        del mask
        water_presence_idx = water_presence==1
        all_unambiguous_water_idx = water_presence==0
        idx_valid = water_presence != 255
        idx_water = water_presence[idx_valid] == 1
        coords_valid = water_presence.ravel() != 255
        print(f'Adding valid coordinates:{self.coords[coords_valid].shape}')
        hdf_writer.add_rs2_coords(self.f_path,self.coords[coords_valid],min_lat,max_lat,min_lon,max_lon,shape=self.grid_dims)
        print(f'Water Presence: {water_presence.shape}')
        hdf_writer.add_water(self.f_path,idx_water)
        
        del self.coords

        for band in ('incidence_angle','HH','HV'):
            band_array = np.array(ds.GetRasterBand(self.available_bands[band]).ReadAsArray()[idx_valid],dtype=np.float32)
            print(f'Adding {band} values')
            if band == 'incidence_angle':
                hdf_writer.add_incidence_angle(self.f_path,band_array)
            else:
                hdf_writer.add_pol(self.f_path,band,band_array,self.polarization[band])
                plt.figure()
                self.plot_histogram_data(ds,band,band_array,idx_water,~idx_water)
                plt.close()
            print('Done')
        
        
        
        #self.get_band_histogram(self.available_bands['HH'],name='HH')
        #self.get_band_histogram(self.available_bands['HV'], bins, name='HV')
  
    def plot_histogram3d_data(self,ds,band='HH',water_presence=None,all_unambiguous_water=None):
        f_name = f'{os.path.splitext(self.base_name)[0]}_histogram_{band}.png'
        f_path = os.path.join(self.output_dir,f_name)
        incidence_angle = ds.GetRasterBand(self.available_bands['incidence_angle']).ReadAsArray()
        band_data = ds.GetRasterBand(self.available_bands[band]).ReadAsArray()
        #hist, bins = np.histogram(band_data.ravel()[all_unambiguous_water], bins=2048)
        hist, xedges, yedges = np.histogram2d(band_data.ravel()[all_unambiguous_water], incidence_angle.ravel()[all_unambiguous_water], bins=2048)
        hist_water, xedges, yedges = np.histogram2d(band_data.ravel()[water_presence],incidence_angle.ravel()[water_presence], bins=(xedges,yedges))
        #plt.zscale('log', nonposz='clip')
        plt.xlim([-30,20])
        self.plot_histogram3d(hist, xedges, yedges)
        self.plot_histogram3d(hist_water, xedges, yedges)
        plt.title(f"Histogram {band}")
        plt.savefig(f_path, bbox_inches='tight')
        
    def plot_histogram_data(self,ds,band='HH',band_data=None,water_presence_idx=None,no_water_idx=None):
        f_name = f'{os.path.splitext(self.base_name)[0]}_histogram_{band}.png'
        f_path = os.path.join(self.images_output_dir,f_name)
        
        hist, bins = np.histogram(band_data[no_water_idx], bins=2048)
        hist_water,bins = np.histogram(band_data[water_presence_idx],bins=bins)
        plt.yscale('log', nonposy='clip')
        plt.xlim([-30,20])
        self.plot_histogram(hist, bins)
        self.plot_histogram(hist_water, bins)
        plt.title(f"Histogram {band}")
        plt.savefig(f_path, bbox_inches='tight')
        
    def plot_histogram(self,hist,bins,save=True):
        width = np.diff(bins)
        plt.bar(bins[:-1],hist,width=width)
    
    def plot_histogram3d(self,hist,xedges,yedges):
        xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
        xpos = xpos.flatten('F')
        ypos = ypos.flatten('F')
        zpos = np.zeros_like(xpos)
        dx = 0.5 * np.ones_like(zpos)
        dy = dx.copy()
        dz = hist.flatten()
        plt.bar3d(xpos, ypos, zpos, dx, dy, dz)

    def get_coords_for_file(self,ds,invert_xy=False):
        print('Get coords for file called')
        (upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = ds.GetGeoTransform()
        if not invert_xy:
            x = (np.arange(ds.RasterYSize,dtype=np.float32) * x_size + upper_left_x + (x_size/2.))
            y = (np.arange(ds.RasterXSize,dtype=np.float32) * y_size + upper_left_y + (y_size / 2.))
        else:
            x = (np.arange(ds.RasterXSize,dtype=np.float32) * x_size + upper_left_x + (x_size/2.))
            y = (np.arange(ds.RasterYSize,dtype=np.float32) * y_size + upper_left_y + (y_size / 2.))
        
        xs,ys=np.meshgrid(x,y)
        #ys,xs=np.meshgrid(y,x)
        print(xs)
        print(ys)
        print(xs.shape)
        self.grid_dims=xs.shape
        coords = np.hstack((xs.ravel()[:,np.newaxis],ys.ravel()[:,np.newaxis]))
        nb_pixels = ds.RasterYSize*ds.RasterXSize
        return coords,nb_pixels
    
    def get_band_histogram(self,band_number,bins=2048,name=None):
        plt.figure(figsize=(1024,768))
        b_name = f'{os.path.splitext(self.base_name)[0]}_histogram_{name}.png'
        band_data = ds.GetRasterBand(band_number).ReadAsArray()
        plt.hist(band_data, bins='auto')  # arguments are passed to np.histogram
        plt.title("Histogram")
        plt.savefig(os.path.join(self.output_dir,b_name), bbox_inches='tight')
        
    def get_water_pixels(self):
        interpolator = GSWInterpolator(sat_f_name=self.f_path,gsw_dir=self.gsw_path,output_dir=self.output_dir)
        min_lat,min_lon,max_lat,max_lon = self.get_bbox_coords(self.coords)
        print(min_lat,min_lon,max_lat,max_lon)
        return min_lat,min_lon,max_lat,max_lon,interpolator.get_water_presence_for_points(min_lat,max_lat,min_lon,max_lon,self.coords)
    
    def make_training_samples(self):
        pass
    
        
class RS2WideAndDeepNN:
    """
    By default random water/land pixels ratio between 20 and 60 percent
    
    6466911572 training pixels as a first batch for training
    - 646K steps of 10K pixels of training per "epoch"
    - Between 3 and 6 files picked at random for training:
        - Around 400M pixels per files
        - All water pixels are used for training
        - Land pixels according to randomly selected ratio [20-60]%
    - Complete learning from all pixels is > 646K steps because pixels 
      can be picked at random more than once.
    """
    def __init__(self,FLAGS=None,output_directory=None,input_directory=None,water_weight=1.0,model_type='wide_deep',num_procs=8):
        print('__INIT__ called')
        self.global_time=time.time()
        self.FLAGS=FLAGS
        self.information={}
        self.output_directory=output_directory
        self.input_directory=input_directory
        self.water_weight=water_weight
        self.data_feed=None
        self.model_type=model_type
        self.global_time=time.time()
        self.num_procs=num_procs
        
        
    def build_model_columns(self):
        """
        Beam_mode,polarity,incidence angle,value,classification
        """
        print('Building model columns')
        
        beam_mode = tf.feature_column.numeric_column('beam_mode',shape=(len(PixStats.beam_modes),))
        polarization= tf.feature_column.numeric_column('polarization',shape=(4,))
        incidence_angle = tf.feature_column.numeric_column('incidence_angle')
        value = tf.feature_column.numeric_column('value')
        
        base_columns=[beam_mode,polarization,incidence_angle,value]
        crossed_columns = [
                          tf.feature_column.crossed_column(
                              ['beam_mode', 'polarization'], hash_bucket_size=1000),
                          tf.feature_column.crossed_column(
                              ['beam_mode', 'polarization','incidence_angle'], hash_bucket_size=10000),
                          tf.feature_column.crossed_column(
                              ['polarization','incidence_angle'], hash_bucket_size=10000),
                          tf.feature_column.crossed_column(
                              ['beam_mode', 'incidence_angle'], hash_bucket_size=10000)
                    ]
        
        d_cols = [tf.feature_column.embedding_column(col,10) for col in crossed_columns]

        wide_columns = base_columns + crossed_columns
        deep_columns = base_columns + d_cols
        
        weight_column = tf.feature_column.numeric_column('weight')
        
        return wide_columns, deep_columns,weight_column
    
    def build_estimator(self,model_dir, model_type):
        """Build an estimator appropriate for the given model type.
        types : 'wide', 'deep', 'wide_deep'
        """
        
        def metric_auc(labels, predictions):
            return {
                'auc_precision_recall': tf.metrics.auc(
                    labels=labels, predictions=predictions['logistic'], num_thresholds=200,
                    curve='PR', summation_method='careful_interpolation')
            }

        def convert_predictions(predictions):
            predictions=tf.string_to_number(predictions['classes'],out_type=tf.int32)
            return predictions
        
        def metric_prec(features, labels, predictions):
            predictions=convert_predictions(predictions)
            #np.array([int(eval(r[0].decode('utf-8'))) for r in predictions['classes']])
            return {'precision':tf.metrics.precision(labels,predictions)}
            
        def metric_acc(features, labels, predictions):
            predictions=convert_predictions(predictions)
            #predictions=np.array([int(eval(r[0].decode('utf-8'))) for r in predictions['classes']])
            return {'accuracy':tf.metrics.accuracy(labels,predictions)}

        def metric_recall(features, labels, predictions):
            predictions=convert_predictions(predictions)
            #predictions=np.array([int(eval(r[0].decode('utf-8'))) for r in predictions['classes']])
            return {'recall':tf.metrics.recall(labels,predictions)}
        
        def metric_f1(features, labels, predictions):
            predictions=convert_predictions(predictions)
            #predictions=np.array([int(eval(r[0].decode('utf-8'))) for r in predictions['classes']])
            P, update_op1 = tf.metrics.precision(labels,predictions)
            R, update_op2 = tf.metrics.recall(labels,predictions)
            eps = 1e-5;
            return {'f1_score':(2*(P*R)/(P+R+eps), tf.group(update_op1, update_op2))}
        
        def metric_fn(features,labels,predictions):
            predictions=convert_predictions(predictions)
            return {'false_negatives': tf.metrics.false_negatives(labels,predictions)}
        
        def metric_tn(features,labels,predictions):
            predictions=convert_predictions(predictions)
            return {'true_negatives': tf.metrics.true_negatives(labels,predictions)}
        
        def metric_tp(features,labels,predictions):
            predictions=convert_predictions(predictions)
            return {'true_positives': tf.metrics.true_positives(labels,predictions)}
        
        def metric_fp(features,labels,predictions):
            predictions=convert_predictions(predictions)
            return {'false_positives': tf.metrics.false_positives(labels,predictions)}
        
        wide_columns, deep_columns,weight_column = self.build_model_columns()
        
        # Let's try from some sources:
        # Number of hidden layers:
        # 0 : only linearly separable problems
        # 1 : any function that contains a continuous mapping
        #     from one finite space to another.
        # 2 : arbitrary decision boundary to arbitrary accuracy
        #     with rational activation functions and can approximate any smooth
        #     mapping to any accuracy.
        #
        # Let's use 2 layers.
        #
        # Lots of rule of thumb in the number of neurons per layer
        # - Between the size of input and output
        # - 2/3 of input neurons 
        # - not more than twice the number of input neurons
        # - Upper bound = number of independant samples / (alpha*(input+output neurons) alpha between 2 and 10
        #   In this case between 4 and ?
        # - Too much neurons = overfitting (1 path through the network for each training sample = memorization) 
        # - Not enough = underfitting
        # Let's try a few times number of input layer since it's a rather low number:
        hidden_units = [24,24]
        
        
        # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
        # trains faster than GPU for this model.
        run_config = tf.estimator.RunConfig().replace(
            session_config=tf.ConfigProto(device_count={'GPU': 0}))
        #run_config=run_config.replace(save_summary_steps=1000000000,save_checkpoints_secs=600)
        run_config=run_config.replace(save_summary_steps=1000000000)
        
        if model_type == 'wide':
            print('Building LinearClassifier')
            model= tf.estimator.LinearClassifier(
                model_dir=model_dir,
                feature_columns=wide_columns,
                config=run_config)
            self.information['Model']='LinearClassifier'
            self.information['Model Parameters']='Default'
        elif model_type == 'deep':
            print('Building DNNClassifier')
            self.information['Model']='DNNClassifier'
            self.information['Model Parameters']='Default'
            model= tf.estimator.DNNClassifier(
                model_dir=model_dir,
                feature_columns=deep_columns,
                hidden_units=hidden_units,
                config=run_config)
        else:
            print('Building DNNLinearCombinedClassifier')
            self.information['Model']='DNNLinearCombinedClassifier'
            model = tf.estimator.DNNLinearCombinedClassifier(
                model_dir=model_dir,
                linear_feature_columns=wide_columns,
                linear_optimizer=tf.train.FtrlOptimizer(
                    learning_rate=0.01,
                    l1_regularization_strength=0.001,
                    l2_regularization_strength=0.001
                    ),
                dnn_feature_columns=deep_columns,
                dnn_hidden_units=hidden_units,
                dnn_dropout=0.5,
                dnn_optimizer=tf.train.ProximalAdagradOptimizer(
                        learning_rate=0.01,
                        l1_regularization_strength=0.001,
                        l2_regularization_strength=0.001),
                weight_column=weight_column,
                config=run_config)
            self.information['Linear Optimizer']='Ftrl Optimizer'
            self.information['Linear Optimizer Learning Rate']=0.001
            self.information['Linear Optimizer L1 Regularization']=0.001
            self.information['Linear Optimizer L2 Regularization']=0.001
            self.information['Hidden Units']=hidden_units
            self.information['DNN Dropout']=0.5
            self.information['DNN Optimizer']='ProximalAdagradOptimizer'
            self.information['DNN Optimizer Learning Rate']=0.001
            self.information['DNN Optimizer L1 Regularization']=0.001
            self.information['DNN Optimizer L2 Regularization']=0.001
            self.information['Water Weight Value']=self.water_weight

        model = tf.contrib.estimator.add_metrics(model, metric_prec)
        model = tf.contrib.estimator.add_metrics(model, metric_f1)
        model = tf.contrib.estimator.add_metrics(model, metric_recall)
        model = tf.contrib.estimator.add_metrics(model, metric_acc)
        model = tf.contrib.estimator.add_metrics(model, metric_tp)
        model = tf.contrib.estimator.add_metrics(model, metric_tn)
        model = tf.contrib.estimator.add_metrics(model, metric_fp)
        model = tf.contrib.estimator.add_metrics(model, metric_fn)
        model = tf.contrib.estimator.add_metrics(model, metric_auc)
          
          
        # Information about the network for eval purposes
        # These Should all be config data or input commandline
        # Must be changed each time a modification to the network is done 
          
          
        return model
    
    def get_predict_fn_for_file(self,file_path,pol,index):
        reader = PixStats(f_path=file_path, output_dir=None, gsw_path=None, images_output_dir=None, fst_converter_path=None)
        features = reader.get_predict_data(index=index,num_procs=self.num_procs,polarization=pol,water_weight=self.water_weight)
        return tf.estimator.inputs.numpy_input_fn(x=features,batch_size=10000,shuffle=False,num_epochs=1)
        
    def get_input_fn_for_file(self,mode=ModelModes.TRAIN):
        if not self.data_feed:
            self.data_feed = HDF5Feeder(hdf_directory=self.input_directory,global_time=self.global_time)
        
        def input_gen_fn():
            try:
                for source in input_gen:
                    for feats,labels in source:
                        yield feats,labels
            except ValueError:
                for feat,labels in source:
                    yield feats,labels
        
        if mode == ModelModes.TRAIN:
            input_gen = self.data_feed.get_batch_generator(self.water_weight)
            ds=tf.data.Dataset.from_generator(
             generator=input_gen_fn,
             output_types=( (tf.int8,tf.int8,tf.float32,tf.float32,tf.float32),tf.int8),
             output_shapes=(((len(PixStats.beam_modes),),(4,),(),(),()) ,())
            )
        elif mode in (ModelModes.EVAL,ModelModes.PREDICT):
            if mode == ModelModes.EVAL:
                eval_data = self.data_feed.get_random_eval_data()
            else:
                eval_data = self.data_feed.get_random_test_data()
                # source[0]: beam_mode <scalar>
                # source[1]: pol one hot , <col>
                # source[2]: sar_data <recarray> 'incidence_angle','value','classification'
            features ={
                            'beam_mode':eval_data[0],
                            'polarization':eval_data[1],
                            'incidence_angle':eval_data[2].incidence_angle.ravel(),
                            'value':eval_data[2].value.ravel(),
                            'weight':self.water_weight*np.ones(eval_data[1].shape[0])
                      }
            print('*************')
            print(features)
            labels = eval_data[2].classification.ravel()
            print('------------')
            print(labels)
            print('Got eval data, returning')
            return tf.estimator.inputs.numpy_input_fn(x=features,y=labels,batch_size=10000,shuffle=False,num_epochs=1)
        
        
        def to_dict(feats,labels):
            keys=('beam_mode','polarization','incidence_angle','value','weight')
            d_feats={}
            #d_feats['beam_mode']=feats[:len(PIXFileConverter2.beam_modes)]
            #d_feats['polarization']=feats[len(PIXFileConverter2.beam_modes):len(PIXFileConverter2.beam_modes)+4]
            d_feats=dict(zip(keys,feats))
            #d_feats['weight']=tf.float32(self.water_weight)
            return d_feats,labels
        
        #
        
        
        #print('map to_dict')
        #ds = ds.map(to_dict,num_parallel_calls=30)
        #print('map and batch')
        #ds = ds.batch(1000)
        ds = ds.apply(tf.contrib.data.map_and_batch(
                        map_func=to_dict,
                        batch_size=10000,
                        num_parallel_calls=30)
                    )
        print('prefetch')
        ds = ds.prefetch(10000)
        iter = ds.make_one_shot_iterator()
        print('Returning features and labels')
        features,labels = iter.get_next()
        
        return features,labels
    
    def get_highest_nn(self,nn_type='backup'):
        if nn_type == 'backup':
            try:   
                highest_nn = max(int(num) 
                                       for p in os.scandir(self.output_directory) if p.is_dir and p.name.startswith('NN') 
                                       for num in p.name.split('_') if num != 'NN'
                                       )+1
            except Exception as exc:
                print(type(exc))
                print(exc)
                highest_nn = 1
        elif nn_type == 'eval':
            try:
                highest_nn = max(int(os.path.splitext(p.name)[0].split('_')[-1])
                                    for p in os.scandir(self.output_directory) 
                                        if p.is_file() and p.name.endswith('.txt') 
                                        and p.name.startswith(f'training_{self.model_type}')
                                )+1
            except Exception as exc:
                print(type(exc))
                print(exc)
                highest_nn = 1
        return highest_nn
        
    
    def train(self):
        model_dir = os.path.join(self.output_directory,'NN')
        model=self.build_estimator(model_dir,self.model_type)
            
        input_fn = lambda:self.get_input_fn_for_file()
        
        model.train(input_fn=input_fn)
        
        higest_backup_NN = self.get_highest_nn('backup')
        backup_nn_path = os.path.join(self.output_directory,f'NN_{higest_backup_NN}')
        print(f'Making copy of ANN to {backup_nn_path}')
        try:
            shutil.copytree(model_dir,backup_nn_path)
        except Exception as ex:
            print(type(ex))
            print(ex)
        try:
            self.clean_model_dir(model_dir)
        except Exception as exc:
            print(type(exc))
            print(exc)
            
            
    def merge_predict(self,file_path=None,a_priori_dir=None):
        a_priori_gtiff = f'{os.path.splitext(os.path.basename(file_path))[0]}_a_priori_classification.tiff'
        a_priori_gtiff_path = os.path.join(a_priori_dir,a_priori_gtiff)
        ds = gdal.Open(a_priori_gtiff_path)
        
    def predict(self,output=None,index=None,file_path=None):
        # Get the model
        model_dir = os.path.join(self.output_directory)
        model=self.build_estimator(model_dir,self.model_type)

        #Get the input layer data, treat polarization separately
        results = None
        for pol in ('HH','HV'):
            print(80*'-')
            print(f'Polarization: {pol}')
            input_fn = self.get_predict_fn_for_file(file_path, pol, index)
            """
            if results is None:
                results = model.predict(input_fn=input_fn)
            else:
                # Keep only when both polarizations detect water:
                results = np.array(np.logical_and(
                                        results,
                                        np.array([bool(eval(r['classes'][0].decode('utf-8'))) for r in results],dtype=np.int8)
                                        ),
                          dtype=np.int8)
            """
            print('Making predict')
            results = model.predict(input_fn=input_fn)
            print('Prediction done, making int8 array')
            results=np.array([bool(eval(r['classes'][0].decode('utf-8'))) for r in results],dtype=np.int8)
            print('int8 array done')
            f_out=f'{os.path.splitext(os.path.basename(file_path))[0]}_classification_{pol}_{index}'
            predict_f_path = os.path.join(os.getcwd(),'predict',f_out)
            print(f'Saving to {predict_f_path}')
            #Just in case in the end we put more arrays there we use npz so the rest of the process won't change
            np.savez(predict_f_path,classification=results)
        
        
        
    def get_evaluate_string(self,results):
        return ''.join((
                            '\n'.join(f'{k:<35}:{self.information[k]}' for k in self.information),
                            f"\n{80*'-'}\n",
                            '\n'.join(f'{k:<35}:{results[k]}' for k in sorted(results))
                      ))
    
    def save_evaluation_results(self,results,index_NN=None,print_evaluate=True):
        eval_string = self.get_evaluate_string(results)
        f_name = f"training_{self.model_type}_{results['global_step']}_{index_NN}.txt"
        with open(os.path.join(self.output_directory,f_name),'w') as fo:
            fo.write(eval_string)
        if print_evaluate:
            print(eval_string)
            
    def eval(self):
        index_nn=self.get_highest_nn('eval')
        model_dir = os.path.join(self.output_directory,'NN')
        model=self.build_estimator(model_dir,self.model_type)
        input_fn = self.get_input_fn_for_file(mode=ModelModes.EVAL)
        results = model.evaluate(input_fn=input_fn)
        nb_water_pixels = results['true_positives']+results['false_negatives']
        nb_land_pixels = results['true_negatives']+results['false_positives']
        self.information['Evaluation File Water Pixels']=nb_water_pixels
        self.information['Evaluation File Land Pixels']=nb_land_pixels
        self.information['Evaluation File Total Pixel']=nb_water_pixels+nb_land_pixels
        self.information['Evaluation File Water Pixel Ratio']=nb_water_pixels/(nb_water_pixels+nb_land_pixels)
        self.information['Evaluation File Land Pixel Ratio']=nb_land_pixels/(nb_water_pixels+nb_land_pixels)
        self.information['Global Steps']=results['global_step']
        self.information['Step Size']=10000
        self.information['Total Trained Pixels']=10000*int(results['global_step'])
        results['False To True Water Detection Ratio']=results['false_positives']/results['true_positives']
        results['False To True Land Detection Ratio']=results['false_negatives']/results['true_negatives']
        results['Land Pixels Detection Ratio']=results['true_negatives']/(results['false_positives']+results['true_negatives'])
        self.save_evaluation_results(results,index_nn,print_evaluate=True)
       
        
    def clean_model_dir(self,path=None):
        if path is None:
            path = os.path.join(self.output_directory,'NN')
        model_files=[f for f in os.scandir(path)]
        consume(os.unlink(f) for f in model_files if f.name.startswith('events.out'))
        try:
            shutil.rmtree(os.path.join(path,'eval'))
        except:
            pass
    

    
    
